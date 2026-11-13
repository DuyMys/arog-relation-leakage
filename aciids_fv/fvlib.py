"""Core library for the ACIIDS 2027 fixed-view (Protocol B) leakage audit.

Everything here works on SYNTHETIC Synthea data. Remote-facing text never contains
patient UUIDs, condition/city nodes, original codes or descriptions. Labels live in a
separate PRIVATE file that is joined only at evaluation time.
Test/reserve splits are locked: they need a PREREG_LOCK.json whose hashes still match.
"""
import hashlib, hmac, json, random, re, math
from pathlib import Path
import numpy as np
import pandas as pd

CLASS_NAMES = {
    "T2D": "Type 2 diabetes mellitus", "AsthmaChild": "Childhood asthma",
    "Appendicitis": "Appendicitis", "Seizure": "Seizure disorder",
    "HLD": "Hyperlipidemia", "PolypColon": "Polyp of colon",
}
CLASSES = list(CLASS_NAMES)
QUERY = "Summarize the available clinical relationship types for the anonymized subject."
LEVELS = ("L0", "L1", "L2")
LOCKED_SPLITS = {"test", "reserve"}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def hmac_hex(secret, *parts):
    return hmac.new(secret, "|".join(str(p) for p in parts).encode(), hashlib.sha256).hexdigest()


def load_config(path, base=None):
    cfg = json.loads(Path(path).read_text())
    root = Path(base) if base else Path(path).resolve().parent / cfg["base_dir"]
    cfg["_root"] = root.resolve()
    for k in ("kg_dir", "clean_dir", "secret_file", "mapping_file", "split_manifest", "banned_tokens"):
        cfg[k] = str(cfg["_root"] / cfg[k])
    cfg["_config_path"] = str(Path(path).resolve())
    return cfg


def load_secret(cfg):
    key = bytes.fromhex(Path(cfg["secret_file"]).read_text().strip())
    assert len(key) >= 32
    return key


def load_kg(cfg, check_mapping=True):
    if check_mapping:
        got = sha256_file(cfg["mapping_file"])
        assert got == cfg["mapping_sha256"], f"Mapping hash mismatch: {got}"
    nodes = pd.read_csv(Path(cfg["kg_dir"]) / "nodes.csv")
    tri = pd.read_csv(Path(cfg["kg_dir"]) / "triples.csv").rename(
        columns={"head": "h", "tail": "tl", "relation_id": "rel"})
    mp = pd.read_csv(cfg["mapping_file"], dtype=str).set_index("node_id")
    split = pd.read_csv(cfg["split_manifest"], dtype=str)
    assert set(split.patient_mid) <= set(nodes.node_id), "split MIDs not in KG"
    edges = {}
    for h, tl, kind, cnt in tri.loc[tri.kind.isin(["medication", "procedure"]),
                                    ["h", "tl", "kind", "event_count"]].itertuples(index=False):
        edges.setdefault(h, []).append((tl, kind, int(cnt)))
    return dict(nodes=nodes, tri=tri, mapping=mp, split=split, edges=edges)


def load_banned(path):
    rules = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        kind, _, val = line.partition("\t")
        if kind == "LITERAL":
            rules.append(re.compile(r"(?<![A-Za-z0-9])" + re.escape(val.strip()) + r"(?![A-Za-z0-9])", re.I))
        elif kind == "REGEX":
            rules.append(re.compile(val.strip(), re.I))
    return rules


def lint(text, rules):
    return [r.pattern for r in rules if r.search(text)]


def require_unlock(split_names, prereg_path):
    """Test/reserve may only be touched with a PREREG_LOCK.json whose hashes still match."""
    if not (set(split_names) & LOCKED_SPLITS):
        return
    if not prereg_path:
        raise SystemExit("LOCKED: test/reserve need --prereg PREREG_LOCK.json (run make_prereg.py first).")
    lock = json.loads(Path(prereg_path).read_text())
    bad = []
    for rel, want in lock["artifact_sha256"].items():
        p = (Path(lock["code_dir"]) / rel.split("/", 1)[1]) if rel.startswith(("__code__/", "__dev__/")) else (Path(lock["root"]) / rel)
        if not p.is_file() or sha256_file(p) != want:
            bad.append(rel)
    if bad:
        raise SystemExit(f"LOCK BROKEN, artifacts changed after freeze: {bad}")
    print(f"[unlock] PREREG verified ({len(lock['artifact_sha256'])} artifacts, locked {lock['locked_utc']})")


# ----------------------------------------------------------------- views
def select_edges(edges, policy, secret, patient_mid, k):
    out = []
    for kind in ("medication", "procedure"):
        es = [e for e in edges if e[1] == kind]
        if policy == "topcount15":
            es = sorted(es, key=lambda r: (-r[2], r[0]))[:k]
        elif policy == "random15":
            rng = random.Random(int(hmac_hex(secret, "sel", patient_mid, kind)[:16], 16))
            es = sorted(es, key=lambda r: r[0])
            rng.shuffle(es)
            es = sorted(es[:k], key=lambda r: r[0])
        else:
            raise ValueError(policy)
        out.extend(es)
    return out


def build_views(kg, cfg, secret, splits, policy):
    """Return (records, labels). Records are remote-facing; labels are PRIVATE."""
    sp = kg["split"][kg["split"].split.isin(splits)]
    recs, labs = [], []
    k = cfg["edges_per_modality"]
    for p in sp.itertuples(index=False):
        kept = select_edges(kg["edges"].get(p.patient_mid, []), policy, secret, p.patient_mid, k)
        canon = hashlib.sha256("|".join(t for t, _, _ in kept).encode()).hexdigest()[:16]
        for lv in LEVELS:
            rid = "R_" + hmac_hex(secret, "rid", p.patient_mid, policy, lv)[:16]
            subj = "P_" + hmac_hex(secret, "subj", rid)[:12]
            alias = {t: "E_" + hmac_hex(secret, "obj", rid, t)[:12] for t, _, _ in kept}
            assert len(set(alias.values())) == len(alias)
            trip = [[subj, kg["mapping"].at[t, lv], alias[t]] for t, _, _ in kept]
            recs.append(dict(record_id=rid, level=lv, policy=policy, query=QUERY, subject=subj, triples=trip))
            labs.append(dict(record_id=rid, patient_mid=p.patient_mid, class_key=p.class_key, split=p.split,
                             level=lv, policy=policy, n_edges=len(kept), canon_edge_hash=canon))
    order = sorted(range(len(recs)), key=lambda i: hmac_hex(secret, "order", recs[i]["record_id"]))
    return [recs[i] for i in order], pd.DataFrame([labs[i] for i in order])


def render_remote(rec, view="triples", variant="plain", secret=None):
    lines = ["KGQA REQUEST (anonymized)", f"Question: {rec['query']}", f"Subject: {rec['subject']}"]
    if variant == "c0":
        return "\n".join(lines)
    trip = rec["triples"]
    if variant == "c2tok":
        rm = {r: "REL_" + hmac_hex(secret, "c2", rec["record_id"], r)[:6] for r in sorted({t[1] for t in trip})}
        trip = [[h, rm[r], t] for h, r, t in trip]
    if view == "names":
        lines.append("Available relation types for the subject:")
        lines += [f"- {r}" for r in sorted({t[1] for t in trip})]
    else:
        lines.append("Evidence triples (subject | relation | object):")
        lines += [f"{h} | {r} | {t}" for h, r, t in trip]
    return "\n".join(lines)


# -------------------------------------------------------------- attacker
SYSTEM = ("You are an evaluator in a privacy-risk experiment on synthetic data. You will see the content of an "
          "anonymized request sent to a remote language model. Infer which ONE candidate condition class is most "
          "likely the subject's primary recorded condition, using only the request content. Identifiers such as "
          "P_xxx or E_xxx are random and carry no meaning. If the content gives no signal, still choose the most "
          "plausible class. Answer with JSON only: {\"prediction\": \"<exact class name>\", "
          "\"confidence\": <0-1>, \"evidence\": \"<one short sentence>\"}.")


def attacker_prompt(remote_text, record_id, secret, note=None):
    order = sorted(CLASSES, key=lambda c: hmac_hex(secret, "cand", record_id, c))
    cand = "\n".join(f"- {CLASS_NAMES[c]}" for c in order)
    user = (f"Candidate classes (exactly one is correct):\n{cand}\n\nObserved remote request content:\n"
            f"\"\"\"\n{remote_text}\n\"\"\"\n")
    if note:
        user += f"\nLocal evaluator note (not part of any KGQA traffic): {note}\n"
    return SYSTEM, user + "\nReturn JSON only.", order


_NORM = {re.sub(r"[^a-z0-9]", "", v.lower()): k for k, v in CLASS_NAMES.items()}
_NORM.update({re.sub(r"[^a-z0-9]", "", k.lower()): k for k in CLASSES})


def parse_prediction(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None, None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None, None
    key = _NORM.get(re.sub(r"[^a-z0-9]", "", str(obj.get("prediction", "")).lower()))
    try:
        conf = float(obj.get("confidence"))
    except Exception:
        conf = None
    return key, conf


# ------------------------------------------------------------- statistics
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def balanced_accuracy(y, p, classes=CLASSES):
    y, p = np.asarray(y), np.asarray(p)
    r = [(p[y == c] == c).mean() for c in classes if (y == c).any()]
    return float(np.mean(r))


def macro_f1(y, p, classes=CLASSES):
    y, p = np.asarray(y), np.asarray(p)
    f = []
    for c in classes:
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); fn = ((p != c) & (y == c)).sum()
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def boot_ci(correct, B=10000, seed=20261106):
    rng = np.random.default_rng(seed)
    c = np.asarray(correct, float)
    idx = rng.integers(0, len(c), size=(B, len(c)))
    m = c[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def mcnemar_exact(a, b):
    from scipy.stats import binomtest
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01 = int((~a & b).sum()); n10 = int((a & ~b).sum())
    n = n01 + n10
    return 1.0 if n == 0 else float(binomtest(n01, n, 0.5).pvalue)


def paired_boot_diff(a, b, B=10000, seed=20261106):
    rng = np.random.default_rng(seed)
    d = np.asarray(a, float) - np.asarray(b, float)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    m = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i]); adj[i] = min(1.0, run)
    return adj.tolist()
