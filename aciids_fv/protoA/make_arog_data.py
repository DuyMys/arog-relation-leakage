#!/usr/bin/env python3
"""Protocol A step 1: project the Synthea KG into a Freebase-style N-Triples file + a WebQSP-format question file,
so that the UNMODIFIED ARoG pipeline (main_freebase.py --dataset webqsp) runs on it through Virtuoso/SPARQL.

Each (patient, level) gets its own session-local subject MID => L0/L1/L2 are separate, non-linkable graphs
that can be loaded into ONE Virtuoso store. Edge set per patient = same top-count/random policy as the fixed-view toolkit.
Outputs (outdir):  FilterFreebase  WebQSP.json  records_PRIVATE.csv (labels, never given to ARoG or the attacker)
"""
import argparse, json, sys, hashlib, re
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fvlib as F

NS = "http://rdf.freebase.com/ns/"
QF = {"medication": "Which medications are recorded for this patient?",
      "procedure": "Which procedures are recorded for this patient?"}


def slug(x):
    return re.sub(r"[^a-z0-9_]", "_", str(x).lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="../fv_config.json"); ap.add_argument("--base")
    ap.add_argument("--split", default="dev"); ap.add_argument("--levels", default="L0,L1,L2")
    ap.add_argument("--policy"); ap.add_argument("--limit_patients", type=int)
    ap.add_argument("--outdir", required=True); ap.add_argument("--prereg")
    ap.add_argument("--skip_mapping_hash", action="store_true", help="TEST FIXTURES ONLY")
    a = ap.parse_args()
    cfg = F.load_config(a.config, a.base); splits = a.split.split(","); levels = a.levels.split(",")
    F.require_unlock(splits, a.prereg)
    policy = a.policy or cfg["primary_policy"]
    kg = F.load_kg(cfg, check_mapping=not a.skip_mapping_hash); secret = F.load_secret(cfg)
    rules = F.load_banned(cfg["banned_tokens"]); k = cfg["edges_per_modality"]
    sp = kg["split"][kg["split"].split.isin(splits)].sort_values("patient_mid")
    if a.limit_patients:   # balanced: take the first n/6 of each class
        per = max(1, a.limit_patients // 6)
        sp = sp.groupby("class_key", group_keys=False).head(per)
    nt, qs, labs = [], [], []
    for p in sp.itertuples(index=False):
        kept = F.select_edges(kg["edges"].get(p.patient_mid, []), policy, secret, p.patient_mid, k)
        fam = "medication" if int(F.hmac_hex(secret, "qf", p.patient_mid)[:8], 16) % 2 == 0 else "procedure"
        for lv in levels:
            rid = "R_" + F.hmac_hex(secret, "rid", p.patient_mid, policy, lv)[:16]
            subj = "m.p" + F.hmac_hex(secret, "subj", rid)[:12]
            ans = []
            nt.append(f'<{NS}{subj}> <{NS}type.object.name> "{subj}" .')
            for t, kind, cnt in kept:
                obj = "m.e" + F.hmac_hex(secret, "obj", rid, t)[:12]
                rel = "clinical." + slug(kg["mapping"].at[t, lv])
                nt.append(f"<{NS}{subj}> <{NS}{rel}> <{NS}{obj}> .")
                nt.append(f'<{NS}{obj}> <{NS}type.object.name> "{obj}" .')
                if kind == fam:
                    ans.append(obj)
            ans = sorted(set(ans))
            qs.append(dict(QuestionId=rid, RawQuestion=QF[fam], topic_entity={subj: subj},
                           Parses=[dict(Answers=[dict(EntityName=o, AnswerArgument=o) for o in ans])]))
            labs.append(dict(record_id=rid, patient_mid=p.patient_mid, class_key=p.class_key, split=p.split, level=lv,
                             policy=policy, qfamily=fam, subj=subj, n_edges=len(kept), n_answers=len(ans)))
    hits = [(h, x) for x in set(re.findall(r"<[^>]+ns/(clinical\.[^>]+)>", "\n".join(nt))) | {q["RawQuestion"] for q in qs}
            for h in F.lint(x, rules)]
    if hits:
        raise SystemExit(f"LINT HITS, refusing to write: {hits[:5]}")
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    (out / "FilterFreebase").write_text("\n".join(nt) + "\n", encoding="utf-8")
    (out / "WebQSP.json").write_text(json.dumps(qs, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(labs).to_csv(out / "records_PRIVATE.csv", index=False)
    rels = sorted({m for m in re.findall(r"ns/(clinical\.[a-z0-9_]+)>", "\n".join(nt))})
    print(f"records={len(qs)} patients={sp.shape[0]} triples={len(nt)} relation_labels={len(rels)} lint=PASS")


if __name__ == "__main__":
    main()
