#!/usr/bin/env python3
"""Statistical attacker (A3) on the relation-label SET that the REAL ARoG transcript exposes.
Features = presence of each relation label parsed from relation_search_prune prompts where the topic entity is the
record's own subject (first-hop full list, before pruning). Trained on TRAIN views only (never on dev/test).
Also checks, per record, that the label set seen in the transcript equals the label set of the fixed view.
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fvlib as F
from stat_attackers import centroid_fit_predict, logreg_fit_predict, score

PAT = re.compile(r"Topic Entity: (m\.p[0-9a-f]+)\nRelations: ([^\n]*)")


def label_sets(transcripts, records, stage="relation_search_prune"):
    subj = records.set_index("record_id")["subj"].to_dict()
    out = {}
    for l in Path(transcripts).read_text().splitlines():
        d = json.loads(l)
        if d["stage"] != stage or d["qid"] not in subj:
            continue
        for s, rels in PAT.findall(d["prompt"]):
            if s == subj[d["qid"]]:
                out.setdefault(d["qid"], set()).update(r.strip().replace("clinical.", "", 1) for r in rels.split(";") if r.strip())
    return out


def presence(sets_by_id, vocab=None):
    rows = [dict(record_id=k, **{v: 1 for v in s}) for k, s in sets_by_id.items()]
    X = pd.DataFrame(rows).fillna(0).set_index("record_id")
    return X if vocab is None else X.reindex(columns=vocab, fill_value=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", required=True); ap.add_argument("--records", required=True)
    ap.add_argument("--train_views", required=True); ap.add_argument("--train_labels", required=True)
    ap.add_argument("--dev_views"); ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="relation_search_prune"); ap.add_argument("--prereg")
    ap.add_argument("--perm", type=int, default=1000, help="C-: refits with permuted TRAIN labels (primary attacker, L0/L1)")
    a = ap.parse_args()
    rec = pd.read_csv(a.records); F.require_unlock(set(rec.split), a.prereg)
    Ltr = pd.read_csv(a.train_labels); assert set(Ltr.split) <= {"train", "validation"}
    Rtr = [json.loads(l) for l in Path(a.train_views).read_text().splitlines()]
    tr_sets = {r["record_id"]: {t[1] for t in r["triples"]} for r in Rtr}
    ts = label_sets(a.transcripts, rec, a.stage)
    print(f"records with parsed label set: {len(ts)}/{len(rec)}")
    if a.dev_views:
        Rdv = {r["record_id"]: {t[1] for t in r["triples"]} for r in map(json.loads, Path(a.dev_views).read_text().splitlines())}
        common = [k for k in ts if k in Rdv]
        eq = np.mean([ts[k] == Rdv[k] for k in common]) if common else float("nan")
        sub = np.mean([ts[k] <= Rdv[k] for k in common]) if common else float("nan")
        print(f"label set == fixed-view set: {eq:.3f}  (subset: {sub:.3f}, n={len(common)})")
    res = []; perm = {}
    for lv in F.LEVELS:
        trid = Ltr[Ltr.level == lv].record_id.tolist()
        Xtr = presence({k: tr_sets[k] for k in trid if k in tr_sets}); vocab = list(Xtr.columns)
        ytr = Ltr.set_index("record_id").loc[Xtr.index, "class_key"].values
        ids = rec[rec.level == lv].record_id.tolist()
        Xev = presence({k: ts.get(k, set()) for k in ids}, vocab)
        yev = rec.set_index("record_id").loc[Xev.index, "class_key"].values
        for name, fn in (("A3_centroid", centroid_fit_predict), ("A3_logreg", logreg_fit_predict)):
            sc = score(yev, fn(Xtr, ytr, Xev)); k = int(round(sc["acc"] * sc["n"]))
            res.append(dict(attacker=name, level=lv, features=f"transcript_labels:{lv}", k=k,
                            p_binom_gt_chance=binomtest(k, sc["n"], 1 / 6, alternative="greater").pvalue, **sc))
        if a.perm and lv in ("L0", "L1"):          # C-: permuted TRAIN labels -> expected chance
            rng = np.random.default_rng(20261108); obs = res[-1]["acc"]
            sims = [float((logreg_fit_predict(Xtr, rng.permutation(ytr), Xev) == yev).mean()) for _ in range(a.perm)]
            perm[lv] = dict(observed=obs, perm_mean=float(np.mean(sims)), perm_p=float((np.sum(np.array(sims) >= obs) + 1) / (a.perm + 1)))
    R = pd.DataFrame(res)
    prim = R[(R.attacker == "A3_logreg") & R.level.isin(["L0", "L1"])].index   # PRIMARY endpoint (pre-registered): Holm over {L0, L1}
    R["p_holm_primary"] = np.nan
    R.loc[prim, "p_holm_primary"] = F.holm(R.loc[prim, "p_binom_gt_chance"].tolist())
    R.to_csv(a.out, index=False)
    pd.set_option("display.width", 220); print(R.round(4).to_string(index=False))
    print("\nPRIMARY endpoint (A3_logreg, label sets from real ARoG prompts), Holm over L0,L1:")
    print(R.loc[prim, ["level", "n", "acc", "wilson_lo", "wilson_hi", "p_binom_gt_chance", "p_holm_primary"]].round(4).to_string(index=False))
    print("C- label-permutation control:", json.dumps(perm))


if __name__ == "__main__":
    main()
