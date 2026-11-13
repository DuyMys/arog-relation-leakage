#!/usr/bin/env python3
"""Join predictions with PRIVATE labels and compute metrics (only here are labels used).

python evaluate.py --pred out/dev_pred_qwen.jsonl --labels out/dev_labels_PRIVATE.csv --out out/dev_metrics.csv
Counting rule: an unparsable answer counts as WRONG (conservative); parse rate is reported separately.
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest
import fvlib as F


def load(pred_paths, labels):
    P = pd.concat([pd.DataFrame(map(json.loads, Path(p).read_text().splitlines())) for p in pred_paths], ignore_index=True)
    L = pd.read_csv(labels)
    D = P.merge(L[["record_id", "patient_mid", "class_key", "split"]], on="record_id", how="left")
    assert D.class_key.notna().all(), "predictions without labels"
    D["correct"] = (D.pred == D.class_key)
    return D


def summarize(D, B=10000):
    rows = []
    for (job, level), g in D.groupby(["job", "level"]):
        g = g.drop_duplicates("patient_mid") if g.patient_mid.duplicated().any() else g
        n = len(g); k = int(g.correct.sum())
        lo, hi = F.wilson(k, n); blo, bhi = F.boot_ci(g.correct.values, B)
        pred = g.pred.fillna("NONE").values
        rows.append(dict(job=job, level=level, model=g.model.iloc[0], n=n, acc=k / n, wilson_lo=lo, wilson_hi=hi,
                         boot_lo=blo, boot_hi=bhi, p_vs_chance=binomtest(k, n, 1 / 6, alternative="greater").pvalue,
                         bal_acc=F.balanced_accuracy(g.class_key.values, pred), macro_f1=F.macro_f1(g.class_key.values, pred),
                         parse_ok=float(g.parse_ok.mean()), most_common_pred=g.pred.value_counts(normalize=True).iloc[0] if g.pred.notna().any() else np.nan))
    return pd.DataFrame(rows)


def paired(D, job="triples"):
    out = []
    g = D[D.job == job].pivot_table(index="patient_mid", columns="level", values="correct", aggfunc="first")
    c0 = D[D.job == "c0"].drop_duplicates("patient_mid").set_index("patient_mid").correct
    pairs = [("L0", "L1"), ("L1", "L2"), ("L0", "L2")]
    ps = []
    for a, b in pairs:
        if a in g and b in g:
            x, y = g[a].dropna().astype(bool), g[b].dropna().astype(bool)
            idx = x.index.intersection(y.index); d, lo, hi = F.paired_boot_diff(x[idx], y[idx])
            p = F.mcnemar_exact(x[idx], y[idx]); ps.append(p)
            out.append(dict(job=job, contrast=f"{a}-{b}", n=len(idx), diff=d, ci_lo=lo, ci_hi=hi, p_mcnemar=p))
    for i, p in enumerate(F.holm(ps) if ps else []):
        out[i]["p_holm"] = p
    for lv in ("L0", "L1", "L2"):
        if lv in g and len(c0):
            idx = g[lv].dropna().index.intersection(c0.index)
            d, lo, hi = F.paired_boot_diff(g[lv][idx].astype(bool), c0[idx].astype(bool))
            out.append(dict(job=job, contrast=f"{lv}-C0 (attack advantage)", n=len(idx), diff=d, ci_lo=lo, ci_hi=hi,
                            p_mcnemar=F.mcnemar_exact(g[lv][idx].astype(bool), c0[idx].astype(bool))))
    return pd.DataFrame(out)


def confusion(D, job, level):
    g = D[(D.job == job) & (D.level == level)]
    return pd.crosstab(g.class_key, g.pred.fillna("NONE")).reindex(index=F.CLASSES).fillna(0).astype(int)


def label_permutation_check(D, job="triples", level="L1", R=2000, seed=20261106):
    """C-: how often would random relabelling of patients reach the observed accuracy (should be ~chance)."""
    g = D[(D.job == job) & (D.level == level)]
    rng = np.random.default_rng(seed); y = g.class_key.values; p = g.pred.fillna("NONE").values
    obs = (y == p).mean(); sims = [(rng.permutation(y) == p).mean() for _ in range(R)]
    return dict(job=job, level=level, observed=float(obs), perm_mean=float(np.mean(sims)),
                perm_p=float((np.sum(np.array(sims) >= obs) + 1) / (R + 1)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", nargs="+", required=True); ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--prereg")
    ap.add_argument("--job", default="triples", help="main job for paired contrasts (use transcript for Protocol A)")
    a = ap.parse_args()
    D = load(a.pred, a.labels)
    F.require_unlock(set(D.split), a.prereg)
    S = summarize(D); S.to_csv(a.out, index=False)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(S.round(3).to_string(index=False))
    Pr = paired(D, a.job); Pr.to_csv(Path(a.out).with_name(Path(a.out).stem + "_paired.csv"), index=False)
    print("\n", Pr.round(4).to_string(index=False))
    print("\nC- label permutation:", label_permutation_check(D, a.job))
    print(f"\nConfusion {a.job}/L1:\n", confusion(D, a.job, "L1"))
