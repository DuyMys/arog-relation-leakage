#!/usr/bin/env python3
"""Statistical attackers on remote-view features (A3) + local shape controls (degree / age).
Only train -> dev by default. test needs --prereg. No LLM involved.

features(view-level): count per relation label (vocab from TRAIN), n_med, n_proc, n_unique_labels
local controls      : medication/procedure degree of the FULL local graph, birth year (age proxy)
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import fvlib as F


def view_features(recs, labels, vocab=None):
    rows = []
    for r in recs:
        c = {}
        for _, rel, _ in r["triples"]:
            c[rel] = c.get(rel, 0) + 1
        rows.append(dict(record_id=r["record_id"], **{"rel:" + k: v for k, v in c.items()},
                         n_edges=len(r["triples"]), n_unique=len(c)))
    X = pd.DataFrame(rows).fillna(0).set_index("record_id")
    if vocab is not None:
        X = X.reindex(columns=vocab, fill_value=0)
    return X


def centroid_fit_predict(Xtr, ytr, Xte, floor=0.25):
    A, B = np.log1p(Xtr.values), np.log1p(Xte.values)
    mu, sd = A.mean(0), np.maximum(A.std(0), floor)
    A, B = (A - mu) / sd, (B - mu) / sd
    cents = np.vstack([A[ytr == c].mean(0) for c in F.CLASSES])
    d = ((B[:, None, :] - cents[None]) ** 2).sum(2)
    return np.array(F.CLASSES)[d.argmin(1)]


def logreg_fit_predict(Xtr, ytr, Xte):
    sc = StandardScaler().fit(np.log1p(Xtr.values))
    m = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(np.log1p(Xtr.values)), ytr)
    return m.predict(sc.transform(np.log1p(Xte.values)))


def score(y, p):
    k = int((np.asarray(y) == np.asarray(p)).sum()); n = len(y); lo, hi = F.wilson(k, n)
    return dict(n=n, acc=k / n, wilson_lo=lo, wilson_hi=hi, bal_acc=F.balanced_accuracy(y, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="fv_config.json"); ap.add_argument("--base")
    ap.add_argument("--train_views", required=True); ap.add_argument("--train_labels", required=True)
    ap.add_argument("--eval_views", required=True); ap.add_argument("--eval_labels", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--prereg")
    a = ap.parse_args()
    cfg = F.load_config(a.config, a.base)
    Ltr, Lev = pd.read_csv(a.train_labels), pd.read_csv(a.eval_labels)
    F.require_unlock(set(Lev.split), a.prereg)
    assert set(Ltr.split) <= {"train", "validation"}, "attacker training must use train/validation only"
    Rtr = [json.loads(l) for l in Path(a.train_views).read_text().splitlines()]
    Rev = [json.loads(l) for l in Path(a.eval_views).read_text().splitlines()]
    res = []
    for lv in F.LEVELS:
        tr = [r for r in Rtr if r["level"] == lv]; ev = [r for r in Rev if r["level"] == lv]
        Xtr = view_features(tr, Ltr); vocab = list(Xtr.columns); Xev = view_features(ev, Lev, vocab)
        ytr = Ltr.set_index("record_id").loc[Xtr.index, "class_key"].values
        yev = Lev.set_index("record_id").loc[Xev.index, "class_key"].values
        for name, fn in (("A3_centroid", centroid_fit_predict), ("A3_logreg", logreg_fit_predict)):
            res.append(dict(attacker=name, features="view:" + lv, **score(yev, fn(Xtr, ytr, Xev))))
    # ---- local shape controls (full local graph, NOT remote-visible)
    kg = F.load_kg(cfg, check_mapping=False)
    pm = pd.read_csv(Path(cfg["clean_dir"]) / "patients_minimal_PRIVATE.csv")
    by = pm.set_index("patient_mid"); sp = kg["split"].set_index("patient_mid")
    def local(mids):
        rows = []
        for m in mids:
            e = kg["edges"].get(m, [])
            rows.append(dict(med_deg=sum(1 for x in e if x[1] == "medication"), proc_deg=sum(1 for x in e if x[1] == "procedure"),
                             birth_year=int(str(by.at[m, "BIRTHDATE"])[:4])))
        return pd.DataFrame(rows, index=list(mids))
    mtr = sorted(set(Ltr.patient_mid)); mev = sorted(set(Lev.patient_mid))
    Ltr_c, Lev_c = Ltr.drop_duplicates("patient_mid").set_index("patient_mid"), Lev.drop_duplicates("patient_mid").set_index("patient_mid")
    Ctr, Cev = local(mtr), local(mev)
    ytr, yev = Ltr_c.loc[mtr, "class_key"].values, Lev_c.loc[mev, "class_key"].values
    for name, cols in (("degree_only", ["med_deg", "proc_deg"]), ("age_only", ["birth_year"]),
                       ("degree+age", ["med_deg", "proc_deg", "birth_year"])):
        Xa = Ctr[cols].clip(lower=0).astype(float); Xb = Cev[cols].clip(lower=0).astype(float)
        for an, fn in (("local_centroid", centroid_fit_predict), ("local_logreg", logreg_fit_predict)):
            res.append(dict(attacker=an, features=name, **score(yev, fn(Xa, ytr, Xb))))
    R = pd.DataFrame(res); R.to_csv(a.out, index=False)
    pd.set_option("display.width", 200)
    print(R.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
