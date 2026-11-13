"""Post-hoc analyses added after the pre-registration lock; the locked code is not modified.
Usage: python posthoc_analysis.py --fv <aciids_fv> --base <ACIIDS_2027> --testA <testA> \
        --transcripts <arog_transcripts_test.jsonl> --arog_result <ARoG_...jsonl> --out <thu_muc_ra>
Writes four CSV files: utility_by_level, c0_same_level, a3_per_class, local_controls_120.
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
for a in ("--fv", "--base", "--testA", "--transcripts", "--arog_result", "--out"):
    ap.add_argument(a, required=True)
A = ap.parse_args()
sys.path[:0] = [A.fv, str(Path(A.fv) / "protoA")]
import fvlib as F
from stat_attackers import logreg_fit_predict
from stat_on_transcripts import label_sets, presence

out = Path(A.out); out.mkdir(parents=True, exist_ok=True)
T = Path(A.testA)
rec = pd.read_csv(T / "records_PRIVATE.csv")
R = rec.set_index("record_id")

def mcn(a, b):
    return F.mcnemar_exact(np.asarray(a, bool), np.asarray(b, bool))

# ARoG utility by level: entity IDs in the answer against the gold answer set
Q = json.load(open(T / "WebQSP.json"))
key2qid = {tuple(sorted(x["AnswerArgument"] for x in q["Parses"][0]["Answers"])): q["QuestionId"] for q in Q}
rows = []
for l in open(A.arog_result):
    r = json.loads(l)
    gold = set(r["answer_list"]); pred = set(re.findall(r"m\.e[0-9a-f]{12}", r["results"]))
    tp = len(pred & gold)
    rows.append(dict(record_id=key2qid[tuple(sorted(gold))], hit=tp > 0,
                     precision=tp / len(pred) if pred else 0.0, recall=tp / len(gold),
                     has_wrong=len(pred - gold) > 0, time_s=r["time"]))
U = pd.DataFrame(rows)
assert U.record_id.is_unique and len(U) == len(rec)
U["level"] = U.record_id.map(R.level); U["pm"] = U.record_id.map(R.patient_mid)
ut = []
for lv, g in U.groupby("level"):
    k = int(g.hit.sum()); lo, hi = F.wilson(k, len(g))
    ut.append(dict(level=lv, n=len(g), hit_rate=k / len(g), wilson_lo=lo, wilson_hi=hi,
                   precision=g.precision.mean(), recall=g.recall.mean(),
                   share_with_wrong_mid=g.has_wrong.mean(), mean_time_s=g.time_s.mean()))
P = U.pivot(index="pm", columns="level", values="hit")
for a, b in (("L0", "L1"), ("L1", "L2"), ("L0", "L2")):
    ut.append(dict(level=f"{a}-{b}", n=len(P), hit_rate=P[a].mean() - P[b].mean(), p_mcnemar=mcn(P[a], P[b])))
pd.DataFrame(ut).to_csv(out / "utility_by_level.csv", index=False)

# advantage over the same-level C0 control (the locked evaluate.py uses one level's C0 for all levels)
c0r = []
for f in ("pred_xtract", "pred_raw", "pred_abl_prune", "pred_abl_post", "pred_fixed"):
    D = pd.DataFrame([json.loads(l) for l in open(T / f"{f}.jsonl")])
    D["ok"] = D.pred == D.record_id.map(R.class_key)
    D["patient_mid"] = D.record_id.map(R.patient_mid) if "patient_mid" not in D else D.patient_mid
    for (job, lv), g in D[D.job != "c0"].groupby(["job", "level"]):
        c0 = D[(D.job == "c0") & (D.level == lv)].set_index("patient_mid").ok
        if c0.empty:  # pred_fixed ran C0 only once; C0 sees no relation labels, so it is shared across levels
            c0 = D[D.job == "c0"].drop_duplicates("patient_mid").set_index("patient_mid").ok
        x = g.set_index("patient_mid").ok; y = c0.loc[x.index]
        c0r.append(dict(file=f, job=job, level=lv, n=len(x), acc=x.mean(), c0_acc=y.mean(),
                        advantage=x.mean() - y.mean(), p_mcnemar=mcn(x, y), parse_ok=g.parse_ok.mean()))
pd.DataFrame(c0r).to_csv(out / "c0_same_level.csv", index=False)

# per-class recall of A3, then local controls on the same 120 patients
cfg = F.load_config(Path(A.fv) / "fv_config.json", A.base)
kg = F.load_kg(cfg); sec = F.load_secret(cfg)
trv, trl = F.build_views(kg, cfg, sec, ["train", "validation"], "topcount15")
tr_sets = {r["record_id"]: {t[1] for t in r["triples"]} for r in trv}
ts = label_sets(A.transcripts, rec)
pc = []
for lv in F.LEVELS:
    tid = trl[trl.level == lv].record_id.tolist()
    Xtr = presence({k: tr_sets[k] for k in tid}); ytr = trl.set_index("record_id").loc[Xtr.index, "class_key"].values
    ids = rec[rec.level == lv].record_id.tolist()
    Xev = presence({k: ts.get(k, set()) for k in ids}, list(Xtr.columns))
    yev = R.loc[Xev.index, "class_key"].values
    p = logreg_fit_predict(Xtr, ytr, Xev)
    for c in sorted(set(yev)):
        m = yev == c; k = int((p[m] == c).sum()); lo, hi = F.wilson(k, int(m.sum()))
        pc.append(dict(level=lv, class_key=c, n=int(m.sum()), recall=k / m.sum(), wilson_lo=lo, wilson_hi=hi))
    pc.append(dict(level=lv, class_key="ALL", n=len(yev), recall=(p == yev).mean()))
pd.DataFrame(pc).to_csv(out / "a3_per_class.csv", index=False)

pm = pd.read_csv(Path(A.base) / "ACIIDS_T01/data_private/raw/kg_v0.1/data/clean/patients_minimal_PRIVATE.csv").set_index("patient_mid")
sp = kg["split"].set_index("patient_mid")
def feats(m):
    e = kg["edges"].get(m, [])
    return dict(med_deg=sum(x[1] == "medication" for x in e), proc_deg=sum(x[1] == "procedure" for x in e),
                birth_year=int(str(pm.at[m, "BIRTHDATE"])[:4]))
tr = sp[sp.split.isin(["train", "validation"])].index; ev = sorted(set(rec.patient_mid))
Ctr = pd.DataFrame([feats(m) for m in tr], index=tr); Cev = pd.DataFrame([feats(m) for m in ev], index=ev)
ytr = sp.loc[tr, "class_key"].values; yev = sp.loc[ev, "class_key"].values
lc = []
for name, cols in (("degree_only", ["med_deg", "proc_deg"]), ("age_only", ["birth_year"]),
                   ("degree+age", ["med_deg", "proc_deg", "birth_year"])):
    p = logreg_fit_predict(Ctr[cols].astype(float), ytr, Cev[cols].astype(float))
    k = int((p == yev).sum()); lo, hi = F.wilson(k, len(yev))
    lc.append(dict(features=name, n=len(yev), acc=k / len(yev), wilson_lo=lo, wilson_hi=hi))
pd.DataFrame(lc).to_csv(out / "local_controls_120.csv", index=False)
print("wrote 4 files to", out)
