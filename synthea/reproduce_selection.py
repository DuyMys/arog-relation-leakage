"""Reproduce the six-class target selection and the patient splits from the Synthea D1 archive.

Usage:
    python reproduce_selection.py --input_zip Synthea_D1.zip --outdir D2_Target_Selection

The selection uses only the raw Synthea CSVs, no model output. splits_PRIVATE.csv and
eligible_single_target_private.csv contain the class label of every patient and must not
be published or sent to an external model.
"""

from __future__ import annotations
import pandas as pd, numpy as np, zipfile, hashlib, json, pathlib, collections, itertools, datetime, re, argparse

parser = argparse.ArgumentParser(
    description="Reproduce the six target diseases from the immutable Synthea D1 ZIP"
)
parser.add_argument("--input_zip", default="/kaggle/input/synthea-d1/Synthea_D1.zip")
parser.add_argument("--outdir", default="/kaggle/working/D2_Target_Selection")
a = parser.parse_args()
z = zipfile.ZipFile(a.input_zip)
prefix = "data/raw/synthea_seed20261104/"
out = pathlib.Path(a.outdir)
out.mkdir(parents=True, exist_ok=True)


def read(name, cols):
    return pd.read_csv(z.open(prefix + name + ".csv"), usecols=cols, dtype="string")


p = read("patients", ["Id", "GENDER", "BIRTHDATE", "DEATHDATE", "CITY"])
p["age_nov_4_2026"] = (pd.Timestamp("2026-11-04") - pd.to_datetime(p.BIRTHDATE)).dt.days / 365.2425
c = read("conditions", ["PATIENT", "CODE", "DESCRIPTION"])
cond = c[["PATIENT", "CODE"]].drop_duplicates()
bycode = {cd: set(g.PATIENT) for cd, g in cond.groupby("CODE")}
md = read("medications", ["PATIENT", "CODE", "DESCRIPTION", "REASONCODE"])
md = md.drop_duplicates(["PATIENT", "CODE", "REASONCODE"])  # one row per patient, code and reason
CAN = {
    "T2D": ("44054006", "Type 2 diabetes mellitus"),
    "AsthmaChild": ("233678006", "Childhood asthma"),
    "Appendicitis": ("74400008", "Appendicitis"),
    "Seizure": ("128613002", "Seizure disorder"),
    "HLD": ("55822004", "Hyperlipidemia"),
    "PolypColon": ("68496003", "Colon polyp"),
    "ADHD": ("192127007", "Child attention deficit disorder"),
    "AtopicDerm": ("24079001", "Atopic dermatitis"),
    "IHD": ("414545008", "Ischemic heart disease"),
    "HTN": ("59621000", "Essential hypertension"),
    "KneeOA": ("239873007", "Knee osteoarthritis"),
}
variants = {
    "A_RECOMMENDED": ["T2D", "AsthmaChild", "Appendicitis", "Seizure", "HLD", "PolypColon"],
    "B_ATOPIC": ["T2D", "ADHD", "Seizure", "HLD", "PolypColon", "AtopicDerm"],
    "C_CARDIO": ["T2D", "AsthmaChild", "ADHD", "Seizure", "IHD", "HLD"],
    "D_KNEE": ["T2D", "AsthmaChild", "ADHD", "Appendicitis", "Seizure", "KneeOA"],
}


def cohorts(variant):
    codes = [CAN[n][0] for n in variant]
    setunion = set().union(*[bycode[x] for x in codes])
    membership = collections.Counter(pid for co in codes for pid in bycode[co])
    return {nm: bycode[CAN[nm][0]] & {pid for pid in setunion if membership[pid] == 1} for nm in variant}


# procedures.csv has about 2.1M rows; read it in chunks and keep only the needed columns
pcod = collections.defaultdict(set)
p_reason = collections.defaultdict(set)
proc_by_patient = collections.defaultdict(set)
proc_rows = collections.Counter()
need_codes = set(x[0] for x in CAN.values())
for j, ch in enumerate(
    pd.read_csv(
        z.open(prefix + "procedures.csv"),
        dtype="string",
        usecols=["PATIENT", "CODE", "DESCRIPTION", "REASONCODE"],
        chunksize=250000,
    )
):
    # procedure codes of patients who have any candidate condition
    relevant = ch[ch.PATIENT.isin(set().union(*(bycode[x] for x in need_codes)))]
    for pt, cd in relevant[["PATIENT", "CODE"]].drop_duplicates().itertuples(index=False, name=None):
        proc_by_patient[pt].add(cd)
    reasons = ch[ch.REASONCODE.isin(need_codes)]
    for pt, rc in reasons[["PATIENT", "REASONCODE"]].drop_duplicates().itertuples(index=False, name=None):
        p_reason[pt].add(rc)
    for (rc, co, desc), num in (
        reasons.groupby(["REASONCODE", "CODE", "DESCRIPTION"], dropna=False).size().items()
    ):
        proc_rows[(rc, co, desc)] += int(num)
    print("proc chunk", j + 1, flush=True)
# per-patient sets of medication codes and of target-linked reason codes
med_by_patient = collections.defaultdict(set)
med_reason = collections.defaultdict(set)
for pt, cd, rc in md[["PATIENT", "CODE", "REASONCODE"]].itertuples(index=False, name=None):
    med_by_patient[pt].add(cd)
    if pd.notna(rc) and rc in need_codes:
        med_reason[pt].add(rc)
# per-class evidence linked through REASONCODE
sc = collections.defaultdict(list)
for (rc, co, dsc), f in proc_rows.items():
    sc[rc].append((co, dsc, f))
primary_med = collections.defaultdict(list)
for (rc, co, dsc), rr in md[md.REASONCODE.isin(need_codes)].groupby(
    ["REASONCODE", "CODE", "DESCRIPTION"], dropna=False
):
    primary_med[rc].append((co, dsc, int(rr.PATIENT.nunique())))

summary = []
candidate_comp = []
for variant_name, variant in variants.items():
    pools = cohorts(variant)
    for nm in variant:
        cd, full = CAN[nm]
        raw = bycode[cd]
        sel = pools[nm]
        n = len(sel)
        n_reason_m = sum(cd in med_reason[pid] for pid in sel)
        n_reason_p = sum(cd in p_reason[pid] for pid in sel)
        n_any_reason = sum(cd in med_reason[pid] or cd in p_reason[pid] for pid in sel)
        ids_p = p[p.Id.isin(sel)]
        ages = ids_p.age_nov_4_2026
        n_other_target_reason = sum(
            bool(((med_reason[pid] | p_reason[pid]) - {cd}) & set(CAN[k][0] for k in variant)) for pid in sel
        )
        candidate_comp.append(
            {
                "variant": variant_name,
                "name": nm,
                "code": cd,
                "description": full,
                "n_raw": len(raw),
                "n_single_target": n,
                "retention_pct": round(100 * n / len(raw), 2),
                "n_med_reason_linked": n_reason_m,
                "n_proc_reason_linked": n_reason_p,
                "n_any_reason_linked": n_any_reason,
                "any_reason_pct": round(100 * n_any_reason / n, 2),
                "med_reason_pct": round(100 * n_reason_m / n, 2),
                "proc_reason_pct": round(100 * n_reason_p / n, 2),
                "other_target_reason_patient_n": n_other_target_reason,
                "age_median": round(float(ages.median()), 1),
                "age_iqr_low": round(float(ages.quantile(0.25)), 1),
                "age_iqr_high": round(float(ages.quantile(0.75)), 1),
                "female_pct": round(float(100 * ids_p.GENDER.eq("F").mean()), 1),
                "death_pct": round(float(100 * ids_p.DEATHDATE.notna().mean()), 1),
            }
        )
    xx = [len(pools[nm]) for nm in variant]
    summary.append(
        {
            "variant": variant_name,
            "min_class_single_target": min(xx),
            "mean_class_single_target": round(float(np.mean(xx)), 1),
            "max_class_single_target": max(xx),
            "total_single_target": sum(xx),
            "valid_ge_60": bool(min(xx) >= 60),
            "valid_ge_110": bool(min(xx) >= 110),
        }
    )

pd.DataFrame(summary).to_csv(out / "variant_comparison.csv", index=False)
cc = pd.DataFrame(candidate_comp)
cc.to_csv(out / "variant_class_detail.csv", index=False)
print("FINAL CLASS DETAILS", flush=True)
print(cc[cc.variant == "A_RECOMMENDED"].to_string(index=False), flush=True)

final = variants["A_RECOMMENDED"]
pool = cohorts(final)
codes = {k: CAN[k][0] for k in final}
row = []
for nm in final:
    for pid in sorted(pool[nm]):
        row.append({"patient_id": pid, "target_code": codes[nm], "class_key": nm})
eligible = pd.DataFrame(row)
assert len(eligible.patient_id) == eligible.patient_id.nunique()
eligible.to_csv(out / "eligible_single_target_private.csv", index=False)

# Patient-level splits: per class, order patients by SHA-256 of a fixed string and take consecutive blocks.
raw_splits = {"train": 40, "validation": 10, "dev": 10, "test": 40, "reserve": 10}
n = sum(raw_splits.values())
assert min(len(pool[x]) for x in final) >= n
split_rows = []
for nm in final:
    pids = sorted(pool[nm], key=lambda pt: hashlib.sha256(f"20261104|D2_v1|{pt}".encode()).digest())
    # The order depends only on the patient ID and the fixed salt.
    cursor = 0
    for split, take in raw_splits.items():
        for pt in pids[cursor : cursor + take]:
            split_rows.append({"patient_id": pt, "split": split, "class_key": nm, "target_code": codes[nm]})
        cursor += take
splits = pd.DataFrame(split_rows)
assert len(splits) == 6 * n
assert splits.patient_id.nunique() == len(splits)
splits.to_csv(out / "splits_PRIVATE.csv", index=False)
# Split membership without class labels. The labels stay in splits_PRIVATE.csv.
splits[["patient_id", "split"]].to_csv(out / "split_membership_no_labels.csv", index=False)
eligible[["patient_id"]].to_csv(out / "eligible_patient_ids.csv", index=False)
# class ids and codes used for evaluation
classmap = pd.DataFrame(
    [
        {
            "class_id": "ABCDEF"[i],
            "disease_name": CAN[nm][1],
            "snomed_code": CAN[nm][0],
            "dataset_class_key": nm,
        }
        for i, nm in enumerate(final)
    ]
)
classmap.to_csv(out / "six_target_classes_v1.csv", index=False)
# pairwise co-occurrence of the six target codes over all 5,000 patients
ov = []
for i, a in enumerate(final):
    for b in final[i + 1 :]:
        inter = len(bycode[codes[a]] & bycode[codes[b]])
        raw_a = len(bycode[codes[a]])
        raw_b = len(bycode[codes[b]])
        ov.append(
            {
                "class_a": a,
                "class_b": b,
                "raw_cooccurring_patients": inter,
                "pct_of_smaller_class": round(100 * inter / min(raw_a, raw_b), 2),
                "jaccard_pct": round(100 * inter / len(bycode[codes[a]] | bycode[codes[b]]), 2),
            }
        )
pd.DataFrame(ov).to_csv(out / "pairwise_comorbidity.csv", index=False)
# medications and procedures linked to each target code through REASONCODE, and how common they are in the other classes
sig = []
for nm in final:
    cd = codes[nm]
    pids = pool[nm]
    for med_code, desc, _ in primary_med[cd]:
        nown = sum(med_code in med_by_patient[pid] for pid in pids)
        other_n = sum(med_code in med_by_patient[pid] for k in final if k != nm for pid in pool[k])
        other_den = sum(len(pool[k]) for k in final if k != nm)
        sig.append(
            {
                "class_key": nm,
                "snomed_code": cd,
                "kind": "medication",
                "evidence_code": med_code,
                "description": desc,
                "patients_in_class_with_evidence": nown,
                "pct_in_class": round(100 * nown / len(pids), 2),
                "patients_other_classes_with_evidence": other_n,
                "pct_other_classes": round(100 * other_n / other_den, 2),
                "target_reason_linked": True,
            }
        )
    for pro_code, desc, _ in sc[cd]:
        nown = sum(pro_code in proc_by_patient[pid] for pid in pids)
        other_n = sum(pro_code in proc_by_patient[pid] for k in final if k != nm for pid in pool[k])
        other_den = sum(len(pool[k]) for k in final if k != nm)
        sig.append(
            {
                "class_key": nm,
                "snomed_code": cd,
                "kind": "procedure",
                "evidence_code": pro_code,
                "description": desc,
                "patients_in_class_with_evidence": nown,
                "pct_in_class": round(100 * nown / len(pids), 2),
                "patients_other_classes_with_evidence": other_n,
                "pct_other_classes": round(100 * other_n / other_den, 2),
                "target_reason_linked": True,
            }
        )
sig = pd.DataFrame(sig)
sig["signature_score"] = sig.pct_in_class - sig.pct_other_classes
sig = sig.sort_values(["class_key", "signature_score"], ascending=[True, False])
sig.to_csv(out / "signal_discriminability.csv", index=False)
print("SIGNATURE TOP", flush=True)
print(sig.groupby("class_key").head(5).to_string(index=False), flush=True)

# Lexical screen: other conditions of an eligible patient whose description matches a keyword of the target disease.
# The patterns are a screening rule, not a complete synonym list.
lex = {
    "T2D": r"diabet(?:es|ic)|hyperglyc|insulin resist",
    "AsthmaChild": r"asthma|asthmat|bronchospasm",
    "Appendicitis": r"appendicitis|appendiceal|appendix",
    "Seizure": r"seizur|epilep|convuls",
    "HLD": r"hyperlipid|hypercholest|dyslipid",
    "PolypColon": r"colonic? polyp|polyp of colon|colon polyposis|colorectal polyp|rectal polyp",
}
other_cond = c[["PATIENT", "CODE", "DESCRIPTION"]].drop_duplicates()
alias_summary = []
alias_examples = []
for nm in final:
    cd = codes[nm]
    tmp = other_cond[(other_cond.PATIENT.isin(pool[nm])) & (other_cond.CODE != cd)].copy()
    flagged = tmp[tmp.DESCRIPTION.str.contains(lex[nm], case=False, regex=True, na=False)]
    alias_summary.append(
        {
            "class_key": nm,
            "n_eligible": len(pool[nm]),
            "n_with_lexical_alias_cue": int(flagged.PATIENT.nunique()),
            "pct_with_lexical_alias_cue": round(100 * flagged.PATIENT.nunique() / len(pool[nm]), 2),
            "regex_rule": lex[nm],
        }
    )
    if len(flagged):
        for (co, ds), g in flagged.groupby(["CODE", "DESCRIPTION"]):
            alias_examples.append(
                {
                    "class_key": nm,
                    "other_condition_code": co,
                    "description": ds,
                    "n_eligible_patients": g.PATIENT.nunique(),
                }
            )
pd.DataFrame(alias_summary).to_csv(out / "alias_leakage_screen.csv", index=False)
pd.DataFrame(alias_examples).sort_values(
    ["class_key", "n_eligible_patients"], ascending=[True, False]
).to_csv(out / "alias_cue_conditions.csv", index=False)
frozen_summary = {
    "chosen_names": final,
    "target_codes": codes,
    "seed": 20261104,
    "hash_salt": "20261104|D2_v1|",
    "single_target_definition": "Exactly one of the six target CODEs in condition history, patient level. Other conditions permitted.",
    "condition_temporality": "ever recorded by reference date",
    "n_single_target_total": int(len(eligible)),
    "n_selected_total": int(len(splits)),
    "splits_per_class": raw_splits,
    "patient_overlap_between_classes": 0,
    "patient_overlap_between_splits": 0,
    "raw_files": "Synthea_D1.zip immutable",
    "note": "Candidate classes and patient-level splits selected from raw epidemiology and evidence statistics only; no attacker model outputs consulted. Local label mapping is private.",
}
(out / "six_target_selection_manifest.json").write_text(
    json.dumps(frozen_summary, ensure_ascii=False, indent=2)
)
print("VARIANTS", pd.DataFrame(summary).to_string(index=False), flush=True)
print("ALIAS", pd.DataFrame(alias_summary).to_string(index=False), flush=True)
print("SPLITS", splits.groupby(["class_key", "split"]).size().unstack().to_string(), flush=True)
print("all done", flush=True)
