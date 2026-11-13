#!/usr/bin/env python3
"""Integrity checks and candidate condition-code screening for the D1 Synthea export.

This is a first screen only. Patient-level splits and the final target selection
are done in reproduce_selection.py.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

REQUIRED = ("patients.csv", "conditions.csv", "medications.csv", "procedures.csv")


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing/empty expected CSV: {path}")
    df = pd.read_csv(path, dtype="string", low_memory=False)
    df.columns = [str(c).upper().strip() for c in df.columns]
    return df


def normalize_str_column(df: pd.DataFrame, col: str):
    if col in df.columns:
        df[col] = df[col].fillna("").astype(str).str.strip()


def reason_patient_counts(events: pd.DataFrame, diagnosed: pd.DataFrame) -> pd.Series:
    if "REASONCODE" not in events.columns:
        return pd.Series(dtype="int64")
    data = events.loc[events["REASONCODE"].ne(""), ["PATIENT", "REASONCODE"]].copy()
    data = data.rename(columns={"REASONCODE": "CODE"}).drop_duplicates()
    if data.empty:
        return pd.Series(dtype="int64")
    # Match on exact (patient, code) pairs, not on similar descriptions.
    matched = diagnosed.merge(data, on=["PATIENT", "CODE"], how="inner")
    return matched.groupby("CODE")["PATIENT"].nunique()


def main(raw: Path, out: Path, expected: int):
    out.mkdir(parents=True, exist_ok=True)
    patients, cond, meds, proc = [load_csv(raw / f) for f in REQUIRED]
    for label, df, cols in [
        ("patients", patients, ["ID"]),
        ("conditions", cond, ["PATIENT", "CODE", "DESCRIPTION"]),
        ("medications", meds, ["PATIENT"]),
        ("procedures", proc, ["PATIENT"]),
    ]:
        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f"{label}.csv missing columns {missing}; columns are {list(df.columns)}")
        for c in ["ID", "PATIENT", "CODE", "DESCRIPTION", "REASONCODE"]:
            normalize_str_column(df, c)

    patient_ids = set(patients["ID"])
    n_patients = patients["ID"].nunique()
    duplicate_ids = int(patients["ID"].duplicated().sum())
    clinical = {"conditions": cond, "medications": meds, "procedures": proc}
    orphans = {name: int((~df["PATIENT"].isin(patient_ids)).sum())
               for name, df in clinical.items()}
    missing_codes = int(cond["CODE"].eq("").sum())
    if n_patients != expected or duplicate_ids or any(orphans.values()) or missing_codes:
        raise AssertionError(
            f"Data quality FAIL: unique_patients={n_patients} expected={expected}; "
            f"duplicate_ID={duplicate_ids}; orphan_rows={orphans}; missing_condition_codes={missing_codes}"
        )

    # One patient can have multiple condition events; count patients once per disease code.
    diagnosed = cond.loc[cond["CODE"].ne(""), ["PATIENT", "CODE"]].drop_duplicates()
    counts = diagnosed.groupby("CODE")["PATIENT"].nunique().rename("n_patients")
    events = cond.groupby("CODE").size().rename("n_condition_events")
    descriptions = (
        cond.loc[cond["CODE"].ne(""), ["CODE", "DESCRIPTION"]]
        .drop_duplicates()
        .groupby("CODE")["DESCRIPTION"].agg(lambda x: " | ".join(sorted(x.unique())[:3]))
        .rename("description")
    )
    medicated = set(meds["PATIENT"])
    procedured = set(proc["PATIENT"])
    diagnosed["med_any"] = diagnosed["PATIENT"].isin(medicated)
    diagnosed["proc_any"] = diagnosed["PATIENT"].isin(procedured)
    med_any = diagnosed.groupby("CODE")["med_any"].sum().rename("n_patients_with_any_medication")
    proc_any = diagnosed.groupby("CODE")["proc_any"].sum().rename("n_patients_with_any_procedure")
    med_direct = reason_patient_counts(meds, diagnosed[["PATIENT", "CODE"]]).rename("n_patients_with_reason_linked_medication")
    proc_direct = reason_patient_counts(proc, diagnosed[["PATIENT", "CODE"]]).rename("n_patients_with_reason_linked_procedure")

    profile = pd.concat([descriptions, counts, events, med_any, proc_any, med_direct, proc_direct], axis=1).reset_index()
    for col in profile.columns:
        if col.startswith("n_"):
            profile[col] = profile[col].fillna(0).astype(int)
    profile["pct_patients"] = (100 * profile["n_patients"] / n_patients).round(2)
    profile["pct_with_any_medication"] = (100 * profile["n_patients_with_any_medication"] / profile["n_patients"]).round(2)
    profile["pct_with_any_procedure"] = (100 * profile["n_patients_with_any_procedure"] / profile["n_patients"]).round(2)
    profile = profile.sort_values(["n_patients", "CODE"], ascending=[False, True]).reset_index(drop=True)
    profile.to_csv(out / "disease_counts.csv", index=False)

    # 60 patients per code is necessary but not sufficient for six single-target classes.
    screen = profile.loc[profile["n_patients"] >= 60].copy()
    screen["passes_raw_60_patient_screen"] = True
    # Co-diagnoses shrink the single-target pool, so record the largest overlap per code.
    patient_by_code = {code: set(group["PATIENT"]) for code, group in diagnosed.groupby("CODE")}
    code_list = screen["CODE"].tolist()
    max_overlap_fraction, overlap_partner = [], []
    for code in code_list:
        pool = patient_by_code[code]
        other_code, frac = "", 0.0
        for candidate in code_list:
            if candidate == code:
                continue
            f = len(pool & patient_by_code[candidate]) / len(pool)
            if f > frac:
                other_code, frac = candidate, f
        max_overlap_fraction.append(round(frac, 4))
        overlap_partner.append(other_code)
    screen["max_overlap_fraction_with_other_screened_code"] = max_overlap_fraction
    screen["most_overlapping_screened_code"] = overlap_partner
    screen["single_target_count"] = pd.NA  # filled in after the six codes are chosen
    screen["review_status"] = "MANUAL_REVIEW_REQUIRED"
    screen.to_csv(out / "candidate_screening.csv", index=False)

    summary = {
        "screening_not_final_selection": True,
        "n_patients": int(n_patients),
        "n_condition_events": int(len(cond)),
        "n_medication_events": int(len(meds)),
        "n_procedure_events": int(len(proc)),
        "n_disease_codes": int(profile.shape[0]),
        "n_raw_candidates_ge_60": int(screen.shape[0]),
        "n_patients_with_any_condition": int(cond["PATIENT"].nunique()),
        "orphan_rows": orphans,
        "duplicate_patient_ids": duplicate_ids,
        "conditions_missing_code": missing_codes,
        "medication_reasoncode_column_present": "REASONCODE" in meds.columns,
        "procedure_reasoncode_column_present": "REASONCODE" in proc.columns,
        "preliminary_candidate_gate": "PASS" if len(screen) >= 6 else "FAIL_REVIEW_POPULATION_OR_SELECTION",
        "note": "60 raw patients/code is not equivalent to 60 patients exclusively in one of the six final target classes. Check on D2."
    }
    (out / "quality_report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nTop 20 disease concepts (code-level, distinct patients):")
    print(profile[["CODE", "description", "n_patients", "n_condition_events", "n_patients_with_reason_linked_medication", "n_patients_with_reason_linked_procedure"]].head(20).to_string(index=False, max_colwidth=45))
    print("\nSaved outputs:", *(str(p) for p in sorted(out.glob("*"))), sep="\n  - ")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=5000)
    args = parser.parse_args()
    main(args.raw, args.out, args.expected)
