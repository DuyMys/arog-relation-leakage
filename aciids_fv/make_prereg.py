#!/usr/bin/env python3
"""Freeze the experiment: hash code, config, mapping, split, prompts -> PREREG_LOCK.json.
Run this ONLY after dev is done and BEFORE touching test. Any later change breaks the lock."""
import argparse, json, datetime
from pathlib import Path
import fvlib as F

CODE = ["fvlib.py", "build_views.py", "run_attack.py", "evaluate.py", "stat_attackers.py", "fv_config.json",
        "protoA/make_arog_data.py", "protoA/patch_arog.py", "protoA/run_transcript_attack.py", "protoA/stat_on_transcripts.py"]
PLAN = {
    "design": "Protocol A: the UNMODIFIED ARoG pipeline (NLPGM/ARoG, patched only to log prompts and set num_ctx=16384, seed) runs "
              "on the projected Synthea KG with Qwen2.5-7B-instruct; every outbound LLM prompt is logged; attackers read the log.",
    "test_sample": "first 20 patients per class by sorted patient_mid of the TEST split (120 patients x L0,L1,L2 = 360 ARoG records); "
                   "make_arog_data.py --limit_patients 120. Chosen before test, from time budget (~31 s/record on dev).",
    "primary_endpoint": "A3_logreg (trained on TRAIN views only) accuracy on the relation-label SET parsed from relation_search_prune "
                        "prompts of the real ARoG log, at L0 and L1: one-sided exact binomial vs chance 1/6, Holm over {L0, L1}; "
                        "plus label-permutation control C-. L2 reported as expected null.",
    "primary_note": "DEVIATION DISCLOSED: the original proposal named the zero-shot LLM attacker (A1) as primary; after dev "
                    "(A1 on raw transcript ~ chance; A3 70-80% on L0/L1) A3 was made primary. A1 variants are secondary.",
    "secondary": ["A3_centroid", "A1 raw transcript (job transcript)", "A1 extracted label list (xtract; ADDED AFTER observing dev A1-raw)",
                  "A1 on fixed views (Protocol B) for the same patients", "controls c0, c2tok, xtract_c2tok",
                  "paired L0-L1, L1-L2, L0-L2 (McNemar, Holm)", "attack advantage vs C0",
                  "stage ablations (--stages relation_search_prune ; relation_search_prune,entity_annotation_all,reasoning)",
                  "A3 on fixed views of ALL 240 test patients (labelled as supplementary, not ARoG traffic)",
                  "audit: fraction of test records whose transcript label set == fixed-view label set"],
    "attackers_tried_on_dev": ["A1 zero-shot Qwen2.5-7B on raw transcript", "A1 on extracted labels (xtract)", "A1 on fixed views",
                               "A3 centroid / logreg on counts (fixed views)", "A3 centroid / logreg on label presence (transcripts)"],
    "controls_gate": "C+ accuracy on DEV >= 0.90 and parse rate >= 0.95 before any test run (met on dev: C+ L1 = 1.000, parse 1.0)",
    "counting": "unparsable answer = wrong; parse rate reported; transcripts truncated at 40000 chars are counted and reported",
    "claim_rule": "'ARoG leaks' may be written only if the primary endpoint rejects chance at L0 or L1 after Holm; wording must be scoped to "
                  "this setup (Synthea, 6-class closed world, Qwen2.5-7B, ARoG as released, v2.1 relation mapping); never 'first' or 'general'.",
    "threat_model": "honest-but-curious remote party sees all outbound prompts (stage=ALL, conservative); stage ablations reported",
    "no_changes_after_freeze": True,
    "disclosure": "test split is not prospectively pristine (D2 exploratory exposure); dev/train used for design only; "
                  "relation mapping v2.1 is operational, not clinically validated; Synthea module structure makes labels informative by construction",
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="fv_config.json"); ap.add_argument("--base")
    ap.add_argument("--attacker_models", nargs="+", required=True, help="e.g. qwen2.5:7b llama3.1:8b")
    ap.add_argument("--out", default="PREREG_LOCK.json"); ap.add_argument("--dev_report", nargs="*", default=[], help="dev metrics files (under this folder) to include in the lock")
    a = ap.parse_args()
    cfg = F.load_config(a.config, a.base)
    here = Path(__file__).resolve().parent
    art = {}
    for f in CODE:
        art[f"__code__/{f}"] = F.sha256_file(here / f)
    root = cfg["_root"]
    for k in ("mapping_file", "split_manifest", "banned_tokens"):
        art[str(Path(cfg[k]).relative_to(root))] = F.sha256_file(cfg[k])
    for d in a.dev_report:
        art["__dev__/" + str(Path(d).resolve().relative_to(here))] = F.sha256_file(d)
    # artifacts under __code__/ and __dev__/ resolve relative to this folder
    lock = dict(locked_utc=datetime.datetime.utcnow().isoformat() + "Z", root=str(root), attacker_models=a.attacker_models,
                system_prompt_sha256=F.hashlib.sha256(F.SYSTEM.encode()).hexdigest(), plan=PLAN,
                artifact_sha256={}, code_dir=str(here))
    for k, v in art.items():
        lock["artifact_sha256"][k] = v
    Path(a.out).write_text(json.dumps(lock, indent=2))
    print("wrote", a.out, "- keep it; give the same file to every test-time script via --prereg")
