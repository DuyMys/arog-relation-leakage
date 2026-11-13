#!/usr/bin/env python3
"""Build remote-facing fixed views (same edges for L0/L1/L2) + PRIVATE labels.

python build_views.py --config fv_config.json --base <ACIIDS_2027> --split dev --policy topcount15 \
       --out out/dev_views.jsonl --labels out/dev_labels_PRIVATE.csv
"""
import argparse, json, sys
from pathlib import Path
import fvlib as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="fv_config.json"); ap.add_argument("--base")
    ap.add_argument("--split", required=True, help="comma list: train,validation,dev[,test with --prereg]")
    ap.add_argument("--policy"); ap.add_argument("--out", required=True); ap.add_argument("--labels", required=True)
    ap.add_argument("--prereg")
    a = ap.parse_args()
    cfg = F.load_config(a.config, a.base)
    splits = a.split.split(",")
    F.require_unlock(splits, a.prereg)
    policy = a.policy or cfg["primary_policy"]
    kg = F.load_kg(cfg); secret = F.load_secret(cfg); rules = F.load_banned(cfg["banned_tokens"])
    recs, labs = F.build_views(kg, cfg, secret, splits, policy)
    # fail-closed lint on EVERY remote rendering (all views and variants)
    hits = 0
    for r in recs:
        for view in ("names", "triples"):
            for var in ("plain", "c2tok", "c0"):
                h = F.lint(F.render_remote(r, view, var, secret), rules)
                if h:
                    hits += 1; print("LINT HIT", r["record_id"], view, var, h, file=sys.stderr)
    if hits:
        raise SystemExit(f"{hits} lint hits: refusing to write views")
    # invariants: paired levels share the identical ordered edge set, no leakage of structure between levels
    g = labs.groupby("patient_mid")
    assert (g.canon_edge_hash.nunique() == 1).all(), "edge sets differ across levels"
    assert (g.size() == 3).all()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in recs))
    labs.to_csv(a.labels, index=False)
    print(f"views={len(recs)} patients={labs.patient_mid.nunique()} policy={policy} lint=PASS paired=PASS")


if __name__ == "__main__":
    main()
