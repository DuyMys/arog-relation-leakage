#!/usr/bin/env python3
"""Pipeline self-test on DEV only with mock attackers. Run before any real LLM call:  python selftest.py --base <ACIIDS_2027>"""
import argparse, json, subprocess, sys, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); a = ap.parse_args()
def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + r.stdout
    return r.stdout
b = f"--base {a.base}"
print(sh(f"python build_views.py {b} --split dev --out out/st_views.jsonl --labels out/st_labels_PRIVATE.csv").strip())
sh(f"rm -f out/st_kw.jsonl out/st_rand.jsonl")
sh(f"python run_attack.py {b} --views out/st_views.jsonl --out out/st_kw.jsonl --backend mock_keyword --jobs triples,names,c0,c2tok,cplus --labels out/st_labels_PRIVATE.csv")
sh(f"python run_attack.py {b} --views out/st_views.jsonl --out out/st_rand.jsonl --backend mock_random --jobs triples,names,c0,c2tok,cplus --labels out/st_labels_PRIVATE.csv")
sh("python evaluate.py --pred out/st_kw.jsonl --labels out/st_labels_PRIVATE.csv --out out/st_kw_metrics.csv")
sh("python evaluate.py --pred out/st_rand.jsonl --labels out/st_labels_PRIVATE.csv --out out/st_rand_metrics.csv")
kw = pd.read_csv("out/st_kw_metrics.csv").set_index(["job", "level"]); rd = pd.read_csv("out/st_rand_metrics.csv").set_index(["job", "level"])
ok = []
ok.append(("C+ reads the note (>=0.90)", kw.loc[("cplus", "L1"), "acc"] >= 0.90))
ok.append(("keyword mock leaks more at L0 than L2", kw.loc[("triples", "L0"), "acc"] > kw.loc[("triples", "L2"), "acc"] + 0.2))
ok.append(("random mock near chance on triples", all(rd.loc[("triples", l), "acc"] < 0.30 for l in ("L0", "L1", "L2"))))
ok.append(("random relation tokens kill keyword mock", kw.loc[("c2tok", "L0"), "acc"] < 0.30))
ok.append(("test split locked", "LOCKED" in subprocess.run(f"python build_views.py {b} --split test --out out/x.jsonl --labels out/x.csv", shell=True, capture_output=True, text=True).stderr + subprocess.run(f"python build_views.py {b} --split test --out out/x.jsonl --labels out/x.csv", shell=True, capture_output=True, text=True).stdout))
for name, v in ok:
    print(("PASS " if v else "FAIL "), name)
sys.exit(0 if all(v for _, v in ok) else 1)
