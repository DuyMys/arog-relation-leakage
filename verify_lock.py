"""Check the files in this repository against PREREG_LOCK.json.

Usage: python verify_lock.py
Files that are kept private (labels, relation mapping, split manifest) are reported as not public.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
lock = json.loads((ROOT / "PREREG_LOCK.json").read_text(encoding="utf-8"))


def repo_path(key):
    head, _, rest = key.partition("/")
    if head in ("__code__", "__dev__"):
        return ROOT / "aciids_fv" / rest
    return ROOT / "ACIIDS_2027" / key


counts = {"match": 0, "MISMATCH": 0, "not public": 0}
for key, want in lock["artifact_sha256"].items():
    p = repo_path(key)
    if not p.exists():
        status = "not public"
    else:
        status = "match" if hashlib.sha256(p.read_bytes()).hexdigest() == want else "MISMATCH"
    counts[status] += 1
    print(f"{status:10s}  {key}")
print(f"\nlocked {lock['locked_utc']}: {counts}")
