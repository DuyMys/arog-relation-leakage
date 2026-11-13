#!/usr/bin/env python3
"""Protocol A step 2: patch an ARoG checkout (run from inside ARoG/) so EVERY outbound LLM prompt is logged.
Idempotent. Changes: (1) run_llm wrapped -> JSONL log (qid, stage=calling function, call_index, prompt, response, sha256,
tokens, latency); (2) num_ctx 8192->16384 and fixed seed; (3) main loop records the current QuestionId.
Does NOT change any ARoG logic. Log path: env AROG_LOG (default arog_transcripts.jsonl).
"""
import re, sys
from pathlib import Path
u = Path("utils.py"); m = Path("main_freebase.py")
s = u.read_text(encoding="utf-8")
if "_run_llm_orig" not in s:
    assert "def run_llm(" in s, "run_llm not found"
    s = s.replace("def run_llm(", "def _run_llm_orig(", 1)
    s = s.replace('"num_ctx": 8192,', '"num_ctx": 16384,\n            "seed": 20261107,')
    s += '''

# ===== Protocol A logging wrapper (added by patch_arog.py) =====
import os as _os, inspect as _inspect, hashlib as _hl, time as _time, json as _json
CUR_QID = None
_CALL_IDX = {}
def run_llm(prompt, temperature, max_tokens, opeani_api_keys, engine="gpt-3.5-turbo"):
    t0 = _time.time()
    result, token_num = _run_llm_orig(prompt, temperature, max_tokens, opeani_api_keys, engine)
    stage = _inspect.stack()[1].function
    i = _CALL_IDX.get(CUR_QID, 0); _CALL_IDX[CUR_QID] = i + 1
    with open(_os.environ.get("AROG_LOG", "arog_transcripts.jsonl"), "a", encoding="utf-8") as f:
        f.write(_json.dumps(dict(qid=CUR_QID, call_index=i, stage=stage, prompt=prompt, response=result,
                                 prompt_sha256=_hl.sha256(prompt.encode()).hexdigest(), tokens=token_num,
                                 latency_s=round(_time.time() - t0, 3)), ensure_ascii=False) + "\\n")
    return result, token_num
'''
    u.write_text(s, encoding="utf-8"); print("utils.py patched")
else:
    print("utils.py already patched")
t = m.read_text(encoding="utf-8")
if "_u.CUR_QID" not in t:
    key = "        question = data[question_string]\n"
    assert key in t
    t = t.replace(key, key + "        import utils as _u; _u.CUR_QID = data.get('QuestionId')\n", 1)
    m.write_text(t, encoding="utf-8"); print("main_freebase.py patched")
else:
    print("main_freebase.py already patched")
# the notebook's 1/8 subsampling must NOT be active for Protocol A
if re.search(r"(?m)^\s*datas = random\.sample\(datas, int\(len\(datas\) / \d+\)\)", t):
    print("WARNING: main_freebase.py subsamples the dataset; comment that line out (all Synthea records must run)")
