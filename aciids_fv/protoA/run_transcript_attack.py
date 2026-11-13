#!/usr/bin/env python3
"""Protocol A step 3: attacker reads the REAL ARoG outbound transcript of one record and guesses the condition class.
Jobs: transcript (real traffic) | cplus (dev-only gate: transcript + evaluator note) | c0 (question text only) | xtract / xtract_c2tok (deterministically extracted relation-label list; added after dev observation of the raw attacker) | c2tok (relation labels in the transcript replaced by random tokens)
Output schema is identical to run_attack.py, so evaluate.py works unchanged (job names differ).
The attacker never sees labels. Stage selection (--stages) defines what the 'remote' party is assumed to see.
"""
import argparse, json, re, time, hashlib, sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fvlib as F
import run_attack as RA

MAXCH = 40000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="../fv_config.json"); ap.add_argument("--base")
    ap.add_argument("--transcripts", required=True); ap.add_argument("--data", required=True, help="WebQSP.json from make_arog_data")
    ap.add_argument("--records", required=True, help="records_PRIVATE.csv (used ONLY for level + relation-label list; no class labels read)")
    ap.add_argument("--out", required=True); ap.add_argument("--backend", default="ollama"); ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--url"); ap.add_argument("--jobs", default="transcript,c0,c2tok")
    ap.add_argument("--stages", default="ALL", help="comma list of calling-function names, or ALL")
    ap.add_argument("--include_responses", action="store_true", help="also show LLM responses (default: prompts only)")
    ap.add_argument("--limit", type=int); ap.add_argument("--prereg")
    a = ap.parse_args()
    cfg = F.load_config(a.config, a.base); secret = F.load_secret(cfg); be = RA.make_backend(a, cfg)
    import pandas as pd
    R = pd.read_csv(a.records).set_index("record_id")
    F.require_unlock(set(R.split), a.prereg)
    Q = {q["QuestionId"]: q for q in json.loads(Path(a.data).read_text())}
    calls = defaultdict(list)
    for l in Path(a.transcripts).read_text().splitlines():
        d = json.loads(l); calls[d["qid"]].append(d)
    keep = None if a.stages == "ALL" else set(a.stages.split(","))
    done = set(); out = Path(a.out)
    if out.exists():
        done = {(j["record_id"], j["job"]) for j in map(json.loads, out.read_text().splitlines())}
    rids = [r for r in R.index if r in calls][: a.limit]
    n = 0
    with out.open("a") as fo:
        for rid in rids:
            cs = sorted((c for c in calls[rid] if keep is None or c["stage"] in keep), key=lambda c: c["call_index"])
            lvl = R.at[rid, "level"]
            for job in a.jobs.split(","):
                if (rid, job) in done:
                    continue
                if job == "cplus" and lvl != "L1":
                    continue                      # positive control: L1 only, dev only (needs labels)
                note = None
                if job == "c0":
                    text = f"KGQA TRAFFIC (anonymized)\nQuestion: {Q[rid]['RawQuestion']}"
                elif job in ("xtract", "xtract_c2tok"):
                    labs_ = sorted({r_.strip().replace("clinical.", "", 1) for c in cs
                                    for s_, rels in re.findall(r"Topic Entity: (m\.p[0-9a-f]+)\nRelations: ([^\n]*)", c["prompt"])
                                    if s_ == R.at[rid, "subj"] for r_ in rels.split(";") if r_.strip()})
                    if job == "xtract_c2tok":
                        labs_ = ["REL" + F.hmac_hex(secret, "c2", rid, x)[:6] for x in labs_]
                    text = (f"KGQA TRAFFIC (anonymized)\nQuestion: {Q[rid]['RawQuestion']}\n"
                            "Relation labels visible to the remote model for the subject:\n" + "\n".join("- " + x for x in labs_))
                else:
                    parts = []
                    for c in cs:
                        parts.append(f"--- LLM call {c['call_index']} ---\n{c['prompt']}" +
                                     (f"\n--- reply ---\n{c['response']}" if a.include_responses else ""))
                    text = "KGQA TRAFFIC (anonymized)\n" + "\n".join(parts)
                    if job == "cplus":
                        assert set(R.split) == {"dev"}, "C+ is a dev-only gate"
                        note = f"the subject's recorded condition = {F.CLASS_NAMES[R.at[rid, 'class_key']]}."
                    if job == "c2tok":
                        labels = sorted({m for m in re.findall(r"clinical\.[a-z0-9_]+", text)} |
                                        {m.split(".", 1)[1] for m in re.findall(r"clinical\.[a-z0-9_]+", text)}, key=len, reverse=True)
                        for lab in labels:
                            text = re.sub(re.escape(lab), "REL" + F.hmac_hex(secret, "c2", rid, lab)[:6], text)
                trunc = len(text) > MAXCH
                if trunc:
                    text = text[:MAXCH]
                system, user, order = F.attacker_prompt(text, rid, secret, note)
                t0 = time.time(); raw = ""; meta = {}; pred = conf = None
                for attempt in range(1 + cfg["attacker"]["retry_on_parse_error"]):
                    try:
                        raw, meta = be(system, user if attempt == 0 else user + "\nYour last reply was not valid JSON. Return JSON only.")
                    except Exception as e:
                        raw, meta = f"ERROR {e}", {}
                    pred, conf = F.parse_prediction(raw)
                    if pred:
                        break
                fo.write(json.dumps(dict(record_id=rid, level=lvl, job=job, view="transcript", pred=pred, conf=conf,
                                         parse_ok=pred is not None, model=a.model if a.backend in ("ollama", "openai") else a.backend,
                                         n_llm_calls=len(cs), chars=len(text), truncated=trunc,
                                         latency_s=round(time.time() - t0, 3),
                                         prompt_sha256=hashlib.sha256((system + user).encode()).hexdigest(), cand_order=order,
                                         raw=raw[:600], meta=meta), ensure_ascii=False) + "\n")
                fo.flush(); n += 1
    print(f"wrote {n} predictions -> {a.out}")


if __name__ == "__main__":
    main()
