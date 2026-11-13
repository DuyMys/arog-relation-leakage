#!/usr/bin/env python3
"""Run an LLM attacker over remote-facing views. The attacker NEVER reads labels.

Jobs (--jobs): triples,names (plain views at the record's level), c2tok (random relation tokens, triples view),
               c0 (query only, one per patient, uses the L2 record), cplus (positive control: needs --labels,
               local evaluator only).
Backends: ollama | openai | mock_keyword | mock_random
"""
import argparse, json, time, urllib.request, random, re, hashlib
from pathlib import Path
import fvlib as F


class Ollama:
    def __init__(s, model, url="http://localhost:11434", num_ctx=16384, seed=20261107, temperature=0):
        s.model, s.url, s.o = model, url.rstrip("/"), dict(temperature=temperature, num_ctx=num_ctx, seed=seed)
    def __call__(s, system, user):
        body = json.dumps(dict(model=s.model, stream=False, format="json", options=s.o,
                               messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])).encode()
        req = urllib.request.Request(s.url + "/api/chat", body, {"Content-Type": "application/json"})
        d = json.loads(urllib.request.urlopen(req, timeout=300).read())
        return d["message"]["content"], dict(prompt_tokens=d.get("prompt_eval_count"), out_tokens=d.get("eval_count"))


class OpenAICompat:
    def __init__(s, model, url, key, temperature=0):
        s.model, s.url, s.key, s.t = model, url.rstrip("/"), key, temperature
    def __call__(s, system, user):
        body = json.dumps(dict(model=s.model, temperature=s.t,
                               messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])).encode()
        req = urllib.request.Request(s.url + "/chat/completions", body,
                                     {"Content-Type": "application/json", "Authorization": "Bearer " + s.key})
        d = json.loads(urllib.request.urlopen(req, timeout=300).read())
        return d["choices"][0]["message"]["content"], dict(usage=d.get("usage"))


class MockKeyword:
    """Test double that 'reads' relation words. NOT a scientific attacker; used only to test the pipeline."""
    RULES = [("lipid", "HLD"), ("glucose", "T2D"), ("metabolic", "T2D"), ("airway", "AsthmaChild"),
             ("respiratory", "AsthmaChild"), ("appendix", "Appendicitis"), ("endoscop", "PolypColon"),
             ("gastrointestinal", "Appendicitis"), ("neurolog", "Seizure"), ("note", None)]
    def __init__(s, seed=0): s.seed = seed
    def __call__(s, system, user):
        body = user.split('"""')[1].lower() if '"""' in user else user.lower()
        m = re.search(r"recorded condition = ([^\n.]+)", user.lower())
        if m:
            for k, v in F.CLASS_NAMES.items():
                if v.lower() in m.group(1):
                    return json.dumps(dict(prediction=v, confidence=1.0, evidence="note")), {}
        for kw, cls in s.RULES:
            if cls and kw in body:
                return json.dumps(dict(prediction=F.CLASS_NAMES[cls], confidence=0.6, evidence=kw)), {}
        rnd = random.Random(int(hashlib.sha256((str(s.seed) + body).encode()).hexdigest()[:8], 16))
        return json.dumps(dict(prediction=F.CLASS_NAMES[rnd.choice(F.CLASSES)], confidence=0.2, evidence="guess")), {}


class MockRandom(MockKeyword):
    def __call__(s, system, user):
        rnd = random.Random(int(hashlib.sha256((str(s.seed) + user).encode()).hexdigest()[:8], 16))
        return json.dumps(dict(prediction=F.CLASS_NAMES[rnd.choice(F.CLASSES)], confidence=0.2, evidence="")), {}


def make_backend(a, cfg):
    t = cfg["attacker"]
    if a.backend == "ollama":
        return Ollama(a.model, a.url or "http://localhost:11434", t["num_ctx"], t["seed"], t["temperature"])
    if a.backend == "openai":
        import os
        return OpenAICompat(a.model, a.url, os.environ["OPENAI_API_KEY"], t["temperature"])
    return MockKeyword(1) if a.backend == "mock_keyword" else MockRandom(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="fv_config.json"); ap.add_argument("--base")
    ap.add_argument("--views", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="ollama"); ap.add_argument("--model", default="qwen2.5:7b"); ap.add_argument("--url")
    ap.add_argument("--jobs", default="triples,names,c0,c2tok"); ap.add_argument("--labels")
    ap.add_argument("--limit", type=int); ap.add_argument("--prereg")
    a = ap.parse_args()
    cfg = F.load_config(a.config, a.base); secret = F.load_secret(cfg); be = make_backend(a, cfg)
    recs = [json.loads(l) for l in Path(a.views).read_text().splitlines()]
    labs = None
    if "cplus" in a.jobs:
        import pandas as pd
        labs = pd.read_csv(a.labels).set_index("record_id")
        F.require_unlock(set(labs.split), a.prereg)
    if a.limit:
        recs = recs[:a.limit]
    done = set()
    out = Path(a.out)
    if out.exists():
        done = {(j["record_id"], j["job"]) for j in map(json.loads, out.read_text().splitlines())}
    jobs = a.jobs.split(","); n = 0
    with out.open("a") as fo:
        for r in recs:
            for job in jobs:
                if job == "c0" and r["level"] != "L2":
                    continue                      # one query-only control per patient
                if job == "cplus" and r["level"] != "L1":
                    continue
                if (r["record_id"], job) in done:
                    continue
                note = None
                if job in ("triples", "names"):
                    text = F.render_remote(r, job)
                elif job == "c2tok":
                    text = F.render_remote(r, "triples", "c2tok", secret)
                elif job == "c0":
                    text = F.render_remote(r, "triples", "c0")
                elif job == "cplus":
                    text = F.render_remote(r, "triples")
                    note = f"the subject's recorded condition = {F.CLASS_NAMES[labs.at[r['record_id'], 'class_key']]}."
                system, user, order = F.attacker_prompt(text, r["record_id"], secret, note)
                t0 = time.time(); raw = ""; meta = {}; pred = conf = None
                for attempt in range(1 + cfg["attacker"]["retry_on_parse_error"]):
                    try:
                        raw, meta = be(system, user if attempt == 0 else user + "\nYour last reply was not valid JSON. Return JSON only.")
                    except Exception as e:
                        raw, meta = f"ERROR {e}", {}
                    pred, conf = F.parse_prediction(raw)
                    if pred:
                        break
                fo.write(json.dumps(dict(record_id=r["record_id"], level=r["level"], job=job, view=("names" if job == "names" else "triples"),
                                         model=a.model if a.backend in ("ollama", "openai") else a.backend,
                                         pred=pred, conf=conf, parse_ok=pred is not None, latency_s=round(time.time() - t0, 3),
                                         prompt_sha256=hashlib.sha256((system + user).encode()).hexdigest(), cand_order=order,
                                         raw=raw[:600], meta=meta), ensure_ascii=False) + "\n")
                fo.flush(); n += 1
    print(f"wrote {n} predictions -> {a.out}")


if __name__ == "__main__":
    main()
