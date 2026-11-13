# Relation-label leakage in ARoG prompts on a Synthea knowledge graph

This repository contains the code, pre-registration lock, LLM prompt logs and results of an audit of [ARoG](https://github.com/NLPGM/ARoG) (Privacy-protected Retrieval-Augmented Generation for KGQA, AAAI 2026). ARoG replaces entity names with random IDs before calling a remote LLM, but it sends relation names unchanged. We ran the released ARoG pipeline on a synthetic medical knowledge graph built from [Synthea](https://github.com/synthetichealth/synthea), logged every prompt sent to the LLM, and measured how well a remote party reading those prompts can infer each patient's disease group.

The work accompanies a paper submitted to ACIIDS 2027. All patient data is synthetic.

## Main results

Test set: 120 patients (20 per class, 6 classes, chance = 16.7%), each queried at three relation-label granularities that share the same edges. L0 uses detailed labels (e.g. `neurologic_therapy_regimen`), L1 groups them by organ system or drug class (e.g. `neurologic_medication`), L2 keeps only `prescribed` and `underwent`.

| Attacker | Input | L0 | L1 | L2 |
| --- | --- | --- | --- | --- |
| A3 logistic regression (primary endpoint) | relation-label set parsed from the ARoG prompts | 71.7% [63.0, 79.0] | 65.0% [56.1, 72.9] | 16.7% [11.1, 24.4] |
| A1 Qwen2.5-7B, zero-shot | extracted relation-label list | 38.3% | 32.5% | 12.5% |
| A1 Qwen2.5-7B, zero-shot | full raw prompts | 20.8% | 15.0% | 16.7% |
| Local control (not visible remotely) | node degree + birth year, full graph | 36.7% | 36.7% | 36.7% |

Brackets are Wilson 95% intervals. The primary endpoint rejects chance at L0 and L1 (one-sided exact binomial, Holm, p < 0.001); a label-permutation control gives 16.3% and 16.7%. Collapsing labels to L2 removes the signal, but the share of ARoG answers containing at least one correct entity drops from 100% (L1) to 84.2% (L2) (McNemar p < 0.001; post-hoc, see `posthoc/`).

## Repository layout

The repository root mirrors `/kaggle/working` of the test run, so paths in the notebook resolve unchanged.

| Path | Contents |
| --- | --- |
| `aciids-main-run.ipynb` | The notebook that ran the test once on Kaggle (2026-11-10) |
| `PREREG_LOCK.json` | Pre-registration lock: analysis plan and SHA-256 of code, mapping, split and dev outputs, written before the test split was read |
| `verify_lock.py` | Checks the files in this repository against the lock |
| `aciids_fv/` | Attack, statistics and data-building code (`fvlib.py`, `build_views.py`, `run_attack.py`, `evaluate.py`, `stat_attackers.py`, `make_prereg.py`, `selftest.py`) |
| `aciids_fv/protoA/` | Protocol A: projecting the KG into ARoG's Freebase format (`make_arog_data.py`), the logging patch for ARoG (`patch_arog.py`), attacks on the prompt logs (`run_transcript_attack.py`, `stat_on_transcripts.py`) |
| `aciids_fv/out/` | Anonymised views for train, dev and test (no labels) and the dev results hashed into the lock |
| `ACIIDS_2027/ACIIDS_T04/configs/banned_tokens.txt` | Patterns that must never appear in anything sent to the LLM (disease names, synonyms, SNOMED codes, UUIDs) |
| `testA/` | Test inputs for ARoG (`WebQSP.json`, `FilterFreebase`), attacker predictions (`pred_*.jsonl`) and metrics (`metrics_*.csv`, `stat_transcripts.csv`, `stat_fixed_all240.csv`) |
| `arog_transcripts_test.jsonl` | All 3,223 prompts and responses ARoG exchanged with the LLM during the test run |
| `arog_run/` | ARoG outputs per chunk, ARoG's own result file and log, and `arog_modifications.diff` (changes to upstream ARoG) |
| `posthoc/` | Analyses added after the lock: ARoG utility by level, per-class recall, local controls on the 120 test patients, attack advantage against the same-level C0 control |
| `ENVIRONMENT.md` | Software and hardware versions |

## Protocol

1. The KG (660 patients, 6 classes of 110) was split by patient into train 40, validation 10, dev 10, test 40 and reserve 10 per class. Test and reserve are locked in code.
2. Each patient's view keeps the top 15 medications and top 15 procedures by event count; condition and city edges are dropped. L0, L1 and L2 views of a patient contain the same edges; only relation names and session IDs differ.
3. Attacks and controls were designed on dev. The positive-control gate (attacker told the answer, C+ >= 0.90, parse rate >= 0.95) passed with 1.00 and 1.00.
4. `make_prereg.py` wrote `PREREG_LOCK.json` at 2026-11-10 16:22:20 UTC. SHA-256 of the lock file: `52a23cac6d97593e05e5481cde9dd22342b0316248c892404108eb117796c62c`.
5. The test data was built and ARoG was run once over 360 records (3 chunks). Every script that reads the test split re-checks the lock hashes.

Deviations recorded in the lock before the test run: the primary endpoint was moved from the zero-shot LLM attacker (A1) to the statistical attacker (A3) after dev results; the `xtract` attacker job was added after dev; the test split had been partly exposed: before the splits were drawn, disease labels and class-level statistics of the full source cohort, including later test patients, were examined during KG construction. No attacker had been run on test patients.

Two known issues after the lock. The stage-ablation evaluation failed inside the notebook (`$PREREG` is not expanded inside a Python `for` loop in IPython) and was rerun separately; its outputs are `testA/metrics_abl_*.csv`. `evaluate.py` compares every level against the C0 control of a single level; same-level comparisons are in `posthoc/c0_same_level.csv` and do not change any conclusion.

## Changes to ARoG

Upstream commit `4867cf5168f682bc0bd8428158dd45919280ee83`. `arog_run/arog_modifications.diff` applies to it with `git apply --ignore-whitespace` and contains:

- `utils.py`: `run_llm` calls a local Ollama server instead of the OpenAI API (temperature from ARoG's arguments, `num_ctx` 16384, seed 20261107); every prompt and response is appended to the JSONL file named by `AROG_LOG`; relation parsing reads quoted strings instead of calling `eval` on model output.
- `main_freebase.py`: records the current question ID for the log.
- `freebase_func.py`: local SPARQL and sentence-encoder endpoints; SPARQL queries are no longer printed.
- `lm_server.py`: the sentence encoder runs on CUDA.

The search logic, prompts and parameters of ARoG are unchanged (`--dataset webqsp --width 3 --depth 3`). The relation names reach the LLM through ARoG's own `relation_search_prune` stage.

## What is not public

The following files are needed to rerun the pipeline but are kept private: the HMAC key that derives all IDs, the patient table, the KG node and edge files, the relation mapping (`relation_mapping_v2_1_CANDIDATE_PRIVATE.csv`), the split manifest, and every `*_PRIVATE.csv` file holding class labels. Without them the predictions in `testA/` cannot be scored. `verify_lock.py` reports 25 locked files that match this repository and 2 that are private (the mapping and the split manifest). The private files are available to reviewers on request.

## Running

The notebook targets Kaggle (GPU T4, internet on) with the code and private data attached as a dataset. To check the lock:

```bash
python verify_lock.py
```

To rerun the post-hoc analyses (requires the private files):

```bash
python posthoc_analysis.py --fv aciids_fv --base ACIIDS_2027 --testA testA \
  --transcripts arog_transcripts_test.jsonl \
  --arog_result arog_run/ARoG_0720_naive_1_webqsp_True_True_depth_3_width_3_-1.jsonl --out posthoc
```

## Scope

Results hold for this setup: Synthea data, a closed set of 6 balanced classes, Qwen2.5-7B-Instruct for both ARoG and the LLM attacker, ARoG as released, and a relation mapping written by the authors that has not been clinically validated. Synthea prescribes medications by fixed disease modules, so drug-class labels are more informative here than they may be in real records.
