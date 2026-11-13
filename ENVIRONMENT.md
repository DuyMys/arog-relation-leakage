# Environment

Values below come from the logs of the test run on Kaggle, 2026-11-10 (times in UTC).

## ARoG

- Repository: github.com/NLPGM/ARoG, commit `4867cf5168f682bc0bd8428158dd45919280ee83`
- Cloned 2026-11-10 15:48:50, before the pre-registration lock (16:22:20)
- Changes: `arog_run/arog_modifications.diff` (see README)
- Command: `main_freebase.py --dataset webqsp --width 3 --depth 3`
- Python 3.10.22 in a separate virtual environment built from the upstream `requirements.txt`, PyTorch for CUDA 11.8

## LLM (ARoG and attacker A1)

- Ollama 0.40.16
- Qwen2.5 7B Instruct, GGUF Q4_K_M, blob `sha256-2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730` (4.4 GiB)
- Temperature 0; context 16384 for every model load after the lock (16:28, 18:50, 20:22). The one load before the lock (15:45) used 32768.
- 0 of 7,806 calls were truncated; the longest prompt was 5,018 tokens.
- Temperature 0 is not fully deterministic on GPU: C0 control accuracy varied between 13.3% and 16.7% across five runs.

## Knowledge-graph store

Virtuoso Open Source 07.20.3229 (Ubuntu package `virtuoso-opensource-7`).

## Hardware

Kaggle, 2 x Tesla T4 (14.6 GiB, compute capability 7.5, CUDA driver 13.0); the model ran on one GPU.

## Attacks and statistics

Python 3.13.16, Linux 6.18.55, glibc 2.39; scikit-learn 1.6.1, scipy 1.16.3, numpy 2.1.3, pandas 2.3.3. Full package list: `ENV.json`.

## Rescoring after the lock (2026-11-11)

The ablation metrics and the post-hoc analyses were computed on Windows with Python 3.14.0, scipy 1.18.1, pandas 2.3.3 and numpy 2.3.4. The primary endpoint reproduced exactly (A3 accuracy 0.7167 / 0.6500 / 0.1667 for L0 / L1 / L2).
