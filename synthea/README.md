# Synthea data generation and patient selection

Generation: 01_generate_synthea.ipynb (Kaggle, CPU, JDK 17+).
Synthea commit bd8a52b, seed 20261104 (-s, -cs, -r, -e), -p 5000 -o false,
Massachusetts, generate.thread_pool_size = 2. Row order can vary with several
threads, so metadata.json stores the SHA-256 of each CSV from the run used in the paper.

Screening: analyze_synthea.py (integrity checks and condition-code counts).

Selection and splits: reproduce_selection.py. Takes the six target codes, keeps
patients with exactly one of them, orders each class by SHA-256("20261104|D2_v1|<patient_id>")
and takes the first 110 patients (40/10/10/40/10).

Not released: the relation mapping used in the experiments, the node-identifier map,
the identifier key, and the generated patient-level files. The files
splits_PRIVATE.csv and eligible_single_target_private.csv contain class labels.