# Lab 4: Data pipeline and quality (UCI Adult)

## Run it (one command)
```
pip install -r requirements.txt
python run_pipeline.py
```
It downloads Adult from OpenML (needs internet the first time) and writes everything below.

## What it produces
| File | Lab step |
|---|---|
| `problem_definition.md` | 1. problem definition |
| `data/raw/adult_raw.csv`, log lines `INGEST rows=... cols=...` | 2. ingestion (raw copy, unmodified) |
| `results/validation_report.md/.json` | 3. validation (nothing dropped) |
| `results/preprocessor.joblib` | 4. fitted cleaner (train-only statistics) |
| `results/baseline_metrics.json` | 5. LogisticRegression with and without the 2 new features |
| `results/label_quality.json`, `results/audit_sample_50.csv` | 6. duplicate-conflict audit + 50 rows to check by hand |
| `results/run_manifest.yaml`, `data/processed/*.csv`, `data_card.md` | 7. SHA256 hashes + config + data card |

## No leakage
Split 70/30 stratified first. Clip limits, medians and modes are learned on train only. Exact
duplicates are dropped from train only. The test set is only transformed.

## Still to do by hand
Open `results/audit_sample_50.csv`, check the 50 rows against the data dictionary and write what you found
in `results/manual_audit_notes.md`.
