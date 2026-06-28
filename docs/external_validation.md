# External Validation Guide

This guide describes how to validate the project from a clean environment.

## Repository-Only Validation

The Git repository includes code, configs, documentation, metrics, figures, submission examples, and compressed feature caches. A machine without the original feather data can still validate the environment and inspect cached results.

```bash
uv venv .venv
uv pip install -r requirements.txt
make restore-caches
make compile
make figures
```

After `make restore-caches`, cached feature CSV files are restored from committed `.csv.gz` files. These restored CSV files are ignored by Git. This command exists to avoid requiring tens of GB of raw feather data for basic validation. It only decompresses committed cache artifacts, so it can validate the code path and cached results quickly; it does not download the original datasets and it is not a substitute for full raw-data rebuilds.

Useful repository-only checks:

```bash
make -C m2mfp-reproduction task1-ablation
make -C memory-failure-prediction-agent task2-agent-quick
```

## Full Raw-Data Validation

Full rebuilds require the SmartMem/SmartHW datasets. Place data outside Git and set `MEMFAIL_DATA_ROOT`:

```bash
export MEMFAIL_DATA_ROOT=/path/to/data
```

Expected layout:

```text
data/
├── stage1_feather/
│   ├── ticket.csv
│   ├── type_A/*.feather
│   └── type_B/*.feather
├── stage2_feature/
│   ├── failure_ticket.csv
│   ├── type_A/*.feather
│   └── type_B/*.feather
└── stage2_feather/          # optional for future extensions
```

Full commands:

```bash
make task1-full-a
make task1-full-b
make task1-full-ab
make task2-agent
make task2-best-submission
```

## Data Sources

Primary public entry points:

- SmartMem / Codabench competition page: `https://www.codabench.org/competitions/3586/`
- SmartMem dataset DOI: `https://zenodo.org/records/15516113`
- Stage2 extracted feature dataset: `https://www.kaggle.com/datasets/smartmem/smartmem-features`
- SmartHW repository: `https://github.com/hwcloud-RAS/SmartHW`
- M2-MFP repository: `https://github.com/hwcloud-RAS/M2-MFP`

Stage1 raw-log reproduction uses `stage1_feather`. Stage2 model training and agent search use `stage2_feature`. The `stage2_feather` raw logs are documented as an extension path.
