# Runbook: M2-MFP Reproduction

All commands below are run from `m2mfp-reproduction/` unless noted otherwise.

## Environment

From the repository root:

```bash
uv venv .venv
uv pip install -r requirements.txt
```

Or from this subproject:

```bash
uv pip install -r requirements.txt
```

## Restore Cached Features

The repository stores large feature tables as `.csv.gz` files. Restore them with:

```bash
make restore-caches
```

This creates uncompressed CSV files under `results/`. Those files are ignored by
Git.

## Data Path

For raw-data rebuilds, set:

```bash
export MEMFAIL_DATA_ROOT=/path/to/data
```

The data directory must contain:

```text
stage1_feather/
├── ticket.csv
├── type_A/*.feather
└── type_B/*.feather
```

## Commands

Quick smoke:

```bash
make task1-smoke MAX_FILES=300
```

Run ablation from existing feature cache:

```bash
make task1-ablation
```

Full rebuilds from Stage1 raw logs:

```bash
make task1-full-a
make task1-full-b
make task1-full-ab
```

## Key Outputs

```text
results/stage1_type_a_full_metrics_ablation.csv
results/stage1_type_b_full_metrics_ablation.csv
results/stage1_type_ab_full_metrics_ablation.csv
reports/figures/
```

The full feature caches are stored as:

```text
results/stage1_type_a_full_features.csv.gz
results/stage1_type_b_full_features.csv.gz
results/stage1_type_ab_full_features.csv.gz
```
