# OS Lab 7: Memory Failure Prediction

This repository contains a course project for memory failure prediction. It is
organized as two reproducible subprojects:

- `m2mfp-reproduction/`: reproduces and evaluates M2-MFP-style raw-log features
  on SmartMem Stage1 CE logs.
- `memory-failure-prediction-agent/`: trains models, searches agent strategies,
  and generates an offline candidate submission on SmartHW/SmartMem Stage2
  extracted features.

The repository intentionally does not include the original GB-scale feather
datasets. It includes code, configs, reports, figures, lightweight metrics, and
compressed intermediate feature caches so that experiments can be inspected
without immediately re-reading all raw feather files.

## Quick Start

Install dependencies with `uv` from the repository root:

```bash
uv venv .venv
uv pip install -r requirements.txt
```

Restore committed compressed feature caches:

```bash
make restore-caches
```

Run static compilation checks:

```bash
make compile
```

Run quick smoke checks:

```bash
make task1-smoke MAX_FILES=300
make task2-agent-quick
```

For full runs, put the data outside Git and point `MEMFAIL_DATA_ROOT` to the
directory that contains `stage1_feather/`, `stage2_feature/`, and optionally
`stage2_feather/`:

```bash
make task1-full-ab MEMFAIL_DATA_ROOT=/path/to/data
make task2-agent MEMFAIL_DATA_ROOT=/path/to/data
```

## Data Used

The experiments were run with these local data directories:

| Data directory | Size | Used by |
| --- | ---: | --- |
| `data/stage1_feather` | about 33 GB | M2-MFP raw-log reproduction |
| `data/stage2_feature` | about 7.9 GB | Stage2 model training and agent search |
| `data/stage2_feather` | about 46 GB | Optional future raw-log enhancement, not used by the default pipeline |

Original data should be downloaded from the SmartMem/SmartHW release sources or
competition mirrors described in each subproject's `docs/data.md`.

## Results Summary

Task 1 best results:

| Dataset | Best ablation | Model | F1 |
| --- | --- | --- | ---: |
| type_A smoke | baseline_time_point | random_forest_small | 0.7270 |
| type_A full | baseline_time_point | random_forest_small | 0.7587 |
| type_B full | baseline_time_point | random_forest_small | 0.6479 |
| type_A + type_B full | baseline_time_point | random_forest_small | 0.7645 |

Task 2 best results:

| Method | Local validation F1 |
| --- | ---: |
| baseline XGBoost | 0.5405 |
| agent `xgb_all_none_per_type` | 0.5538 |

The Stage2 submission file is an offline candidate output. It has not been
submitted to Codabench for a hidden-test F1 score.

## Repository Layout

```text
oslab-7-memory-failure-prediction/
├── m2mfp-reproduction/
├── memory-failure-prediction-agent/
├── requirements.txt
├── Makefile
└── README.md
```

Each subproject contains its own `README.md`, `docs/`, `configs/`, `src/`,
`results/`, `reports/`, and `Makefile`.
