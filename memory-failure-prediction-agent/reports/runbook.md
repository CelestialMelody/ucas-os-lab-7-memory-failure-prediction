# Runbook: Memory Failure Prediction Agent

All commands below are run from `memory-failure-prediction-agent/` unless noted
otherwise.

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

The repository stores the Stage2 training table as a compressed cache:

```bash
make restore-caches
```

This creates `results/stage2_feature/stage2_train_features.csv`, which is
ignored by Git.

## Data Path

For full rebuilds, set:

```bash
export MEMFAIL_DATA_ROOT=/path/to/data
```

The data directory must contain:

```text
stage2_feature/
├── failure_ticket.csv
├── type_A/*.feather
└── type_B/*.feather
```

## Commands

Quick agent check:

```bash
make task2-agent-quick
```

Rebuild training table and baseline metrics:

```bash
make task2-train
```

Run full agent strategy search:

```bash
make task2-agent
```

Generate the best per-type offline submission:

```bash
make task2-best-submission
```

## Key Outputs

```text
results/stage2_feature/stage2_metrics.csv
results/stage2_feature/agent_ablation_results.csv
results/stage2_feature/agent_best_config.json
results/stage2_feature/stage2_best_per_type_submission.csv
reports/figures/
```
