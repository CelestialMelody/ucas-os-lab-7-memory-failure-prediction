# Reproducibility

## Setup

From the repository root:

```bash
uv venv .venv
uv pip install -r requirements.txt
```

Or from this subproject:

```bash
uv pip install -r requirements.txt
```

## Use Cached Features

```bash
make restore-caches
make task2-agent-quick
```

## Rebuild from Stage2 Feature Data

Set the data path:

```bash
export MEMFAIL_DATA_ROOT=/path/to/data
```

Then run:

```bash
make task2-train
make task2-agent
make task2-best-submission
```

`task2-train` rebuilds `stage2_train_features.csv` from the official extracted
feature feather files. `task2-agent` reuses that cache unless explicitly rebuilt.

## Outputs

- Model metrics: `results/stage2_feature/stage2_metrics.csv`.
- Agent metrics: `results/stage2_feature/agent_ablation_results.csv`.
- Best config: `results/stage2_feature/agent_best_config.json`.
- Offline candidate submission:
  `results/stage2_feature/stage2_best_per_type_submission.csv`.
