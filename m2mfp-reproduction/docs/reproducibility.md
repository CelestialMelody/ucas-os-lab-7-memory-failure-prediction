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
make task1-ablation
```

## Rebuild from Raw Data

Set the data path:

```bash
export MEMFAIL_DATA_ROOT=/path/to/data
```

Then run:

```bash
make task1-smoke MAX_FILES=300
make task1-full-a
make task1-full-b
make task1-full-ab
```

Full runs read the 33 GB Stage1 raw feather data and can take tens of minutes.

## Outputs

- Feature caches: `results/*_features.csv`.
- Ablation metrics: `results/*_metrics_ablation.csv`.
- Run summaries: `results/*_metrics_ablation.json`.
