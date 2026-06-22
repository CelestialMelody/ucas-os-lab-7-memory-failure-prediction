# Data

## Expected Data Layout

The repository does not include original GB-scale data. Place the data outside
Git and set `MEMFAIL_DATA_ROOT` to the directory that contains `stage2_feature/`.

```text
data/
└── stage2_feature/
    ├── failure_ticket.csv
    ├── readme.md
    ├── type_A/*.feather
    └── type_B/*.feather
```

The experiments in this repository were run with `stage2_feature`, about 7.9 GB:

- type_A: 64,794 feather files.
- type_B: 7,175 feather files.

`stage2_feather`, about 46 GB, contains raw logs and is not used by the default
pipeline. It is kept as a future enhancement path.

## Training Table

`stage2_train_features.csv` is not an original downloaded file. It is generated
by `src/stage2_feature_pipeline.py` from the official extracted feature feather
files. The generated table has:

- 72,459 samples;
- 824 positive samples;
- 71,635 negative samples;
- 100 model feature columns.

## Cached Feature Table

The repository includes the compressed cache:

```text
results/stage2_feature/stage2_train_features.csv.gz
```

Restore it with:

```bash
make restore-caches
```
