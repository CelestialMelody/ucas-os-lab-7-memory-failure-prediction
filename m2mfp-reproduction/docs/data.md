# Data

## Expected Data Layout

The repository does not include the original feather data. Place the data outside
Git and set `MEMFAIL_DATA_ROOT` to the directory that contains `stage1_feather/`.

```text
data/
└── stage1_feather/
    ├── ticket.csv
    ├── type_A/*.feather
    └── type_B/*.feather
```

The experiments in this repository were run with `stage1_feather`, about 33 GB:

- type_A: 56,403 feather files.
- type_B: 5,821 feather files.

## Labels

`ticket.csv` provides failure DIMM serial numbers and failure timestamps. A
positive sample is built from logs before the failure time, leaving a 15-minute
lead time. Negative samples are built from an earlier safe point for failed DIMMs
or from the last observed point for non-failed DIMMs.

## Cached Feature Tables

The full raw logs are large. This repository includes compressed feature caches:

```text
results/stage1_type_a_full_features.csv.gz
results/stage1_type_b_full_features.csv.gz
results/stage1_type_ab_full_features.csv.gz
```

Restore them with:

```bash
make restore-caches
```

The uncompressed CSV files are ignored by Git because some exceed 100 MB.
