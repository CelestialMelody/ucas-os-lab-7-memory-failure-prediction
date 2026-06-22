# Method

## Baseline Features

The baseline aggregates CE logs over 15m, 1h, 6h, and 24h windows. It includes CE
counts, READ/SCRUB counts, unique device/bank/row/column counts, CE storm counts,
and low-level retry parity statistics.

## BSFE

BSFE converts `RetryRdErrLogParity` into a 32-bit pattern and reshapes it as an
8x4 DQ-Beat matrix. It then extracts row-wise and column-wise statistics such as
bit count, bit interval, maximum consecutive length, and pooled spatial signals.

## Time-Patch

Time-patch computes multi-window features over 15m, 1h, and 6h. Each window
contains temporal CE intensity, spatial fault mode, and aggregated BSFE features.

## Time-Point

Time-point captures event-level failure precursors:

- latest CE distance to the prediction timestamp;
- short-window versus long-window CE ratio;
- longest CE run within 60s and 300s;
- dominant device/bank/row/column/cell ratios.

## Evaluation

The project uses Logistic Regression and a lightweight RandomForest as local
evaluators. This keeps the focus on whether M2-MFP-style features are useful,
not on heavy model tuning. Public M2-MFP code contains a custom decision-tree
time-point module, but not a complete private training pipeline.
