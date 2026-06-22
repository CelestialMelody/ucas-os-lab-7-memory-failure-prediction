# Method

## Stage2 Feature Pipeline

The pipeline reads the official extracted feature feather files, builds a
supervised training table, splits by prediction timestamp, trains candidate
models, searches thresholds, and generates submission-format candidate rows.

Positive samples are built from failed DIMMs in the prediction window before the
failure time. Negative samples are built from earlier safe points or from
non-failed DIMMs.

## Models

The candidate model set includes:

- RandomForest;
- HistGradientBoosting;
- LightGBM;
- XGBoost.

SmartHW starter materials use XGBoost and LightGBM-style baselines, while
RandomForest and HistGradientBoosting are local comparison baselines.

## Agent Strategy Search

The agent enumerates:

- sampling: none, balanced, hard negative;
- feature set: all, temporal, spatial/bit;
- decision rule: threshold search or top-k;
- model family: RF, HGB, LightGBM, XGBoost;
- per-type split: one global model or separate type_A/type_B models.

The best local validation strategy is `xgb_all_none_per_type`.

## Submission Generation

The best submission path retrains per-type XGBoost models and scans the Stage2
target period. If no target-period sample exceeds the validation threshold, it
falls back to top-k highest-risk DIMMs. This guarantees a valid offline candidate
file, but it is not a hidden-test score.
