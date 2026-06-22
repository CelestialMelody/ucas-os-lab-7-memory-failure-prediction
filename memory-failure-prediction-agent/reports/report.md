# Memory Failure Prediction Report

## Task Definition

The project predicts future uncorrectable memory failures from historical
correctable error logs and extracted memory features. It contains two
subprojects:

1. M2-MFP-style feature reproduction on Stage1 raw CE logs.
2. SmartHW/SmartMem Stage2 model training, agent strategy search, and offline
   candidate submission generation.

## Data

| Data | Size | Usage |
| --- | ---: | --- |
| `stage1_feather` | about 33 GB | Raw-log feature reproduction |
| `stage2_feature` | about 7.9 GB | Model training and agent search |
| `stage2_feather` | about 46 GB | Optional future raw-log enhancement |

The Git repository stores compressed intermediate feature caches and lightweight
result files. Original feather data is kept outside Git and referenced with
`MEMFAIL_DATA_ROOT`.

## M2-MFP Reproduction

The reproduction implements three feature groups:

- BSFE: bit-level features from the 8x4 DQ-Beat parity matrix.
- Time-patch: temporal, spatial, and bit features over 15m, 1h, and 6h windows.
- Time-point: CE burst, recent gap, short/long ratio, and spatial concentration
  features near the prediction time.

Local evaluation uses Logistic Regression and a lightweight RandomForest. The
focus is feature effectiveness and ablation, with public M2-MFP code used as
the reference for core feature ideas.

| Dataset | Best ablation | Model | Precision | Recall | F1 |
| --- | --- | --- | ---: | ---: | ---: |
| type_A smoke | baseline_time_point | random_forest_small | 0.7078 | 0.7474 | 0.7270 |
| type_A full | baseline_time_point | random_forest_small | 0.6372 | 0.9375 | 0.7587 |
| type_B full | baseline_time_point | random_forest_small | 0.6216 | 0.6765 | 0.6479 |
| type_A + type_B full | baseline_time_point | random_forest_small | 0.6355 | 0.9593 | 0.7645 |

The results show that time-point features are the strongest and most stable
feature group in this local reproduction. BSFE and time-patch add interpretable
spatial and bit-level signals, with weaker aggregate F1 than time-point under
the RandomForest configuration used here.

## Stage2 Agent Optimization

The Stage2 pipeline uses official extracted features. It builds a supervised
training table, performs timestamp-based validation, trains model candidates,
searches decision thresholds, and generates submission-format candidate rows.

The agent search covers:

- RandomForest, HistGradientBoosting, LightGBM, and XGBoost.
- none, balanced, and hard-negative sampling.
- all, temporal, and spatial/bit feature subsets.
- threshold, top-k, and per-type decision strategies.

Baseline model comparison:

| Model | Threshold | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| xgboost | 0.26 | 0.4813 | 0.6164 | 0.5405 |
| hist_gradient_boosting | 0.26 | 0.4607 | 0.5616 | 0.5062 |
| lightgbm | 0.80 | 0.3887 | 0.7055 | 0.5012 |
| random_forest | 0.74 | 0.4053 | 0.6301 | 0.4933 |

Top agent strategies:

| Strategy | Precision | Recall | F1 | Threshold |
| --- | ---: | ---: | ---: | ---: |
| xgb_all_none_per_type | 0.5028 | 0.6164 | 0.5538 | 0.28 |
| xgb_all_none_threshold | 0.4863 | 0.6096 | 0.5410 | 0.27 |
| xgb_temporal_none_threshold | 0.5030 | 0.5753 | 0.5367 | 0.31 |
| lgb_all_none_per_type | 0.3877 | 0.7329 | 0.5071 | 0.78 |
| rf_all_balanced_per_type | 0.4272 | 0.6233 | 0.5070 | 0.79 |

The best local validation strategy is XGBoost with all features and per-type
training. It improves F1 from 0.5405 to 0.5538.

## Submission File

The best per-type submission contains 200 rows with no null values and no
duplicate serial numbers. Target-period scores did not exceed the validation
thresholds, so the submission uses top-k fallback. The file is an offline
candidate output and does not include a Codabench hidden-test score.

## Conclusion

The project completes both experiment goals:

- M2-MFP-style raw-log features were implemented and evaluated on Stage1 data.
- A Stage2 agent search improved local validation F1 over the baseline XGBoost
  model and produced a valid offline candidate submission.
