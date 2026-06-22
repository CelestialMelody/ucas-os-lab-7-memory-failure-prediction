# Results

## Baseline Models

| Model | Threshold | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| xgboost | 0.26 | 0.4813 | 0.6164 | 0.5405 |
| hist_gradient_boosting | 0.26 | 0.4607 | 0.5616 | 0.5062 |
| lightgbm | 0.80 | 0.3887 | 0.7055 | 0.5012 |
| random_forest | 0.74 | 0.4053 | 0.6301 | 0.4933 |

## Agent Search

| Strategy | Precision | Recall | F1 | Threshold |
| --- | ---: | ---: | ---: | ---: |
| xgb_all_none_per_type | 0.5028 | 0.6164 | 0.5538 | 0.28 |
| xgb_all_none_threshold | 0.4863 | 0.6096 | 0.5410 | 0.27 |
| xgb_temporal_none_threshold | 0.5030 | 0.5753 | 0.5367 | 0.31 |
| lgb_all_none_per_type | 0.3877 | 0.7329 | 0.5071 | 0.78 |
| rf_all_balanced_per_type | 0.4272 | 0.6233 | 0.5070 | 0.79 |

The agent improves local validation F1 from 0.5405 to 0.5538.

## Offline Submission

`stage2_best_per_type_submission.csv` contains 200 rows with:

- no null values;
- no duplicate `serial_number`;
- timestamps in the target prediction period.

The target-period scores did not exceed the per-type validation thresholds, so
the final candidate list uses top-k fallback. This file is a valid offline
candidate output, not a Codabench hidden-test result.

## Figures

Figures are stored in `reports/figures/`:

- `task2_model_comparison.png`
- `task2_threshold_f1.png`
- `task2_threshold_f1_panels.png`
- `stage2_label_distribution.png`
- `agent_strategy_ablation.png`
- `task2_best_submission_scores.png`
- `task2_winners_reference.png`
