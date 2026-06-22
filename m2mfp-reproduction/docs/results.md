# Results

## Full-Data Ablation

| Dataset | Best ablation | Model | Precision | Recall | F1 |
| --- | --- | --- | ---: | ---: | ---: |
| type_A smoke | baseline_time_point | random_forest_small | 0.7078 | 0.7474 | 0.7270 |
| type_A full | baseline_time_point | random_forest_small | 0.6372 | 0.9375 | 0.7587 |
| type_B full | baseline_time_point | random_forest_small | 0.6216 | 0.6765 | 0.6479 |
| type_A + type_B full | baseline_time_point | random_forest_small | 0.6355 | 0.9593 | 0.7645 |

## Interpretation

Time-point is the most stable feature group across type_A, type_B, and combined
data. BSFE and time-patch improve some settings, especially type_B compared with
baseline, but they do not consistently outperform time-point alone under the
RandomForest configuration used in this project.

## Figures

Figures are stored in `reports/figures/`:

- `task1_ablation_f1.png`
- `task1_type_a_full_ablation_f1.png`
- `task1_type_b_full_ablation_f1.png`
- `task1_type_ab_full_ablation_f1.png`
