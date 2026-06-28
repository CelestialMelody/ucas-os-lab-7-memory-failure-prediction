# 6. type_B 正样本少，细小 F1 差异解释范围受限

## 问题背景

Stage1 type_B full 读取 5821 个 feather 后，监督样本中的正样本只有 68 个。正样本少时，单个样本的预测变化会明显影响 precision、recall 和 F1。

## 现象

type_B full 中 `baseline_time_patch`、`baseline_time_point` 和 full 的 F1 都在 0.647 左右。三者差异很小。

| Ablation | F1 |
| --- | ---: |
| baseline_time_patch | 0.6471 |
| baseline_time_point | 0.6479 |
| full | 0.6471 |

消融脚本会在 summary JSON 中记录样本数、正样本数和 full 特征列数。这些数值用于判断指标解释范围。

```python
summary = {
    "samples": int(len(full_features)),
    "positive_samples": int(full_features["label"].sum()),
    "feature_count_full": len(feature_columns(full_features)),
}
```

## 处理方法

结果表保留完整数值。正文只记录 baseline 到 M2-MFP 风格特征的整体提升，以及 type_B 正样本规模对细小差异的影响。

## 结果

任务一总体结论主要由 type_A full 和 A+B full 支撑。type_B full 作为补充证据，用于记录 M2-MFP 风格特征在 type_B 上的信号。

## 结论

type_B 正样本数量较少。0.001 级别 F1 差异不足以支持稳定排序。该组结果记录 baseline 到 M2-MFP 风格特征的整体提升。
