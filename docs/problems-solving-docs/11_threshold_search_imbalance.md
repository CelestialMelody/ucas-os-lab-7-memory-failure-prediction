# 11. 默认 0.5 阈值不适合低正样本比例

## 问题背景

任务二 Stage2 train 约 72459 个样本，正样本约 824 个。正样本比例约 1%。在这种数据上，默认 0.5 阈值通常不能取得最佳 F1。

## 处理方法

对每个模型在验证集上遍历 0.01 到 0.99 的阈值，并按 F1、recall、precision 排序。完整曲线保存为：

- `threshold_search_random_forest.csv`
- `threshold_search_hist_gradient_boosting.csv`
- `threshold_search_lightgbm.csv`
- `threshold_search_xgboost.csv`

关键实现如下。每个阈值都会生成一组二分类预测，并计算 precision、recall 和 F1。

```python
def threshold_search(y_true: pd.Series, score: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in np.linspace(0.01, 0.99, 99):
        pred = (score >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": precision_score(y_true, pred, zero_division=0),
                "recall": recall_score(y_true, pred, zero_division=0),
                "f1": f1_score(y_true, pred, zero_division=0),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["f1", "recall", "precision"],
        ascending=False,
    )
```

## 结果

baseline XGBoost 在阈值 0.26 处取得 F1 = 0.5405。agent 最佳策略在阈值搜索和 per-type 建模后取得 F1 = 0.5538。

## 结论

任务二在验证集上搜索阈值。该步骤降低了低正样本比例下固定阈值带来的偏差。
