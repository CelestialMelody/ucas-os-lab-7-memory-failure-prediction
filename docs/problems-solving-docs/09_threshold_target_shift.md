# 9. 阈值搜索和目标期分布存在漂移

## 问题背景

验证集最佳阈值来自有标签数据。目标预测期无标签，分数分布可能低于验证期。若直接套用验证阈值，目标期可能没有任何样本被选中。

## 问题表现

best per-type submission 生成时，目标期样本分数均低于 per-type 验证阈值。严格按阈值筛选会得到空 submission。

## 处理方法

提交生成分两步执行。第一步按验证阈值筛选。第二步检查是否有候选；若没有候选，则记录每个 DIMM 在目标期的最高风险点，并取 top-200。

best per-type submission 的关键逻辑如下。每个 DIMM 先保留目标期最高风险点。阈值没有选出样本时，使用 `submission_top_k` 生成候选文件。

```python
best_any = target_df.sort_values("score", ascending=False).iloc[0]
all_score_rows.append(
    {
        "serial_number": sn_name,
        "prediction_timestamp": int(best_any["LogTime"]),
        "serial_number_type": sn_type,
        "score": float(best_any["score"]),
        "threshold": float(thresholds.get(sn_type, 0.5)),
    }
)

if submission.empty and config.submission_top_k > 0:
    all_score_df = pd.DataFrame(all_score_rows).sort_values("score", ascending=False)
    score_df = all_score_df.head(config.submission_top_k)
    submission = score_df[["serial_number", "prediction_timestamp", "serial_number_type"]].copy()
```

## 结果

最终 submission 是 200 行离线候选文件。它表示目标期相对风险最高的 DIMM 时间点，不表示 hidden-test 效果。

## 结论

submission 是离线候选结果。目标期阈值未命中后使用 top-k 兜底。候选文件对应提交格式和风险排序。当前 Codabench 评估入口关闭，项目没有 hidden-test F1。
