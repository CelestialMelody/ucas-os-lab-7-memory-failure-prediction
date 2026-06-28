# 17. 目标期分数整体低于验证阈值

## 问题背景

任务二的最佳 agent 策略为 `xgb_all_none_per_type`。该策略在本地验证集上分别为 type_A 和 type_B 搜索阈值。submission 生成阶段使用同一策略遍历 Stage2 官方特征目标期，并为每根 DIMM 的目标期特征点计算风险分数。

目标期来自 `data/stage2_feature/type_A/*.feather` 和 `data/stage2_feature/type_B/*.feather`。配置文件给出的目标时间范围为：

```json
{
  "target_start": 1722441600,
  "target_end": 1727712000
}
```

代码只读取该时间范围内的特征行：

```python
target_df = df[
    (df["LogTime"] >= config.target_start)
    & (df["LogTime"] <= config.target_end)
].copy()
```

## 问题表现

`stage2_best_per_type_prediction_scores.csv` 中记录了最终 top-200 候选的分数和阈值。当前文件显示，目标期最高分约为 0.00535，type_B 验证阈值为 0.12。

```text
serial_number,prediction_timestamp,serial_number_type,score,threshold
sn_72161,1724889569,B,0.005348367150872946,0.12
sn_72905,1726276499,B,0.005057293456047773,0.12
sn_69898,1725179397,B,0.00425881939008832,0.12
sn_70412,1724802251,B,0.00425881939008832,0.12
```

严格使用验证阈值时，目标期没有样本满足 `score >= threshold`。最终 submission 由 top-k fallback 生成。

## 是否说明实现错误

该现象本身不能证明实现错误。验证集和目标期来自不同时间段，目标期没有标签，CE 强度和硬件类型分布可能不同。XGBoost 的 `predict_proba` 输出也没有做跨时间段概率校准。验证集阈值在目标期失效属于内存故障预测中可能出现的分布漂移现象。

该现象同时提示实现必须做四项检查：

1. 目标期时间范围是否读取正确。
2. 预测特征列是否与训练特征列一致。
3. per-type 模型和阈值是否按 DIMM 类型匹配。
4. submission 为空时的 top-k 兜底是否保留每根 DIMM 的最高风险点。

本项目已完成上述检查。目标期使用配置中的 `target_start` 和 `target_end`。预测阶段调用 `align_prediction_features()` 对齐列。per-type bundle 按 `serial_number_type` 选择模型和阈值。top-k fallback 从每根 DIMM 的目标期最高风险点中取 200 行。

## 处理方法

submission 生成分为两步。第一步按验证阈值筛选候选。第二步在阈值未命中时取 top-k。

```python
selected = target_df[target_df["score"] >= float(thresholds.get(sn_type, 0.5))]
if selected.empty:
    continue
```

阈值筛选结束后，若 submission 为空，代码使用所有 DIMM 的最高风险点生成候选。

```python
if submission.empty and config.submission_top_k > 0:
    all_score_df = pd.DataFrame(all_score_rows).sort_values("score", ascending=False)
    score_df = all_score_df.head(config.submission_top_k)
    submission = score_df[
        ["serial_number", "prediction_timestamp", "serial_number_type"]
    ].copy()
```

每根 DIMM 的最高风险点在阈值筛选前已经记录。

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
```

## 结果

最终生成的 `stage2_best_per_type_submission.csv` 含 200 行。列为 `serial_number`、`prediction_timestamp`、`serial_number_type`。文件无空值，无重复 `serial_number`。

`task2_best_submission_scores.png` 展示了这 200 个候选的风险分数排序。图中最高目标期分数低于验证阈值，因此该图同时记录了 top-k fallback 的触发原因。

## 结论

目标期分数整体低于验证阈值属于分布漂移或概率未校准的可解释现象。现有实现完成了时间范围、列对齐、per-type 匹配和 top-k 兜底检查。该 submission 可作为格式正确的离线候选结果。当前 Codabench 评估入口关闭，项目没有 hidden-test F1。
