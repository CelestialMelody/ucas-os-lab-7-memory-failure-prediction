# 1. 故障预测样本存在时间泄漏风险

## 问题背景

内存故障预测的标签由故障 ticket 给出。若某根 DIMM 在 `failure_time` 发生故障，模型训练时只能使用故障前某个预测时刻以前的 CE 日志。真实运维还会保留 lead time，用于数据传输、模型推理和人工响应。故障前最后几分钟到几十分钟的日志不能进入正样本特征。

负样本也存在同类风险。同一根故障 DIMM 在故障前 7 天内的日志已经处于预测窗口内。若将这些片段标成负类，模型会学习到接近故障的正常样本。

## 定位过程

任务一和任务二的数据都包含故障时间。任务一使用 Stage1 原始 CE 日志，任务二使用 Stage2 官方预提取特征。两个流程都明确三个边界：预测时刻、lead time 和 prediction window。

## 处理方法

任务一在 `m2mfp-reproduction/src/raw_smoke.py` 中处理。若 DIMM 出现在 ticket 中，正样本预测时刻设为 `failure_time - lead_minutes`。所有窗口特征都以该时间为右边界，只使用 `LogTime <= end_time` 的日志。故障 DIMM 的负样本放在预测窗口之前。无故障 DIMM 使用最后日志时间点作为负样本。

```python
lead = config.lead_minutes * 60
pred = config.prediction_period_days * ONE_DAY

positive_end = int(failure_time) - lead
sample_specs.append((positive_end, 1))

negative_end = max(
    int(df["LogTime"].min()),
    int(failure_time) - lead - pred - ONE_DAY,
)
if negative_end < positive_end:
    sample_specs.append((negative_end, 0))
```

任务二在 `memory-failure-prediction-agent/src/stage2_feature_pipeline.py` 中处理。正样本从 `[failure_time - lead - prediction_period, failure_time - lead]` 内选择最后一个官方特征点。故障 DIMM 的负样本使用更早安全时间点。无故障 DIMM 使用最后一个特征点。

```python
pos_start = alarm_time - lead - pred
pos_end = alarm_time - lead
pos_df = df[(df["LogTime"] >= pos_start) & (df["LogTime"] <= pos_end)]
if not pos_df.empty:
    rows.append(add_meta(pos_df.iloc[-1], sn_name, sn_type, 1))

neg_df = df[df["LogTime"] < pos_start - ONE_DAY]
if not neg_df.empty:
    rows.append(add_meta(neg_df.iloc[-1], sn_name, sn_type, 0))
```

## 结果

任务一和任务二的样本构造都显式遵守 lead time 和 prediction window。任务一 full run 和任务二 agent 搜索的指标都基于该标签定义。

## 结论

正样本只使用预测时刻之前的数据。故障 DIMM 的负样本位于预测窗口之前。该设计控制了时间泄漏风险。
