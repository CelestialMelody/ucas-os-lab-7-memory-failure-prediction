# 16. 深度学习模型依赖额外样本设计

## 问题背景

raw CE 日志是不规则稀疏事件流。Transformer Encoder 依赖固定长度序列。若直接用滑窗扩增，同一故障附近的相邻窗口高度相关。样本数增加不等于故障模式增加。

## 处理方法

Transformer 方案拆分为独立实验设计：

- 先把 CE 事件聚合为固定窗口。
- 构造过去 14 天、每 6 小时一个窗口的序列。
- 使用 `BCEWithLogitsLoss(pos_weight=neg/pos)` 或 focal loss 处理类别不平衡。
- 保持时间切分验证。

序列化阶段先把 CE 事件聚合为固定步长的表，再把同一 DIMM 的连续窗口拼成张量。

```python
def build_sequence(events, end_time, step_seconds=21600, steps=56):
    start = end_time - step_seconds * steps
    bins = np.arange(start, end_time + step_seconds, step_seconds)
    counts, _ = np.histogram(events["LogTime"], bins=bins)
    read_counts, _ = np.histogram(events.loc[events["is_read"], "LogTime"], bins=bins)
    scrub_counts, _ = np.histogram(events.loc[events["is_scrub"], "LogTime"], bins=bins)
    return np.stack([counts, read_counts, scrub_counts], axis=1)
```

该设计还要解决两个验证问题：同一故障 DIMM 的相邻窗口不能同时进入训练和验证；lead time 内的日志不能进入正样本序列。

## 结果

主线保留树模型和梯度提升模型。Transformer Encoder 作为扩展方案记录在 `private_docs/09_future_work.md`。

## 结论

深度学习方案依赖额外序列化、样本设计和验证策略。本项目收口版本使用表格模型完成两条任务线。
