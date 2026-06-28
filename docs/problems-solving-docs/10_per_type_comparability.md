# 10. per-type 策略的验证集可比性

## 问题背景

Stage2 数据按 type_A/type_B 分目录。两类 DIMM 可能有不同错误模式和特征分布。per-type 建模保留类型差异。若每个类型单独切分验证集，验证时间范围和样本数会变化，无法和全局单模型策略公平比较。

## 处理方法

agent 中先做一次全局时间切分，再在 train/valid 内按 `serial_number_type` 分组。per-type 模型只改变训练方式，不改变验证时间段。两个类型验证子集的分数合并后计算总体 F1。

关键实现如下。`global_train` 和 `global_valid` 先由完整训练表切出，再在各自内部按类型过滤。

```python
if strategy.per_type:
    global_train, global_valid = time_split(features, config.validation_quantile)
    frames = [
        (str(sn_type), part.copy())
        for sn_type, part in features.groupby("serial_number_type")
    ]

for sn_type, frame in frames:
    if strategy.per_type:
        train = global_train[global_train["serial_number_type"] == sn_type].copy()
        valid = global_valid[global_valid["serial_number_type"] == sn_type].copy()
```

各类型验证分数随后拼接，再统一计算 precision、recall 和 F1。

## 结果

`xgb_all_none_per_type` 和 `xgb_all_none_threshold` 使用同一验证时间段，指标具有可比性。

| 策略 | F1 |
| --- | ---: |
| xgb_all_none_threshold | 0.5410 |
| xgb_all_none_per_type | 0.5538 |

## 结论

per-type 的提升来自分类型训练。验证时间段与全局单模型策略保持一致。
