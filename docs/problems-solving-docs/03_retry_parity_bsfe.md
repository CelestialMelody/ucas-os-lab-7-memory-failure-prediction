# 3. `RetryRdErrLogParity` 的 bit 空间特征转换

## 问题背景

`RetryRdErrLogParity` 是 CE 日志中的 32 位整数。直接把它作为数值输入时，数值大小没有明确硬件含义。M2-MFP 论文指出，bit 级错误模式可以表示为 DQ-Beat 矩阵，并从行列方向提取空间结构。

## 定位过程

任务一检验 M2-MFP 风格 bit 特征。本项目没有直接调用论文私有训练流水线，因此在本地代码中实现 parity 到 bit 特征的转换。关键目标是让模型看到 bit 错误的分布形态，避免只输入一个缺少硬件含义的整数。

## 处理方法

`m2mfp-reproduction/src/bsfe.py` 中实现了 `BSFEExtractor`。代码将 parity 转为 32 位二进制字符串，并 reshape 为 8x4 DQ-Beat 矩阵。随后分别沿行和列提取 bit 数、最小间隔、最大跨度、最长连续 1 和连续聚集程度。

```python
def transform_one(self, parity: object) -> list[int]:
    try:
        value = int(parity)
    except (TypeError, ValueError):
        value = 0
    bits = bin(max(value, 0))[2:].zfill(self.row_count * self.column_count)[-32:]
    rows = [bits[idx : idx + self.column_count] for idx in range(0, len(bits), self.column_count)]
    columns = [bits[idx:: self.column_count] for idx in range(self.column_count)]
    return self._row_features(rows) + self._column_features(columns)
```

窗口级聚合位于 `time_patch.aggregate_bsfe()`。max 描述窗口中最强 bit 模式，sum 描述累计强度，avg 描述有效日志下的平均强度。

```python
bsfe_df = BSFEExtractor().transform_series(window_df["RetryRdErrLogParity"], prefix="bsfe")
for name in names:
    result[f"max_{name}"] = float(bsfe_df[name].max())
    result[f"sum_{name}"] = float(bsfe_df[name].sum())
    result[f"avg_{name}"] = float(bsfe_df[name].sum() / divisor)
```

## 结果

任务一比较低阶 parity 统计、BSFE bit 特征和 time-patch 聚合。实验结果显示 BSFE/time-patch 相对 baseline 有信号，但直接拼接全部高维特征没有超过 time-point。

## 结论

BSFE 将 32 位 parity 转成 DQ/beat 空间模式。该模块对应 M2-MFP 的 bit-level 特征思想。
