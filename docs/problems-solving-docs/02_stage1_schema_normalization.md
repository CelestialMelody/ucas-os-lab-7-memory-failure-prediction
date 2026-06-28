# 2. Stage1 原始 CE 日志字段类型不稳定

## 问题背景

Stage1 feather 是原始 CE 日志。不同文件中的字段可能存在缺失值、字符串值或数值类型差异。`LogTime`、`deviceID`、`RetryRdErrLogParity`、`RetryRdErrLog` 等字段会被多个特征模块共同使用。若每个模块各自处理缺失值和类型转换，特征含义会不一致。

## 定位过程

任务一同时运行 baseline、BSFE、time-patch 和 time-point。baseline 依赖时间窗口和硬件地址统计，BSFE 依赖 retry parity，time-patch 依赖 CE 类型、空间地址和 bit 特征，time-point 依赖预测时间点附近的 CE 分布。字段规范化若分散在多个函数中，缺失地址可能被当作真实地址。

## 处理方法

`m2mfp-reproduction/src/time_patch.py` 中的 `normalize_ce_schema()` 集中处理字段类型、缺失值和派生列。

```python
out["LogTime"] = pd.to_numeric(out["LogTime"], errors="coerce").fillna(0).astype(int)
out["deviceID"] = pd.to_numeric(out["deviceID"], errors="coerce").fillna(-1).astype(int)
out["RetryRdErrLogParity"] = (
    pd.to_numeric(out["RetryRdErrLogParity"], errors="coerce")
    .fillna(0)
    .astype(np.int64)
)
out["RetryRdErrLog"] = (
    pd.to_numeric(out.get("RetryRdErrLog", 0), errors="coerce")
    .fillna(0)
    .astype(np.int64)
)
out["error_type_is_READ_CE"] = (out["error_type_full_name"] == "CE.READ").astype(int)
out["error_type_is_SCRUB_CE"] = (out["error_type_full_name"] == "CE.SCRUB").astype(int)
out["CellId"] = out["RowId"].astype(str) + "_" + out["ColumnId"].astype(str)
```

空间唯一值统计排除 -1，避免把缺失地址计为空间扩散。

```python
def unique_num_filtered(values: np.ndarray, impute_value: int = -1) -> int:
    unique = np.unique(values)
    return int(len(unique) - int(impute_value in unique))
```

## 结果

BSFE、time-patch 和 time-point 使用同一输入规范。Stage1 smoke 和 full run 复用同一套特征函数，字段类型差异对结果的影响降低。

## 结论

原始日志进入特征工程前先做统一 schema 规范化。缺失硬件地址使用哨兵值，并在空间统计时排除。
