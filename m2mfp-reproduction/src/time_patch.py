from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from bsfe import BSFEExtractor, bsfe_feature_names
except ImportError:  # pragma: no cover - 兼容以包形式执行时的相对导入。
    from .bsfe import BSFEExtractor, bsfe_feature_names


# M2-MFP time-patch 的核心是多尺度观察窗口。这里先使用 15m/1h/6h
# 三个窗口，避免 Stage1 smoke run 过慢；24h 基础窗口由 raw_smoke.py 单独保留。
TIME_PATCH_WINDOWS = [900, 3600, 21600]


def normalize_ce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """统一 SmartMem 原始 CE 日志字段类型。

    Stage1 feather 来自比赛数据，不同字段可能包含缺失值或字符串类型。
    在进入 time-patch/BSFE 聚合前先做规范化，可以让后续特征函数只关心
    预测逻辑，而不用在每个函数里重复处理脏数据。
    """

    out = df.copy()
    out["LogTime"] = pd.to_numeric(out["LogTime"], errors="coerce").fillna(0).astype(int)
    out["deviceID"] = pd.to_numeric(out["deviceID"], errors="coerce").fillna(-1).astype(int)
    out["RetryRdErrLogParity"] = pd.to_numeric(out["RetryRdErrLogParity"], errors="coerce").fillna(0).astype(np.int64)
    out["RetryRdErrLog"] = pd.to_numeric(out.get("RetryRdErrLog", 0), errors="coerce").fillna(0).astype(np.int64)
    out["error_type_is_READ_CE"] = (out["error_type_full_name"] == "CE.READ").astype(int)
    out["error_type_is_SCRUB_CE"] = (out["error_type_full_name"] == "CE.SCRUB").astype(int)
    out["retry_log_is_valid"] = ((out["RetryRdErrLog"] & 1) == 1).astype(int)
    out["CellId"] = out["RowId"].astype(str) + "_" + out["ColumnId"].astype(str)
    return out.sort_values("LogTime").reset_index(drop=True)


def ce_storm_count(times: pd.Series, interval_seconds: int = 60, threshold: int = 10) -> int:
    """统计短时间密集 CE 事件。

    多个获奖方案和 M2-MFP time-point 思想都强调：CE 的“突然爆发”比单纯
    累计数量更有故障指示意义。这里用一个轻量规则近似该现象。
    """

    values = sorted(int(value) for value in times)
    storms = 0
    consecutive = 0
    for prev, cur in zip(values, values[1:]):
        if cur - prev <= interval_seconds:
            consecutive += 1
        else:
            consecutive = 0
        if consecutive > threshold:
            storms += 1
            consecutive = 0
    return storms


def unique_num_filtered(values: np.ndarray, impute_value: int = -1) -> int:
    """统计唯一值数量，并排除缺失值填充值。"""

    unique = np.unique(values)
    return int(len(unique) - int(impute_value in unique))


def spatio_features(window_df: pd.DataFrame) -> dict[str, float]:
    """提取窗口内空间故障模式。

    这些规则对应内存故障定位中的常见层级：device、bank、row、column、cell。
    它们用于判断 CE 是否集中在某个局部结构，还是跨多个结构扩散。
    """

    features = {
        "fault_mode_others": 0.0,
        "fault_mode_device": 0.0,
        "fault_mode_bank": 0.0,
        "fault_mode_row": 0.0,
        "fault_mode_column": 0.0,
        "fault_mode_cell": 0.0,
        "fault_row_num": 0.0,
        "fault_column_num": 0.0,
    }
    if window_df.empty:
        return features
    if unique_num_filtered(window_df["deviceID"].values) > 1:
        features["fault_mode_others"] = 1.0
    elif unique_num_filtered(window_df["BankId"].values) > 1:
        features["fault_mode_device"] = 1.0
    elif unique_num_filtered(window_df["ColumnId"].values) > 1 and unique_num_filtered(window_df["RowId"].values) > 1:
        features["fault_mode_bank"] = 1.0
    elif unique_num_filtered(window_df["ColumnId"].values) > 1:
        features["fault_mode_row"] = 1.0
    elif unique_num_filtered(window_df["RowId"].values) > 1:
        features["fault_mode_column"] = 1.0
    elif unique_num_filtered(window_df["CellId"].values) == 1:
        features["fault_mode_cell"] = 1.0

    features["fault_row_num"] = float(window_df.groupby(["deviceID", "BankId", "RowId"])["ColumnId"].nunique().gt(1).sum())
    features["fault_column_num"] = float(window_df.groupby(["deviceID", "BankId", "ColumnId"])["RowId"].nunique().gt(1).sum())
    return features


def aggregate_bsfe(window_df: pd.DataFrame) -> dict[str, float]:
    """在一个 time-patch 窗口内聚合 BSFE 特征。

    `BSFEExtractor` 先对每条日志生成 bit 级空间特征；本函数再对窗口内多条
    日志做 max/sum/avg 聚合，从而得到一个固定长度的监督学习特征向量。
    """

    names = bsfe_feature_names("bsfe")
    empty = {f"{stat}_{name}": 0.0 for name in names for stat in ["max", "sum", "avg"]}
    empty.update({"error_bit_count": 0.0, "all_valid_err_log_count": 0.0})
    for dq in [1, 2, 3, 4]:
        empty[f"dq_count={dq}"] = 0.0
    for burst in range(1, 9):
        empty[f"burst_count={burst}"] = 0.0
    if window_df.empty:
        return empty

    bsfe_df = BSFEExtractor().transform_series(window_df["RetryRdErrLogParity"], prefix="bsfe")
    result: dict[str, float] = {}
    result["error_bit_count"] = float(bsfe_df["bsfe_rowwise_sum_pooling_F_bit_count"].sum())
    valid_count = float(window_df["retry_log_is_valid"].sum())
    result["all_valid_err_log_count"] = valid_count
    divisor = valid_count if valid_count else max(len(window_df), 1)
    for name in names:
        result[f"max_{name}"] = float(bsfe_df[name].max())
        result[f"sum_{name}"] = float(bsfe_df[name].sum())
        result[f"avg_{name}"] = float(bsfe_df[name].sum() / divisor)
    dq_counts = bsfe_df["bsfe_rowwise_F_max_pooling_bit_count"].value_counts().to_dict()
    burst_counts = bsfe_df["bsfe_columnwise_F_max_pooling_bit_count"].value_counts().to_dict()
    for dq in [1, 2, 3, 4]:
        result[f"dq_count={dq}"] = float(dq_counts.get(dq, 0))
    for burst in range(1, 9):
        result[f"burst_count={burst}"] = float(burst_counts.get(burst, 0))
    return result


def time_patch_features(df: pd.DataFrame, end_time: int, windows: list[int] | None = None) -> dict[str, float]:
    """构造 M2-MFP 风格的多尺度 time-patch 特征。

    对每个窗口，输出三类信息：
    1. 时间强度：READ/SCRUB CE 数量、日志频率、CE storm。
    2. 空间模式：device/bank/row/column/cell 层级集中程度。
    3. bit 模式：BSFE 高阶 DQ-Beat 特征。
    """

    windows = windows or TIME_PATCH_WINDOWS
    df = normalize_ce_schema(df)
    features: dict[str, float] = {}
    for window in windows:
        wdf = df[(df["LogTime"] <= end_time) & (df["LogTime"] > end_time - window)]
        prefix = f"m2patch_{window}"
        temporal = {
            "read_ce_log_num": float(wdf["error_type_is_READ_CE"].sum()),
            "scrub_ce_log_num": float(wdf["error_type_is_SCRUB_CE"].sum()),
            "all_ce_log_num": float(len(wdf)),
            "log_happen_frequency": float(window / len(wdf)) if len(wdf) else 0.0,
            "ce_storm_count": float(ce_storm_count(wdf["LogTime"])),
        }
        merged = temporal | spatio_features(wdf) | aggregate_bsfe(wdf)
        features.update({f"{prefix}_{key}": value for key, value in merged.items()})
    return features
