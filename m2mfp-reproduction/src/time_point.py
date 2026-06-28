from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_numeric(series: pd.Series) -> pd.Series:
    """把输入转成数值列，无法解析的值按 0 处理。"""

    return pd.to_numeric(series, errors="coerce").fillna(0)


def _max_run_count(times: np.ndarray, interval_seconds: int) -> int:
    """统计给定间隔内的最长连续 CE 链。

    time-point 关注故障前 CE 是否在短时间内连续出现。最长连续链比窗口总数
    更能反映局部爆发。
    """

    if len(times) <= 1:
        return int(len(times))
    values = np.sort(times.astype(int))
    run = 1
    best = 1
    for prev, cur in zip(values[:-1], values[1:]):
        if int(cur) - int(prev) <= interval_seconds:
            run += 1
        else:
            run = 1
        best = max(best, run)
    return int(best)


def _recent_gap_stats(times: np.ndarray, end_time: int) -> dict[str, float]:
    """计算最近 CE 距离和事件间隔。

    这些统计只使用 `end_time` 之前的日志。`last_gap_seconds` 越小，说明预测
    时刻附近刚出现过 CE；`burstiness` 用于描述最近间隔相对历史间隔的压缩程度。
    """

    if len(times) == 0:
        return {
            "last_gap_seconds": 0.0,
            "min_interarrival_seconds": 0.0,
            "median_interarrival_seconds": 0.0,
            "burstiness": 0.0,
        }
    values = np.sort(times.astype(int))
    gaps = np.diff(values)
    if len(gaps) == 0:
        min_gap = 0.0
        median_gap = 0.0
    else:
        min_gap = float(gaps.min())
        median_gap = float(np.median(gaps))
    # burstiness 越大，说明最近的 CE 间隔相对历史间隔越短。
    burstiness = float((median_gap + 1.0) / (min_gap + 1.0)) if min_gap or median_gap else 0.0
    return {
        "last_gap_seconds": float(max(end_time - int(values[-1]), 0)),
        "min_interarrival_seconds": min_gap,
        "median_interarrival_seconds": median_gap,
        "burstiness": burstiness,
    }


def _dominant_ratio(series: pd.Series) -> float:
    """计算最高频地址值占比，用于判断短窗口内是否出现局部集中。"""

    if series.empty:
        return 0.0
    counts = series.astype(str).value_counts(dropna=False)
    return float(counts.iloc[0] / max(len(series), 1))


def time_point_features(
    df: pd.DataFrame,
    end_time: int,
    short_window: int = 900,
    long_window: int = 21600,
) -> dict[str, float]:
    """提取 M2-MFP 风格 time-point 规则特征。

    本地适配关注三类故障前时间点信号：
    1. 最近 CE 到预测时间点的距离。
    2. 短窗口相对长窗口的 CE 激增比例。
    3. 短窗口内 CE 是否集中在同一 device/bank/row/column/cell。

    这些规则特征维度少，含义直接。它们在 full run 中表现强，说明故障前的
    时间邻近性和空间集中度对当前数据有稳定信号。
    """

    if df.empty:
        return {}
    work = df.copy()
    work["LogTime"] = _safe_numeric(work["LogTime"]).astype(int)
    for col in ["deviceID", "BankId", "RowId", "ColumnId"]:
        if col not in work.columns:
            work[col] = -1
    work["CellId"] = work["RowId"].astype(str) + "_" + work["ColumnId"].astype(str)

    recent = work[(work["LogTime"] <= end_time) & (work["LogTime"] > end_time - short_window)]
    long = work[(work["LogTime"] <= end_time) & (work["LogTime"] > end_time - long_window)]
    all_times = work[work["LogTime"] <= end_time]["LogTime"].to_numpy()
    recent_times = recent["LogTime"].to_numpy()
    long_count = max(len(long), 1)

    features: dict[str, float] = {f"m2point_{key}": value for key, value in _recent_gap_stats(all_times, end_time).items()}
    features.update(
        {
            "m2point_short_ce_count": float(len(recent)),
            "m2point_long_ce_count": float(len(long)),
            "m2point_short_long_ratio": float(len(recent) / long_count),
            "m2point_max_run_60s": float(_max_run_count(recent_times, 60)),
            "m2point_max_run_300s": float(_max_run_count(recent_times, 300)),
            "m2point_read_ratio_short": float((recent.get("error_type_full_name", "") == "CE.READ").sum() / max(len(recent), 1)),
            "m2point_scrub_ratio_short": float((recent.get("error_type_full_name", "") == "CE.SCRUB").sum() / max(len(recent), 1)),
        }
    )
    for col in ["deviceID", "BankId", "RowId", "ColumnId", "CellId"]:
        features[f"m2point_{col}_dominant_ratio"] = _dominant_ratio(recent[col]) if col in recent else 0.0
        features[f"m2point_{col}_unique_short"] = float(recent[col].nunique()) if col in recent else 0.0
    features["m2point_is_cell_localized"] = float(features["m2point_CellId_dominant_ratio"] >= 0.8 and len(recent) >= 2)
    features["m2point_is_row_or_column_spread"] = float(
        (features["m2point_RowId_unique_short"] > 1 or features["m2point_ColumnId_unique_short"] > 1) and len(recent) >= 3
    )
    return features
