from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from common.path_utils import config_path_prefix, resolve_path
    from time_patch import TIME_PATCH_WINDOWS, normalize_ce_schema, time_patch_features
    from time_point import time_point_features
except ImportError:  # pragma: no cover - 兼容以包形式执行时的相对导入。
    from ...common.path_utils import config_path_prefix, resolve_path
    from .time_patch import TIME_PATCH_WINDOWS, normalize_ce_schema, time_patch_features
    from .time_point import time_point_features


ONE_DAY = 86400


@dataclass(frozen=True)
class Config:
    """task1 的最小配置。

    这个类放在 task1 目录下，使 task1 可以作为独立的 M2-MFP 复现/检验项目运行。
    """

    data_path: Path
    ticket_path: Path
    feature_path: Path
    model_path: Path
    metrics_path: Path
    importance_path: Path
    submission_path: Path
    prediction_period_days: int = 7
    lead_minutes: int = 15
    window_seconds: tuple[int, ...] = (900, 3600, 21600, 86400)
    feature_groups: tuple[str, ...] = ("baseline", "time_patch", "time_point")
    data_paths: tuple[Path, ...] = ()
    random_state: int = 42

    @classmethod
    def from_json(cls, path: Path) -> "Config":
        """读取 JSON 配置，并把相对路径解析到 task1 项目根目录。

        配置文件位于 `task1_m2mfp_reproduction/configs/`，因此 `path.parent.parent`
        就是 task1 根目录。数据路径使用 `../../data/...`，会继续解析到仓库根目录
        下的数据集。
        """

        raw = json.loads(path.read_text(encoding="utf-8"))
        base = path.parent.parent
        data_prefix = config_path_prefix(raw, base)
        for key in [
            "feature_path",
            "model_path",
            "metrics_path",
            "importance_path",
            "submission_path",
        ]:
            raw[key] = resolve_path(raw[key], base)
        raw["ticket_path"] = resolve_path(raw["ticket_path"], base, data_prefix)
        if "data_paths" in raw:
            raw["data_paths"] = tuple(resolve_path(item, base, data_prefix) for item in raw["data_paths"])
            raw["data_path"] = raw["data_paths"][0]
        else:
            raw["data_path"] = resolve_path(raw["data_path"], base, data_prefix)
            raw["data_paths"] = (raw["data_path"],)
        raw.pop("path_prefix", None)
        raw["window_seconds"] = tuple(int(v) for v in raw.get("window_seconds", [900, 3600, 21600, 86400]))
        raw["feature_groups"] = tuple(raw.get("feature_groups", ["baseline", "time_patch", "time_point"]))
        return cls(**raw)


def load_ticket(config: Config) -> pd.DataFrame:
    """读取故障 ticket，并统一成 `serial_number/failure_time` 两列。

    Stage1 原始数据里的 ticket 是 M2-MFP 复现任务的标签来源：某个 DIMM
    在 `failure_time` 发生 UE/故障，则故障前的观察窗口可以构造为正样本。
    """

    ticket = pd.read_csv(config.ticket_path)
    rename = {
        "sn_name": "serial_number",
        "alarm_time": "failure_time",
        "serial_number_type": "sn_type",
    }
    ticket = ticket.rename(columns={old: new for old, new in rename.items() if old in ticket.columns})
    ticket["serial_number"] = ticket["serial_number"].astype(str)
    ticket["failure_time"] = pd.to_numeric(ticket["failure_time"], errors="coerce").astype("Int64")
    return ticket.dropna(subset=["serial_number", "failure_time"]).copy()


def load_sn_file(path: Path) -> pd.DataFrame:
    """读取单条 DIMM 的 CE 日志 feather 文件，并做基础字段规范化。"""

    return normalize_ce_schema(pd.read_feather(path))


def select_data_files(config: Config, max_files: int) -> list[Path]:
    """选择本次 smoke run 使用的原始日志文件。

    为了让小规模实验仍然包含足够正样本，这里优先保留 ticket 中出现过的故障
    DIMM 文件，然后再补充非故障 DIMM。这样 `--max-files` 是负样本侧的软限制：
    如果正样本数量本身很多，实际文件数可能超过 `max_files`。
    """

    all_files = []
    for data_path in config.data_paths:
        all_files.extend(sorted(data_path.glob("*.feather")))
    ticket_serials = set(load_ticket(config)["serial_number"].astype(str))
    positive_files = [path for path in all_files if path.stem in ticket_serials]
    negative_files = [path for path in all_files if path.stem not in ticket_serials]
    if max_files <= 0:
        return positive_files + negative_files
    remaining = max(0, max_files - len(positive_files))
    return positive_files + negative_files[:remaining]


def ce_storm_count(times: pd.Series, interval_seconds: int = 60, threshold: int = 10) -> int:
    """统计 CE storm 次数。

    这是 SmartMem/M2-MFP 类方法常用的时间点特征：短时间内连续出现大量 CE，
    往往意味着内存单元或局部结构已经进入不稳定状态。
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


def parity_to_matrix(parity: object) -> np.ndarray:
    """把 RetryRdErrLogParity 转为 8x4 的 DQ-Beat 二值矩阵。

    M2-MFP 的 BSFE 模块把 32 位 retry parity 看成空间矩阵：8 行近似表示
    burst/beat 方向，4 列近似表示 DQ 方向。这里保留这个解释方式，便于后续
    和 `bsfe.py` 中的高阶 bit 特征对应。
    """

    try:
        value = max(int(parity), 0)
    except (TypeError, ValueError):
        value = 0
    bits = bin(value)[2:].zfill(32)[-32:]
    return np.array([int(bit) for bit in bits], dtype=np.int8).reshape(8, 4)


def parity_stats(window_df: pd.DataFrame, prefix: str) -> dict[str, float]:
    """提取轻量 bit/DQ/beat 统计特征。

    完整的 M2-MFP 高阶 BSFE 特征由 `time_patch_features` 调用 `bsfe.py`
    生成。这里保留一组更直观的低阶统计，便于在报告中解释 RetryRdErrLogParity
    如何参与预测。
    """

    features: dict[str, float] = {}
    if window_df.empty:
        features[f"{prefix}_parity_bit_sum"] = 0.0
        features[f"{prefix}_parity_max_row_sum"] = 0.0
        features[f"{prefix}_parity_max_col_sum"] = 0.0
        return features

    matrices = np.stack([parity_to_matrix(value) for value in window_df["RetryRdErrLogParity"]])
    bit_sum_per_log = matrices.reshape(len(matrices), -1).sum(axis=1)
    features[f"{prefix}_parity_bit_sum"] = float(bit_sum_per_log.sum())
    features[f"{prefix}_parity_bit_mean"] = float(bit_sum_per_log.mean())
    features[f"{prefix}_parity_max_row_sum"] = float(matrices.sum(axis=2).max())
    features[f"{prefix}_parity_max_col_sum"] = float(matrices.sum(axis=1).max())
    return features


def aggregate_window_features(df: pd.DataFrame, end_time: int, window: int) -> dict[str, float]:
    """构造 v1 基础窗口特征。

    这些特征是课程项目 v1 已验证有效的基础时间/空间统计。task1 中保留它们的原因是：复现 M2-MFP 时需要一个
    可解释的 baseline，与后续 `m2patch_*` 论文特征比较。
    """

    window_df = df[(df["LogTime"] <= end_time) & (df["LogTime"] > end_time - window)]
    prefix = f"win_{window}"
    features: dict[str, float] = {
        f"{prefix}_ce_count": float(len(window_df)),
        f"{prefix}_read_ce_count": float((window_df["error_type_full_name"] == "CE.READ").sum()) if not window_df.empty else 0.0,
        f"{prefix}_scrub_ce_count": float((window_df["error_type_full_name"] == "CE.SCRUB").sum()) if not window_df.empty else 0.0,
        f"{prefix}_unique_device": float(window_df["deviceID"].nunique()) if not window_df.empty else 0.0,
        f"{prefix}_unique_bank": float(window_df["BankId"].nunique()) if not window_df.empty else 0.0,
        f"{prefix}_unique_row": float(window_df["RowId"].nunique()) if not window_df.empty else 0.0,
        f"{prefix}_unique_column": float(window_df["ColumnId"].nunique()) if not window_df.empty else 0.0,
        f"{prefix}_ce_storm_count": float(ce_storm_count(window_df["LogTime"])) if not window_df.empty else 0.0,
    }
    features.update(parity_stats(window_df, prefix))
    return features


def make_sample(
    df: pd.DataFrame,
    serial_number: str,
    sn_type: str,
    end_time: int,
    label: int,
    config: Config,
) -> dict[str, object]:
    """把单个 DIMM 在某个观察时间点前的日志聚合成一个监督学习样本。"""

    row: dict[str, object] = {
        "serial_number": serial_number,
        "serial_number_type": sn_type,
        "prediction_timestamp": int(end_time),
        "label": int(label),
    }
    if "baseline" in config.feature_groups:
        for window in config.window_seconds:
            row.update(aggregate_window_features(df, end_time, int(window)))
    # time_patch 内部包含 BSFE 聚合列；bsfe 单独消融会通过 filter_feature_groups
    # 只保留 bit-level 子列。
    if "time_patch" in config.feature_groups or "bsfe" in config.feature_groups:
        row.update(time_patch_features(df, end_time, TIME_PATCH_WINDOWS))
    if "time_point" in config.feature_groups:
        row.update(time_point_features(df, end_time))
    return row


def feature_columns(features: pd.DataFrame) -> list[str]:
    """返回模型训练列，排除标签和元信息列。"""

    excluded = {"serial_number", "serial_number_type", "prediction_timestamp", "label"}
    return [col for col in features.columns if col not in excluded]


def filter_feature_groups(features: pd.DataFrame, groups: tuple[str, ...]) -> pd.DataFrame:
    """按消融配置保留特征组。

    baseline 是 `win_*` 基础 CE 统计；time_patch 是 `m2patch_*` 多尺度窗口；
    bsfe 是 time_patch 中的 bit-level 子列；time_point 是 `m2point_*` 规则。
    """

    meta = ["serial_number", "serial_number_type", "prediction_timestamp", "label"]
    keep = set(meta)
    for col in features.columns:
        if "baseline" in groups and col.startswith("win_"):
            keep.add(col)
        if "time_patch" in groups and col.startswith("m2patch_"):
            keep.add(col)
        if "bsfe" in groups and col.startswith("m2patch_") and any(
            token in col for token in ["bsfe", "dq_count", "burst_count", "error_bit_count", "valid_err_log"]
        ):
            keep.add(col)
        if "time_point" in groups and col.startswith("m2point_"):
            keep.add(col)
    ordered = meta + [col for col in features.columns if col in keep and col not in meta]
    return features[ordered].copy()


def build_models(random_state: int) -> dict[str, object]:
    """构造 smoke run 使用的小模型集合。

    这里刻意不用过大的模型，保证复现检查可以在普通 CPU 上较快完成。完整
    调参和 GPU 加速应放到 task2 的 agent/优化流程中。
    """

    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
        ),
        "random_forest_small": RandomForestClassifier(
            n_estimators=120,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
    }


def build_raw_smoke_features(config: Config, max_files: int) -> pd.DataFrame:
    """从 Stage1 原始 feather 日志构造 M2-MFP smoke 特征表。

    正样本：若 DIMM 在 ticket 中有故障时间，则取故障前 `lead_minutes`
    处作为预测时刻，标签为 1。

    负样本：对故障 DIMM，取正样本窗口之前更早的安全时间点；对非故障 DIMM，
    取该 DIMM 最后一条日志作为负样本。这样能在小规模实验里形成基本的
    正负对照，但不能替代完整比赛数据上的严格评估。
    """

    files = select_data_files(config, max_files=max_files)
    ticket = load_ticket(config)
    ticket_map = dict(zip(ticket["serial_number"], ticket["failure_time"]))
    rows: list[dict[str, object]] = []
    lead = config.lead_minutes * 60
    pred = config.prediction_period_days * ONE_DAY

    start = time.perf_counter()
    for idx, path in enumerate(files, start=1):
        if idx % 100 == 0:
            elapsed = time.perf_counter() - start
            print(f"[raw-smoke] {idx}/{len(files)} elapsed={elapsed:.1f}s", flush=True)
        serial_number = path.stem
        df = load_sn_file(path)
        if df.empty:
            continue

        failure_time = ticket_map.get(serial_number)
        sn_type = "A" if "type_A" in str(path.parent) else "B" if "type_B" in str(path.parent) else "A"
        sample_specs: list[tuple[int, int]] = []

        if failure_time is not None and not pd.isna(failure_time):
            positive_end = int(failure_time) - lead
            if positive_end > int(df["LogTime"].min()):
                sample_specs.append((positive_end, 1))

            negative_end = max(int(df["LogTime"].min()), int(failure_time) - lead - pred - ONE_DAY)
            if negative_end < positive_end:
                sample_specs.append((negative_end, 0))
        else:
            sample_specs.append((int(df["LogTime"].max()), 0))

        for end_time, label in sample_specs:
            row = make_sample(df, serial_number, sn_type, end_time, label, config)
            rows.append(row)

    return pd.DataFrame(rows).fillna(0)


def evaluate(features: pd.DataFrame, config: Config) -> pd.DataFrame:
    """用交叉验证评估 smoke 特征表。

    如果正负样本数量太少，自动退化为训练集内评估，避免小数据调试阶段直接
    报错。正式报告中应优先引用有独立验证切分或完整数据的结果。
    """

    cols = feature_columns(features)
    y = features["label"].astype(int)
    n_splits = min(3, int(y.value_counts().min())) if y.nunique() == 2 else 0
    rows = []
    for name, model in build_models(config.random_state).items():
        x = features[cols]
        if n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.random_state)
            pred = cross_val_predict(model, x, y, cv=cv)
        else:
            model.fit(x, y)
            pred = model.predict(x)
        rows.append(
            {
                "model": name,
                "precision": precision_score(y, pred, zero_division=0),
                "recall": recall_score(y, pred, zero_division=0),
                "f1": f1_score(y, pred, zero_division=0),
                "samples": len(features),
                "positive_samples": int(y.sum()),
                "feature_count": len(cols),
            }
        )
    return pd.DataFrame(rows).sort_values(["f1", "recall", "precision"], ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/stage1_type_a_m2mfp_smoke.json",
    )
    parser.add_argument("--max-files", type=int, default=300)
    args = parser.parse_args()

    config = Config.from_json(Path(args.config))
    features = build_raw_smoke_features(config, args.max_files)
    features = filter_feature_groups(features, config.feature_groups)
    config.feature_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(config.feature_path, index=False)

    metrics = evaluate(features, config)
    config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(config.metrics_path, index=False)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
