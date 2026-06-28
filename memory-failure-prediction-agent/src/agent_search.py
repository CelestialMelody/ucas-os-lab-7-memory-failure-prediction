from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

try:
    from stage2_feature_pipeline import Stage2Config, build_training_table, feature_columns, threshold_search, time_split
except ImportError:  # pragma: no cover
    from .stage2_feature_pipeline import Stage2Config, build_training_table, feature_columns, threshold_search, time_split

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None


@dataclass(frozen=True)
class AgentStrategy:
    """SmartHW agent 的单次实验策略。

    一个策略固定模型、特征组、采样方式和决策方式。搜索时只比较这些显式维度，
    避免把参数调优和策略贡献混在一起。
    """

    name: str
    model: str
    feature_set: str
    sampling: str
    threshold_mode: str
    top_k: int | None = None
    per_type: bool = False


DEFAULT_STRATEGIES = [
    # 第一组：不采样的模型公平对比，对齐 stage2_feature_pipeline.py 的 baseline 语义。
    AgentStrategy("rf_all_none_threshold", "random_forest", "all", "none", "threshold"),
    AgentStrategy("hgb_all_none_threshold", "hist_gradient_boosting", "all", "none", "threshold"),
    # 第二组：采样和 hard negative，用来验证类别不平衡处理是否真正有效。
    AgentStrategy("rf_all_balanced_threshold", "random_forest", "all", "balanced", "threshold"),
    AgentStrategy("hgb_all_balanced_threshold", "hist_gradient_boosting", "all", "balanced", "threshold"),
    AgentStrategy("rf_all_hard_negative_threshold", "random_forest", "all", "hard_negative", "threshold"),
    # 第三组：特征组合消融。
    AgentStrategy("rf_temporal_balanced_threshold", "random_forest", "temporal", "balanced", "threshold"),
    AgentStrategy("rf_spatial_bit_balanced_threshold", "random_forest", "spatial_bit", "balanced", "threshold"),
    AgentStrategy("hgb_temporal_none_threshold", "hist_gradient_boosting", "temporal", "none", "threshold"),
    AgentStrategy("hgb_spatial_bit_none_threshold", "hist_gradient_boosting", "spatial_bit", "none", "threshold"),
    # 第四组：提交侧常见策略。
    AgentStrategy("rf_all_balanced_top100", "random_forest", "all", "balanced", "top_k", top_k=100),
    AgentStrategy("rf_all_balanced_per_type", "random_forest", "all", "balanced", "threshold", per_type=True),
    AgentStrategy("hgb_all_none_per_type", "hist_gradient_boosting", "all", "none", "threshold", per_type=True),
]
if LGBMClassifier is not None:
    DEFAULT_STRATEGIES.extend(
        [
            AgentStrategy("lgb_all_none_threshold", "lightgbm", "all", "none", "threshold"),
            AgentStrategy("lgb_all_balanced_threshold", "lightgbm", "all", "balanced", "threshold"),
            AgentStrategy("lgb_temporal_none_threshold", "lightgbm", "temporal", "none", "threshold"),
            AgentStrategy("lgb_all_none_per_type", "lightgbm", "all", "none", "threshold", per_type=True),
        ]
    )
if XGBClassifier is not None:
    DEFAULT_STRATEGIES.extend(
        [
            AgentStrategy("xgb_all_none_threshold", "xgboost", "all", "none", "threshold"),
            AgentStrategy("xgb_all_balanced_threshold", "xgboost", "all", "balanced", "threshold"),
            AgentStrategy("xgb_temporal_none_threshold", "xgboost", "temporal", "none", "threshold"),
            AgentStrategy("xgb_all_none_top100", "xgboost", "all", "none", "top_k", top_k=100),
            AgentStrategy("xgb_all_none_per_type", "xgboost", "all", "none", "threshold", per_type=True),
        ]
    )


def select_feature_columns(features: pd.DataFrame, feature_set: str) -> list[str]:
    """按 agent 的特征组名称选择训练列。

    Stage2 官方特征列名已经编码窗口、空间和 bit 统计含义。本函数用列名 token
    做轻量分组，用于比较时间强度特征和空间/bit 特征。若某组没有命中列，退回
    全量列，避免官方数据列名变化导致实验中断。
    """

    cols = feature_columns(features)
    if feature_set == "all":
        return cols
    temporal_tokens = ["LogTime", "log_num", "count", "frequency", "storm"]
    spatial_tokens = ["fault_mode", "fault_row", "fault_column", "dq_", "burst", "bit", "interval"]
    if feature_set == "temporal":
        selected = [col for col in cols if any(token in col for token in temporal_tokens)]
    elif feature_set == "spatial_bit":
        selected = [col for col in cols if any(token in col for token in spatial_tokens)]
    else:
        raise ValueError(f"unknown feature_set: {feature_set}")
    return selected or cols


def build_model(name: str, random_state: int):
    """构造 agent 策略中的候选模型。

    这些参数是固定的轻量配置，用于比较策略维度。XGBoost/LightGBM 只有在环境
    安装对应包时才会进入策略库。
    """

    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=160,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(max_iter=160, max_leaf_nodes=31, learning_rate=0.05, random_state=random_state)
    if name == "lightgbm" and LGBMClassifier is not None:
        return LGBMClassifier(
            n_estimators=260,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )
    if name == "xgboost" and XGBClassifier is not None:
        return XGBClassifier(
            n_estimators=220,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=random_state,
        )
    raise ValueError(f"model is unavailable: {name}")


def json_ready(value):
    """把 numpy/pandas 标量和 NaN 转成标准 JSON 可写对象。"""

    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if pd.isna(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def apply_sampling(train: pd.DataFrame, strategy: AgentStrategy, random_state: int) -> pd.DataFrame:
    """应用采样策略。

    `balanced` 让负样本数量接近正样本 5 倍；`hard_negative` 优先保留 CE 强度
    较高的负样本。采样只作用于训练集，验证集保持原始比例。
    """

    if strategy.sampling == "none":
        return train
    pos = train[train["label"] == 1]
    neg = train[train["label"] == 0]
    if pos.empty or neg.empty:
        return train
    target_neg = min(len(neg), max(len(pos) * 5, len(pos)))
    if strategy.sampling == "balanced":
        neg_sample = neg.sample(n=target_neg, random_state=random_state)
    elif strategy.sampling == "hard_negative":
        cols = [col for col in neg.columns if "count" in col or "log_num" in col or "storm" in col]
        if cols:
            ranked = neg.assign(_hard_score=neg[cols].sum(axis=1)).sort_values("_hard_score", ascending=False)
            neg_sample = ranked.head(target_neg).drop(columns=["_hard_score"])
        else:
            neg_sample = neg.sample(n=target_neg, random_state=random_state)
    else:
        raise ValueError(f"unknown sampling: {strategy.sampling}")
    return pd.concat([pos, neg_sample], ignore_index=True).sample(frac=1.0, random_state=random_state)


def score_model(model, frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """返回正类风险分数。

    大多数模型支持 `predict_proba`；若模型只提供硬预测，则退化为 0/1 分数。
    阈值搜索和 top-k 都消费同一个一维风险分数。
    """

    if hasattr(model, "predict_proba"):
        return model.predict_proba(frame[cols])[:, 1]
    return np.asarray(model.predict(frame[cols]), dtype=float)


def evaluate_scores(y_true: pd.Series, scores: np.ndarray, strategy: AgentStrategy) -> dict[str, float]:
    """按策略定义把风险分数转为 precision/recall/F1。

    `threshold` 模式在验证集上搜索最佳阈值；`top_k` 模式模拟提交文件只允许
    固定告警数量时的选择方式。两者都保留在 agent 中，用于区分本地验证指标
    和 submission 生成策略。
    """

    if strategy.threshold_mode == "top_k" and strategy.top_k:
        pred = np.zeros(len(scores), dtype=int)
        top_idx = np.argsort(scores)[-min(strategy.top_k, len(scores)) :]
        pred[top_idx] = 1
        return {
            "threshold": float("nan"),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
        }
    thresholds = threshold_search(y_true.astype(int), scores)
    return thresholds.iloc[0][["threshold", "precision", "recall", "f1"]].to_dict()


def run_one_strategy(features: pd.DataFrame, config: Stage2Config, strategy: AgentStrategy) -> dict[str, object]:
    """执行一个 agent 策略并返回统一指标行。

    per-type 策略会在同一全局时间切分下分别训练 type_A/type_B 模型，再把两个
    验证子集的分数拼回一起算总体 F1。这样结果可与全局单模型策略比较，且验证
    时间范围保持一致。
    """

    frames = [("", features)]
    global_train = None
    global_valid = None
    if strategy.per_type:
        # 先做一次全局时间切分，再在 train/valid 内按类型分别训练和验证。
        # 这样 per-type 与单模型策略共享同一验证时间段。
        global_train, global_valid = time_split(features, config.validation_quantile)
        frames = [(str(sn_type), part.copy()) for sn_type, part in features.groupby("serial_number_type")]

    parts = []
    detail_rows = []
    for sn_type, frame in frames:
        if strategy.per_type and global_train is not None and global_valid is not None:
            train = global_train[global_train["serial_number_type"] == sn_type].copy()
            valid = global_valid[global_valid["serial_number_type"] == sn_type].copy()
        else:
            train, valid = time_split(frame, config.validation_quantile)
        if train["label"].nunique() < 2 or valid["label"].nunique() < 2:
            continue
        cols = select_feature_columns(frame, strategy.feature_set)
        sampled = apply_sampling(train, strategy, config.random_state)
        model = build_model(strategy.model, config.random_state)
        model.fit(sampled[cols], sampled["label"].astype(int))
        scores = score_model(model, valid, cols)
        metric = evaluate_scores(valid["label"].astype(int), scores, strategy)
        detail_rows.append(
            {
                "serial_number_type": sn_type or "all",
                "train_samples": len(sampled),
                "valid_samples": len(valid),
                "train_positive": int(sampled["label"].sum()),
                "valid_positive": int(valid["label"].sum()),
                "feature_count": len(cols),
                **metric,
            }
        )
        parts.append((valid["label"].astype(int).to_numpy(), scores))

    if not parts:
        return {**asdict(strategy), "status": "skipped", "precision": 0.0, "recall": 0.0, "f1": 0.0}
    y = np.concatenate([part[0] for part in parts])
    scores = np.concatenate([part[1] for part in parts])
    metric = evaluate_scores(pd.Series(y), scores, strategy)
    return {
        **asdict(strategy),
        "status": "ok",
        "precision": float(metric["precision"]),
        "recall": float(metric["recall"]),
        "f1": float(metric["f1"]),
        "threshold": float(metric["threshold"]) if pd.notna(metric["threshold"]) else np.nan,
        "valid_samples": int(sum(row["valid_samples"] for row in detail_rows)),
        "valid_positive": int(sum(row["valid_positive"] for row in detail_rows)),
        "details": detail_rows,
    }


def load_or_build_features(config: Stage2Config, rebuild: bool) -> pd.DataFrame:
    """读取 Stage2 训练缓存，必要时从官方 feather 重新构造。

    agent 搜索默认复用缓存，避免每次策略搜索都遍历上万级 feather 文件。只有
    显式传入 `--rebuild-features` 时才重建监督学习表。
    """

    path = config.output_dir / "stage2_train_features.csv"
    if path.exists() and not rebuild:
        return pd.read_csv(path)
    features = build_training_table(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(path, index=False)
    return features


def run_agent(config_path: Path, dry_run: bool, rebuild_features: bool, max_strategies: int | None) -> pd.DataFrame:
    """运行 agent 策略登记、搜索和最佳配置落盘。

    产物分三类：registry 记录计划实验，ablation_results 记录指标，runtime_log
    记录耗时和样本规模。报告据此说明试验范围、最佳策略和运行成本。
    """

    config = Stage2Config.from_json(config_path)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    strategies = DEFAULT_STRATEGIES[: max_strategies or None]
    registry = pd.DataFrame([asdict(strategy) | {"status": "planned"} for strategy in strategies])
    registry_path = config.output_dir / "agent_experiment_registry.csv"
    registry.to_csv(registry_path, index=False)
    if dry_run:
        return registry

    features = load_or_build_features(config, rebuild_features)
    rows = []
    details = {}
    runtime_rows = []
    total_start = time.perf_counter()
    for strategy in strategies:
        print(f"[agent] running {strategy.name}", flush=True)
        start = time.perf_counter()
        result = run_one_strategy(features, config, strategy)
        elapsed = time.perf_counter() - start
        strategy_details = result.pop("details", [])
        details[strategy.name] = strategy_details
        result["elapsed_seconds"] = elapsed
        runtime_rows.append(
            {
                "name": strategy.name,
                "model": strategy.model,
                "feature_set": strategy.feature_set,
                "sampling": strategy.sampling,
                "threshold_mode": strategy.threshold_mode,
                "per_type": strategy.per_type,
                "elapsed_seconds": elapsed,
                "input_rows": len(features),
                "input_columns": len(features.columns),
                "train_samples": int(sum(row.get("train_samples", 0) for row in strategy_details)),
                "valid_samples": int(sum(row.get("valid_samples", 0) for row in strategy_details)),
                "feature_count": int(max((row.get("feature_count", 0) for row in strategy_details), default=0)),
                "f1": result.get("f1", 0.0),
            }
        )
        rows.append(result)
    result_df = pd.DataFrame(rows).sort_values(["f1", "recall", "precision"], ascending=False)
    result_df.to_csv(config.output_dir / "agent_ablation_results.csv", index=False)
    runtime_df = pd.DataFrame(runtime_rows).sort_values("elapsed_seconds", ascending=False)
    runtime_df.to_csv(config.output_dir / "agent_runtime_log.csv", index=False)
    best = result_df.iloc[0].to_dict() if not result_df.empty else {}
    total_elapsed = time.perf_counter() - total_start
    (config.output_dir / "agent_best_config.json").write_text(
        json.dumps(
            json_ready({"best": best, "total_elapsed_seconds": total_elapsed, "details": details}),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return result_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage2_feature.json")
    parser.add_argument("--dry-run", action="store_true", help="只输出实验登记表，不训练模型。")
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--max-strategies", type=int, default=None)
    args = parser.parse_args()
    result = run_agent(Path(args.config), args.dry_run, args.rebuild_features, args.max_strategies)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
