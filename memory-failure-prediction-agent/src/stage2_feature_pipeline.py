from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

try:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from common.path_utils import config_path_prefix, resolve_path
except ImportError:  # pragma: no cover
    from ...common.path_utils import config_path_prefix, resolve_path

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


ONE_DAY = 86400


@dataclass(frozen=True)
class Stage2Config:
    """SmartHW/SmartMem 第二阶段特征训练配置。

    task2 读取官方预提取特征，训练模型，搜索阈值，并生成可提交的
    `serial_number/prediction_timestamp/serial_number_type` 表。目标期边界和
    `submission_top_k` 放在配置中，便于复查 submission 的生成条件。
    """

    feature_root: Path
    ticket_path: Path
    output_dir: Path
    prediction_period_days: int = 7
    lead_minutes: int = 15
    validation_quantile: float = 0.8
    target_start: int = 1722441600
    target_end: int = 1727712000
    submission_top_k: int = 200
    random_state: int = 42

    @classmethod
    def from_json(cls, path: Path) -> "Stage2Config":
        """读取配置，并把相对路径解析到 task2 项目根目录。"""

        raw = json.loads(path.read_text(encoding="utf-8"))
        base = path.parent.parent
        data_prefix = config_path_prefix(raw, base)
        for key in ["feature_root", "ticket_path"]:
            raw[key] = resolve_path(raw[key], base, data_prefix)
        raw["output_dir"] = resolve_path(raw["output_dir"], base)
        raw.pop("path_prefix", None)
        return cls(**raw)


def iter_feature_files(root: Path) -> list[tuple[Path, str]]:
    """枚举 Stage2 官方特征文件。

    SmartHW 阶段二特征数据按 type_A/type_B 分目录保存。每个 feather 文件对应
    一个 DIMM 的时间序列特征。返回时携带 DIMM 类型，供 per-type 训练和提交
    文件使用。
    """

    files: list[tuple[Path, str]] = []
    for type_dir, sn_type in [(root / "type_A", "A"), (root / "type_B", "B")]:
        if type_dir.exists():
            files.extend((path, sn_type) for path in sorted(type_dir.glob("*.feather")))
    return files


def read_ticket(path: Path) -> pd.DataFrame:
    """读取 Stage2 failure ticket，并统一列名。"""

    ticket = pd.read_csv(path)
    return ticket.rename(
        columns={
            "serial_number": "sn_name",
            "failure_time": "alarm_time",
            "serial_number_type": "sn_type",
        }
    )


def feature_columns(df: pd.DataFrame) -> list[str]:
    """返回训练用特征列，排除样本元信息和标签。"""

    excluded = {"serial_number", "serial_number_type", "prediction_timestamp", "label", "split"}
    return [col for col in df.columns if col not in excluded]


def align_prediction_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """按训练列顺序对齐预测特征，缺失窗口列补 0。

    Stage2 官方 feather 在 type_A/type_B 间列空间不完全一致：部分 type_B 文件
    缺少 6h 窗口特征。训练缓存表合并后包含完整列集，因此提交生成必须按训练列
    顺序重排。缺失列补 0，多余列丢弃，避免预测期输入矩阵和训练期列空间不一致。
    """

    return df.reindex(columns=cols, fill_value=0).fillna(0)


def add_meta(row: pd.Series, serial_number: str, sn_type: str, label: int) -> dict[str, object]:
    """给官方特征行追加监督学习所需的元信息。

    元信息列不参与模型训练，但用于时间切分、per-type 分组和 submission 输出。
    """

    out = row.to_dict()
    out["serial_number"] = serial_number
    out["serial_number_type"] = sn_type
    out["prediction_timestamp"] = int(row["LogTime"])
    out["label"] = int(label)
    return out


def build_training_table(config: Stage2Config) -> pd.DataFrame:
    """把每个 DIMM 的时间序列特征压成训练样本表。

    正样本：对 ticket 中有故障的 DIMM，在故障前 `prediction_period_days` 到
    `lead_minutes` 的预测区间内取最后一个特征点。该点模拟在 lead time 之前
    发出告警。

    负样本：对故障 DIMM 取更早的安全时间点；对无故障 DIMM 取最后一个
    特征点。故障 DIMM 的负样本必须早于预测窗口，避免把故障前片段标成负类。
    """

    ticket = read_ticket(config.ticket_path)
    ticket_map = {
        str(row.sn_name): (int(row.alarm_time), str(row.sn_type))
        for row in ticket.itertuples(index=False)
    }
    rows: list[dict[str, object]] = []
    lead = config.lead_minutes * 60
    pred = config.prediction_period_days * ONE_DAY
    for path, sn_type in iter_feature_files(config.feature_root):
        sn_name = path.stem
        df = pd.read_feather(path).sort_values("LogTime").reset_index(drop=True)
        if df.empty:
            continue
        if sn_name in ticket_map:
            alarm_time, ticket_type = ticket_map[sn_name]
            sn_type = ticket_type or sn_type
            pos_start = alarm_time - lead - pred
            pos_end = alarm_time - lead
            pos_df = df[(df["LogTime"] >= pos_start) & (df["LogTime"] <= pos_end)]
            if not pos_df.empty:
                rows.append(add_meta(pos_df.iloc[-1], sn_name, sn_type, 1))
            neg_df = df[df["LogTime"] < pos_start - ONE_DAY]
            if not neg_df.empty:
                rows.append(add_meta(neg_df.iloc[-1], sn_name, sn_type, 0))
        else:
            rows.append(add_meta(df.iloc[-1], sn_name, sn_type, 0))
    features = pd.DataFrame(rows).fillna(0)
    meta = ["serial_number", "serial_number_type", "prediction_timestamp", "label"]
    ordered = meta + [col for col in features.columns if col not in meta]
    return features[ordered]


def build_models(random_state: int) -> dict[str, object]:
    """构造 task2 的候选模型库。

    默认使用 CPU 可运行模型，保证无 GPU 环境也能复现。若环境安装了
    LightGBM/XGBoost，会自动加入候选。参数保持固定，便于比较模型类型差异。
    """

    models: dict[str, object] = {
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=200,
            max_leaf_nodes=31,
            learning_rate=0.05,
            random_state=random_state,
        ),
    }
    if LGBMClassifier is not None:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )
    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=random_state,
        )
    return models


def time_split(features: pd.DataFrame, quantile: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按预测时间做近似时序验证切分。

    内存故障预测存在时间分布漂移风险，所以默认用较早时间训练、较晚时间验证。
    如果切分后某一侧只有单一类别，则退回随机切分，保证指标仍可计算。
    """

    cutoff = int(features["prediction_timestamp"].quantile(quantile))
    train = features[features["prediction_timestamp"] <= cutoff].copy()
    valid = features[features["prediction_timestamp"] > cutoff].copy()
    if valid["label"].nunique() < 2 or train["label"].nunique() < 2:
        rng = np.random.default_rng(42)
        mask = rng.random(len(features)) < quantile
        train = features[mask].copy()
        valid = features[~mask].copy()
    train["split"] = "train"
    valid["split"] = "valid"
    return train, valid


def threshold_search(y_true: pd.Series, score: np.ndarray) -> pd.DataFrame:
    """搜索概率阈值，按 F1/Recall/Precision 排序。

    SmartMem 类任务正负样本比例悬殊，默认 0.5 阈值通常不能给出最佳 F1。
    因此每个模型都在验证集上搜索阈值，并把完整曲线写入 CSV。
    """

    rows = []
    for threshold in np.linspace(0.01, 0.99, 99):
        pred = (score >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": precision_score(y_true, pred, zero_division=0),
                "recall": recall_score(y_true, pred, zero_division=0),
                "f1": f1_score(y_true, pred, zero_division=0),
            }
        )
    return pd.DataFrame(rows).sort_values(["f1", "recall", "precision"], ascending=False)


def train_and_evaluate(config: Stage2Config, features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """训练候选模型、验证并保存最佳模型信息。

    先在时序验证集上为每个模型搜索阈值，再用全量训练表重训最佳模型。保存的
    bundle 可直接用于 submission 生成；指标仍来自未参与训练的验证时间段。
    """

    train, valid = time_split(features, config.validation_quantile)
    cols = feature_columns(features)
    metrics = []
    bundles: dict[str, dict[str, object]] = {}
    for name, model in build_models(config.random_state).items():
        model.fit(train[cols], train["label"].astype(int))
        if hasattr(model, "predict_proba"):
            score = model.predict_proba(valid[cols])[:, 1]
        else:
            score = model.predict(valid[cols])
        thresholds = threshold_search(valid["label"].astype(int), score)
        best_threshold = float(thresholds.iloc[0]["threshold"])
        metrics.append(
            {
                "model": name,
                "threshold": best_threshold,
                "precision": float(thresholds.iloc[0]["precision"]),
                "recall": float(thresholds.iloc[0]["recall"]),
                "f1": float(thresholds.iloc[0]["f1"]),
                "train_samples": len(train),
                "valid_samples": len(valid),
                "train_positive": int(train["label"].sum()),
                "valid_positive": int(valid["label"].sum()),
                "feature_count": len(cols),
            }
        )
        bundles[name] = {"model": model, "columns": cols, "thresholds": thresholds}
        thresholds.to_csv(config.output_dir / f"threshold_search_{name}.csv", index=False)
    metrics_df = pd.DataFrame(metrics).sort_values(["f1", "recall", "precision"], ascending=False)
    best_name = str(metrics_df.iloc[0]["model"])
    best = bundles[best_name]
    final_model = build_models(config.random_state)[best_name]
    final_model.fit(features[cols], features["label"].astype(int))
    bundle = {
        "model": final_model,
        "columns": cols,
        "model_name": best_name,
        "threshold": float(metrics_df.iloc[0]["threshold"]),
        "config": asdict(config),
    }
    return metrics_df, bundle


def generate_submission(config: Stage2Config, bundle: dict[str, object]) -> pd.DataFrame:
    """在目标预测时间段上生成提交文件。

    先按验证集最佳阈值筛选告警。若目标期没有样本过阈值，则退回 top-k 最高风险
    DIMM，生成非空候选文件。top-k 只表示相对风险排序，不代表隐藏测试效果。
    """

    model = bundle["model"]
    cols = bundle["columns"]
    threshold = float(bundle["threshold"])
    rows = []
    score_rows = []
    files = iter_feature_files(config.feature_root)
    for idx, (path, sn_type) in enumerate(files, start=1):
        if idx % 5000 == 0:
            print(f"[submission] processed {idx}/{len(files)} files, selected {len(rows)} rows", flush=True)
        sn_name = path.stem
        df = pd.read_feather(path)
        target_df = df[(df["LogTime"] >= config.target_start) & (df["LogTime"] <= config.target_end)].copy()
        if target_df.empty:
            continue
        pred_frame = align_prediction_features(target_df, cols)
        score = model.predict_proba(pred_frame)[:, 1] if hasattr(model, "predict_proba") else model.predict(pred_frame)
        target_df["score"] = score
        selected = target_df[target_df["score"] >= threshold]
        if selected.empty:
            continue
        # 每个 DIMM 只保留最高分告警，避免同一序列号在提交文件中重复出现。
        best = selected.sort_values("score", ascending=False).iloc[0]
        rows.append(
            {
                "serial_number": sn_name,
                "prediction_timestamp": int(best["LogTime"]),
                "serial_number_type": sn_type,
            }
        )
        score_rows.append({**rows[-1], "score": float(best["score"])})
    submission = pd.DataFrame(rows)
    score_df = pd.DataFrame(score_rows).sort_values("score", ascending=False) if score_rows else pd.DataFrame()
    if score_df.empty and config.submission_top_k > 0:
        # 目标期无标签，分数分布可能低于验证期。若阈值策略没有选出样本，
        # 使用 top-k 生成候选告警文件。
        all_scores = []
        for idx, (path, sn_type) in enumerate(files, start=1):
            if idx % 5000 == 0:
                print(f"[submission-topk] processed {idx}/{len(files)} files", flush=True)
            sn_name = path.stem
            df = pd.read_feather(path)
            target_df = df[(df["LogTime"] >= config.target_start) & (df["LogTime"] <= config.target_end)].copy()
            if target_df.empty:
                continue
            pred_frame = align_prediction_features(target_df, cols)
            score = model.predict_proba(pred_frame)[:, 1] if hasattr(model, "predict_proba") else model.predict(pred_frame)
            target_df["score"] = score
            best = target_df.sort_values("score", ascending=False).iloc[0]
            all_scores.append(
                {
                    "serial_number": sn_name,
                    "prediction_timestamp": int(best["LogTime"]),
                    "serial_number_type": sn_type,
                    "score": float(best["score"]),
                }
            )
        score_df = pd.DataFrame(all_scores).sort_values("score", ascending=False).head(config.submission_top_k)
        rows = score_df[["serial_number", "prediction_timestamp", "serial_number_type"]].to_dict("records")
        submission = pd.DataFrame(rows)
    if not submission.empty:
        submission = submission.sort_values(["serial_number_type", "serial_number", "prediction_timestamp"])
    score_df.to_csv(config.output_dir / "stage2_prediction_scores.csv", index=False)
    return submission


def generate_submission_from_train_table(config: Stage2Config, bundle: dict[str, object]) -> pd.DataFrame:
    """调试用提交生成函数。

    它只读取已缓存的 `stage2_train_features.csv`，速度比重新遍历所有 feather
    文件快，适合检查阈值和 top-k 输出格式。
    """

    path = config.output_dir / "stage2_train_features.csv"
    features = pd.read_csv(path)
    target = features[(features["prediction_timestamp"] >= config.target_start) & (features["prediction_timestamp"] <= config.target_end)].copy()
    model = bundle["model"]
    cols = bundle["columns"]
    threshold = float(bundle["threshold"])
    pred_frame = align_prediction_features(target, cols)
    score = model.predict_proba(pred_frame)[:, 1] if hasattr(model, "predict_proba") else model.predict(pred_frame)
    target["score"] = score
    selected = target[target["score"] >= threshold].copy()
    if selected.empty and config.submission_top_k > 0:
        selected = target.sort_values("score", ascending=False).head(config.submission_top_k).copy()
    score_cols = ["serial_number", "prediction_timestamp", "serial_number_type", "score", "label"]
    selected[score_cols].to_csv(config.output_dir / "stage2_prediction_scores_from_train_table.csv", index=False)
    return selected[["serial_number", "prediction_timestamp", "serial_number_type"]].copy()


def load_agent_best_thresholds(config: Stage2Config) -> dict[str, float]:
    """读取 agent 最佳 per-type 阈值，缺失时退回验证结果中的默认值。

    `agent_best_config.json` 中记录验证集搜索得到的分类型阈值。若该文件缺失，
    使用已知最佳实验阈值作为可复现默认值，保证 submission 入口不依赖手工传参。
    """

    path = config.output_dir / "agent_best_config.json"
    defaults = {"A": 0.28, "B": 0.12}
    if not path.exists():
        return defaults
    raw = json.loads(path.read_text(encoding="utf-8"))
    details = raw.get("details", {}).get("xgb_all_none_per_type", [])
    thresholds = defaults.copy()
    for row in details:
        sn_type = str(row.get("serial_number_type", ""))
        if sn_type in thresholds and row.get("threshold") is not None:
            thresholds[sn_type] = float(row["threshold"])
    return thresholds


def train_best_per_type_bundle(config: Stage2Config) -> dict[str, object]:
    """训练 agent 最佳策略：XGBoost + 全部特征 + per-type。

    每个 DIMM 类型单独训练一个模型，但共享同一组官方特征列。这样保留
    type_A/type_B 的分布差异，同时让两个类型的输入列顺序一致。
    """

    try:
        from agent_search import build_model, select_feature_columns
    except ImportError:  # pragma: no cover
        from .agent_search import build_model, select_feature_columns

    train_path = config.output_dir / "stage2_train_features.csv"
    features = pd.read_csv(train_path) if train_path.exists() else build_training_table(config)
    if not train_path.exists():
        features.to_csv(train_path, index=False)
    thresholds = load_agent_best_thresholds(config)
    cols = select_feature_columns(features, "all")
    models = {}
    for sn_type, part in features.groupby("serial_number_type"):
        model = build_model("xgboost", config.random_state)
        model.fit(part[cols], part["label"].astype(int))
        models[str(sn_type)] = model
    return {
        "strategy": "xgb_all_none_per_type",
        "models": models,
        "columns": cols,
        "thresholds": thresholds,
        "config": asdict(config),
    }


def generate_best_per_type_submission(config: Stage2Config, bundle: dict[str, object]) -> pd.DataFrame:
    """使用最佳 per-type agent 策略遍历 Stage2 feature 目标期并生成提交。

    阈值命中的样本会先被选入提交。若目标预测期所有分数都低于验证阈值，
    再从每个 DIMM 的最高风险点中取 top-k。score CSV 保留分数和阈值，用于
    说明 200 行候选来自 top-k 兜底。
    """

    models = bundle["models"]
    cols = bundle["columns"]
    thresholds = bundle["thresholds"]
    rows = []
    score_rows = []
    all_score_rows = []
    files = iter_feature_files(config.feature_root)
    for idx, (path, sn_type) in enumerate(files, start=1):
        if idx % 5000 == 0:
            print(f"[best-submission] processed {idx}/{len(files)} files, selected {len(rows)} rows", flush=True)
        model = models.get(sn_type)
        if model is None:
            continue
        sn_name = path.stem
        df = pd.read_feather(path)
        target_df = df[(df["LogTime"] >= config.target_start) & (df["LogTime"] <= config.target_end)].copy()
        if target_df.empty:
            continue
        pred_frame = align_prediction_features(target_df, cols)
        target_df["score"] = model.predict_proba(pred_frame)[:, 1]
        # 无论是否超过阈值，都记录每个 DIMM 在目标期的最高风险点。这些记录用于
        # top-k 兜底和分数分布图。
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
        selected = target_df[target_df["score"] >= float(thresholds.get(sn_type, 0.5))]
        if selected.empty:
            continue
        best = selected.sort_values("score", ascending=False).iloc[0]
        row = {
            "serial_number": sn_name,
            "prediction_timestamp": int(best["LogTime"]),
            "serial_number_type": sn_type,
        }
        rows.append(row)
        score_rows.append({**row, "score": float(best["score"]), "threshold": float(thresholds.get(sn_type, 0.5))})
    score_df = pd.DataFrame(score_rows).sort_values("score", ascending=False) if score_rows else pd.DataFrame()
    submission = pd.DataFrame(rows)
    if submission.empty and config.submission_top_k > 0:
        # 验证阈值没有选出目标期告警时会走到这里。该兜底只生成候选告警列表，
        # 不代表目标期隐藏测试 F1。
        all_score_df = pd.DataFrame(all_score_rows).sort_values("score", ascending=False)
        score_df = all_score_df.head(config.submission_top_k)
        submission = score_df[["serial_number", "prediction_timestamp", "serial_number_type"]].copy()
    if not submission.empty:
        submission = submission.sort_values(["serial_number_type", "serial_number", "prediction_timestamp"])
    score_df.to_csv(config.output_dir / "stage2_best_per_type_prediction_scores.csv", index=False)
    return submission


def run(config_path: Path, process: str = "all") -> None:
    """task2 命令行主流程。"""

    config = Stage2Config.from_json(config_path)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if process == "submission":
        bundle = joblib.load(config.output_dir / "stage2_best_model.joblib")
        submission = generate_submission_from_train_table(config, bundle)
        submission.to_csv(config.output_dir / "stage2_submission.csv", index=False)
        print(f"[submission] wrote {len(submission)} rows", flush=True)
        return
    if process == "best_per_type_submission":
        bundle = train_best_per_type_bundle(config)
        joblib.dump(bundle, config.output_dir / "stage2_best_per_type_model.joblib")
        submission = generate_best_per_type_submission(config, bundle)
        submission.to_csv(config.output_dir / "stage2_best_per_type_submission.csv", index=False)
        print(f"[best-submission] wrote {len(submission)} rows", flush=True)
        return

    features = build_training_table(config)
    features.to_csv(config.output_dir / "stage2_train_features.csv", index=False)
    metrics, bundle = train_and_evaluate(config, features)
    metrics.to_csv(config.output_dir / "stage2_metrics.csv", index=False)
    joblib.dump(bundle, config.output_dir / "stage2_best_model.joblib")
    submission = generate_submission(config, bundle)
    submission.to_csv(config.output_dir / "stage2_submission.csv", index=False)
    summary = {
        "data": {
            "feature_root": str(config.feature_root),
            "ticket_path": str(config.ticket_path),
            "files": len(iter_feature_files(config.feature_root)),
        },
        "samples": {
            "train_table_rows": len(features),
            "positive": int(features["label"].sum()),
            "negative": int((features["label"] == 0).sum()),
        },
        "best": metrics.iloc[0].to_dict(),
        "submission_rows": len(submission),
        "notes": "Stage2 feature data is pre-engineered by the competition baseline. It is used for SmartHW task training/testing and agent optimization, not as the sole evidence for M2-MFP raw-log reproduction.",
    }
    (config.output_dir / "stage2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage2_feature.json")
    parser.add_argument("--process", choices=["all", "submission", "best_per_type_submission"], default="all")
    args = parser.parse_args()
    run(Path(args.config), args.process)


if __name__ == "__main__":
    main()
