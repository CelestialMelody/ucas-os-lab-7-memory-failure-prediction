from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

try:
    from raw_smoke import Config, build_raw_smoke_features, evaluate, feature_columns, filter_feature_groups
except ImportError:  # pragma: no cover
    from .raw_smoke import Config, build_raw_smoke_features, evaluate, feature_columns, filter_feature_groups


ABLATIONS: dict[str, tuple[str, ...]] = {
    "baseline": ("baseline",),
    "baseline_bsfe": ("baseline", "bsfe"),
    "baseline_time_patch": ("baseline", "time_patch"),
    "baseline_time_point": ("baseline", "time_point"),
    "baseline_bsfe_time_patch": ("baseline", "bsfe", "time_patch"),
    "baseline_bsfe_time_patch_time_point": ("baseline", "bsfe", "time_patch", "time_point"),
}


def _with_suffix(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def run_ablation(config_path: Path, max_files: int, reuse_features: bool = True) -> pd.DataFrame:
    """运行 task1 特征消融。

    默认复用配置中的 `feature_path`，避免每个消融配置都重新遍历 Stage1 原始日志。
    若缓存缺失或传入 `--rebuild-features`，先生成包含全部候选特征组的表，再用
    `filter_feature_groups` 切出各组。这样不同消融共享同一批样本，结果差异只来自
    可见特征列。
    """

    start = time.perf_counter()
    config = Config.from_json(config_path)
    full_config = replace(config, feature_groups=("baseline", "bsfe", "time_patch", "time_point"))
    if reuse_features and config.feature_path.exists():
        full_features = pd.read_csv(config.feature_path)
    else:
        full_features = build_raw_smoke_features(full_config, max_files)
        config.feature_path.parent.mkdir(parents=True, exist_ok=True)
        full_features.to_csv(config.feature_path, index=False)
    missing_groups = []
    if not any(col.startswith("m2point_") for col in full_features.columns):
        missing_groups.append("time_point")
    if not any(col.startswith("m2patch_") for col in full_features.columns):
        missing_groups.append("time_patch/bsfe")
    if missing_groups:
        print(
            "[ablation] cached feature table misses "
            + ", ".join(missing_groups)
            + "; rerun with --rebuild-features when you need complete ablation.",
            flush=True,
        )

    rows = []
    for name, groups in ABLATIONS.items():
        features = filter_feature_groups(full_features, groups)
        if not feature_columns(features):
            # 缓存来自旧版本时可能缺少某个特征组。记录 skipped 行比隐式报错更利于
            # 对照报告中的结果来源。
            metrics = pd.DataFrame(
                [
                    {
                        "model": "skipped_no_features",
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1": 0.0,
                        "samples": len(features),
                        "positive_samples": int(features["label"].sum()) if "label" in features else 0,
                        "feature_count": 0,
                    }
                ]
            )
        else:
            metrics = evaluate(features, replace(config, feature_groups=groups)).copy()
        metrics.insert(0, "ablation", name)
        metrics.insert(1, "feature_groups", "+".join(groups))
        metrics.to_csv(_with_suffix(config.metrics_path, name), index=False)
        rows.append(metrics)
    result = pd.concat(rows, ignore_index=True).sort_values(["ablation", "f1", "recall"], ascending=[True, False, False])
    out_path = _with_suffix(config.metrics_path, "ablation")
    result.to_csv(out_path, index=False)
    elapsed = time.perf_counter() - start
    summary = {
        "config": str(config_path),
        "feature_source": str(config.feature_path),
        "output": str(out_path),
        "missing_cached_feature_groups": missing_groups,
        "elapsed_seconds": elapsed,
        "max_files": max_files,
        "samples": int(len(full_features)),
        "positive_samples": int(full_features["label"].sum()) if "label" in full_features else 0,
        "feature_count_full": len(feature_columns(full_features)),
        "ablations": {name: list(groups) for name, groups in ABLATIONS.items()},
    }
    out_path.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/stage1_type_a_m2mfp_smoke.json",
    )
    parser.add_argument("--max-files", type=int, default=300)
    parser.add_argument("--rebuild-features", action="store_true")
    args = parser.parse_args()
    result = run_ablation(Path(args.config), args.max_files, reuse_features=not args.rebuild_features)
    print(result.groupby("ablation").head(1).to_string(index=False))


if __name__ == "__main__":
    main()
