from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
TASK1_RESULTS = ROOT / "m2mfp-reproduction" / "results"
TASK1_FIGURES = ROOT / "m2mfp-reproduction" / "reports" / "figures"
TASK2_RESULTS = ROOT / "memory-failure-prediction-agent" / "results" / "stage2_feature"
TASK2_FIGURES = ROOT / "memory-failure-prediction-agent" / "reports" / "figures"

PALETTE = {
    "blue": "#6D97FB",
    "blue_light": "#B6C9F4",
    "green": "#95B780",
    "green_light": "#C1D6B5",
    "red": "#F7716C",
    "red_dark": "#9F3737",
    "yellow": "#F7B964",
    "ink": "#2F3349",
    "grid": "#D9DEE8",
}

ABLATION_LABELS = {
    "baseline": "Baseline",
    "baseline_bsfe": "Baseline + BSFE",
    "baseline_time_patch": "Baseline + Time-patch",
    "baseline_time_point": "Baseline + Time-point",
    "baseline_bsfe_time_patch": "Baseline + BSFE + Time-patch",
    "baseline_bsfe_time_patch_time_point": "Full",
}


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": PALETTE["grid"],
            "axes.labelcolor": PALETTE["ink"],
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def read_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    gz_path = path.with_suffix(path.suffix + ".gz")
    if gz_path.exists():
        return pd.read_csv(gz_path)
    raise FileNotFoundError(path)


def ensure_dirs() -> None:
    TASK1_FIGURES.mkdir(parents=True, exist_ok=True)
    TASK2_FIGURES.mkdir(parents=True, exist_ok=True)


def best_ablation_rows(path: Path) -> pd.DataFrame:
    df = read_csv(path)
    order = list(ABLATION_LABELS)
    best = (
        df.sort_values(["ablation", "f1", "recall", "precision"], ascending=[True, False, False, False])
        .groupby("ablation", as_index=False)
        .head(1)
    )
    best["ablation"] = pd.Categorical(best["ablation"], categories=order, ordered=True)
    best = best.sort_values("ablation")
    best["label"] = best["ablation"].astype(str).map(ABLATION_LABELS)
    return best


def annotate_bars(ax: plt.Axes, fmt: str = "{:.3f}", pad: float = 0.008) -> None:
    for patch in ax.patches:
        height = patch.get_height()
        if np.isnan(height):
            continue
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height + pad,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9,
            color=PALETTE["ink"],
        )


def save(fig: plt.Figure, path: Path, rect: tuple[float, float, float, float] | None = None) -> None:
    fig.tight_layout(rect=rect)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_task1_ablation(csv_name: str, output_name: str, title: str) -> None:
    rows = best_ablation_rows(TASK1_RESULTS / csv_name)
    colors = [PALETTE["blue_light"]] * len(rows)
    if "baseline_time_point" in rows["ablation"].astype(str).tolist():
        colors[rows["ablation"].astype(str).tolist().index("baseline_time_point")] = PALETTE["blue"]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.bar(rows["label"], rows["f1"], color=colors, edgecolor="white", linewidth=1.2)
    annotate_bars(ax)
    ax.set_ylim(0, max(0.82, float(rows["f1"].max()) + 0.08))
    ax.set_ylabel("F1")
    ax.set_xlabel("")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=24)
    ax.grid(axis="x", visible=False)
    save(fig, TASK1_FIGURES / output_name)


def make_task1_figures() -> None:
    plot_task1_ablation(
        "stage1_type_a_m2mfp_smoke_metrics_ablation.csv",
        "task1_ablation_f1.png",
        "Task 1 Type A Smoke Ablation",
    )
    plot_task1_ablation(
        "stage1_type_a_full_metrics_ablation.csv",
        "task1_type_a_full_ablation_f1.png",
        "Task 1 Type A Full Ablation",
    )
    plot_task1_ablation(
        "stage1_type_b_full_metrics_ablation.csv",
        "task1_type_b_full_ablation_f1.png",
        "Task 1 Type B Full Ablation",
    )
    plot_task1_ablation(
        "stage1_type_ab_full_metrics_ablation.csv",
        "task1_type_ab_full_ablation_f1.png",
        "Task 1 Type A+B Full Ablation",
    )
    make_task1_combined_ablation()


def make_task1_combined_ablation() -> None:
    datasets = [
        ("Type A", "stage1_type_a_full_metrics_ablation.csv", PALETTE["yellow"]),
        ("Type B", "stage1_type_b_full_metrics_ablation.csv", PALETTE["green"]),
        ("Type A+B", "stage1_type_ab_full_metrics_ablation.csv", PALETTE["blue"]),
    ]
    rows = []
    for dataset, csv_name, color in datasets:
        best = best_ablation_rows(TASK1_RESULTS / csv_name)
        best["dataset"] = dataset
        best["color"] = color
        rows.append(best[["ablation", "label", "f1", "dataset", "color"]])
    df = pd.concat(rows, ignore_index=True)
    order = [
        "baseline_time_point",
        "baseline_bsfe_time_patch_time_point",
        "baseline_bsfe",
        "baseline_bsfe_time_patch",
        "baseline_time_patch",
        "baseline",
    ]
    label_order = [ABLATION_LABELS[key] for key in order]
    y = np.arange(len(order))
    height = 0.18
    offsets = {"Type A": height, "Type B": 0.0, "Type A+B": -height}
    colors = {dataset: color for dataset, _, color in datasets}
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for dataset, _, _ in datasets:
        part = df[df["dataset"] == dataset].set_index("ablation").reindex(order)
        values = part["f1"].to_numpy(dtype=float)
        ax.barh(y + offsets[dataset], values, height, label=dataset, color=colors[dataset])
        for yi, value in zip(y, values):
            if np.isnan(value):
                continue
            ax.text(value + 0.008, yi + offsets[dataset], f"{value:.3f}", va="center", ha="left", fontsize=8.5, color=PALETTE["ink"])
    ax.set_yticks(y)
    ax.set_yticklabels(label_order)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.84)
    ax.set_xlabel("F1")
    ax.set_ylabel("")
    ax.set_title("Task 1 Full Ablation Summary")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), title="")
    save(fig, TASK1_FIGURES / "task1_full_ablation_f1_combined.png")


def make_task2_model_comparison() -> None:
    df = read_csv(TASK2_RESULTS / "stage2_metrics.csv").sort_values("f1", ascending=False)
    name_map = {
        "xgboost": "XGBoost",
        "hist_gradient_boosting": "HistGB",
        "lightgbm": "LightGBM",
        "random_forest": "RandomForest",
    }
    df["model_label"] = df["model"].map(name_map).fillna(df["model"])
    y = np.arange(len(df))
    height = 0.22
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.barh(y + height, df["precision"], height, label="Precision", color=PALETTE["green"])
    ax.barh(y, df["recall"], height, label="Recall", color=PALETTE["yellow"])
    ax.barh(y - height, df["f1"], height, label="F1", color=PALETTE["blue"])
    ax.set_yticks(y)
    ax.set_yticklabels(df["model_label"])
    ax.invert_yaxis()
    ax.set_xlim(0, 0.82)
    ax.set_xlabel("Score")
    ax.set_ylabel("")
    ax.set_title("Task 2 Model Comparison")
    ax.grid(axis="y", visible=False)
    ax.legend(ncols=3, loc="lower right")
    for values, offset in [(df["precision"], height), (df["recall"], 0), (df["f1"], -height)]:
        for yi, value in zip(y, values):
            ax.text(value + 0.012, yi + offset, f"{value:.3f}", va="center", ha="left", fontsize=8, color=PALETTE["ink"])
    save(fig, TASK2_FIGURES / "task2_model_comparison.png")


def threshold_files() -> list[Path]:
    return sorted(TASK2_RESULTS.glob("threshold_search_*.csv"))


def make_task2_threshold_f1() -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    colors = [PALETTE["blue"], PALETTE["green"], PALETTE["red"], PALETTE["yellow"]]
    for idx, path in enumerate(threshold_files()):
        model = path.stem.replace("threshold_search_", "")
        df = read_csv(path).sort_values("threshold")
        ax.plot(df["threshold"], df["f1"], label=model, color=colors[idx % len(colors)], linewidth=2.0)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1")
    ax.set_title("Task 2 Threshold Search")
    ax.set_ylim(0, 0.62)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncols=2)
    save(fig, TASK2_FIGURES / "task2_threshold_f1.png")


def make_task2_threshold_panels() -> None:
    files = threshold_files()
    name_map = {
        "xgboost": "XGBoost",
        "hist_gradient_boosting": "HistGB",
        "lightgbm": "LightGBM",
        "random_forest": "RandomForest",
    }
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.6), sharex=True, sharey=True)
    for ax, path in zip(axes.ravel(), files):
        model = path.stem.replace("threshold_search_", "")
        df = read_csv(path).sort_values("threshold")
        ax.plot(df["threshold"], df["precision"], label="Precision", color=PALETTE["green"], linewidth=1.8)
        ax.plot(df["threshold"], df["recall"], label="Recall", color=PALETTE["yellow"], linewidth=1.8)
        ax.plot(df["threshold"], df["f1"], label="F1", color=PALETTE["blue"], linewidth=2.1)
        best = df.sort_values(["f1", "recall", "precision"], ascending=False).iloc[0]
        ax.axvline(best["threshold"], color=PALETTE["red"], linewidth=1.2, linestyle="--")
        ax.annotate(
            f"best={best['threshold']:.2f}",
            xy=(best["threshold"], best["f1"]),
            xytext=(6, 8),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=8,
            color=PALETTE["red_dark"],
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": PALETTE["grid"], "linewidth": 0.6},
        )
        ax.set_title(name_map.get(model, model))
        ax.set_ylim(0, 1.0)
    for ax in axes[-1, :]:
        ax.set_xlabel("Threshold")
    for ax in axes[:, 0]:
        ax.set_ylabel("Score")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Task 2 Threshold Search by Model", y=0.99, fontsize=13, color=PALETTE["ink"])
    fig.legend(handles, labels, loc="upper center", ncols=3, bbox_to_anchor=(0.5, 0.955))
    save(fig, TASK2_FIGURES / "task2_threshold_f1_panels.png", rect=(0, 0, 1, 0.91))


def make_stage2_label_distribution() -> None:
    df = read_csv(TASK2_RESULTS / "stage2_train_features.csv")
    counts = (
        df.groupby(["serial_number_type", "label"])
        .size()
        .rename("count")
        .reset_index()
        .replace({"label": {0: "Negative", 1: "Positive"}})
    )
    type_order = sorted(counts["serial_number_type"].unique())
    label_order = ["Negative", "Positive"]
    pivot = counts.pivot(index="serial_number_type", columns="label", values="count").reindex(index=type_order, columns=label_order).fillna(0)
    x = np.arange(len(type_order))
    width = 0.22
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.bar(x - width / 2, pivot["Negative"], width, label="Negative", color=PALETTE["blue_light"], edgecolor="white", linewidth=1.0)
    ax.bar(x + width / 2, pivot["Positive"], width, label="Positive", color=PALETTE["red"], edgecolor="white", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(type_order)
    ax.set_xlim(-0.55, len(type_order) - 0.45)
    ax.set_xlabel("Serial Number Type")
    ax.set_ylabel("Samples (log scale)")
    ax.set_title("Stage2 Label Distribution")
    ax.grid(axis="x", visible=False)
    ax.legend(title="")
    save(fig, TASK2_FIGURES / "stage2_label_distribution.png")


def make_agent_strategy_ablation() -> None:
    df = read_csv(TASK2_RESULTS / "agent_ablation_results.csv").sort_values("f1", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    colors = [PALETTE["blue"] if idx == 0 else PALETTE["green_light"] for idx in range(len(df))]
    ax.barh(df["name"][::-1], df["f1"][::-1], color=colors[::-1], edgecolor="white", linewidth=1.0)
    ax.set_xlim(0, max(0.60, float(df["f1"].max()) + 0.04))
    ax.set_xlabel("F1")
    ax.set_ylabel("")
    ax.set_title("Agent Strategy Ablation")
    ax.grid(axis="y", visible=False)
    for patch in ax.patches:
        width = patch.get_width()
        ax.text(width + 0.006, patch.get_y() + patch.get_height() / 2, f"{width:.3f}", va="center", fontsize=9)
    save(fig, TASK2_FIGURES / "agent_strategy_ablation.png")


def make_best_submission_scores() -> None:
    df = read_csv(TASK2_RESULTS / "stage2_best_per_type_prediction_scores.csv").sort_values("score", ascending=False)
    df = df.reset_index(drop=True)
    x = df.index + 1
    max_score = float(df["score"].max()) if len(df) else 0.0
    thresholds = {
        str(sn_type): float(part["threshold"].iloc[0])
        for sn_type, part in df.groupby("serial_number_type")
    }
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.plot(x, df["score"], color=PALETTE["blue"], linewidth=2.2)
    ax.fill_between(x, df["score"], color=PALETTE["blue_light"], alpha=0.35)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Risk Score")
    ax.set_title("Task 2 Top-200 Submission Scores")
    ax.set_xlim(1, max(len(df), 1))
    ax.set_ylim(0, max_score * 1.35 if max_score > 0 else 1)
    ax.grid(axis="x", visible=False)
    if thresholds:
        threshold_text = "\n".join(f"type {k} validation threshold: {v:.3f}" for k, v in sorted(thresholds.items()))
        threshold_text += f"\nmax target score: {max_score:.4f}"
        ax.text(
            0.985,
            0.94,
            threshold_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            color=PALETTE["ink"],
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": PALETTE["grid"], "linewidth": 0.8},
        )
    save(fig, TASK2_FIGURES / "task2_best_submission_scores.png")


def make_winners_reference() -> None:
    rows = [
        ("NOVA", "Codabench", 0.314),
        ("NOVA", "Reproduced", 0.284),
        ("Qiming", "Codabench", 0.618),
        ("Qiming", "Reproduced", 0.301),
        ("Dushi Yuanfen", "Codabench", 0.615),
        ("Dushi Yuanfen", "Reproduced", 0.285),
        ("Tilimo", "Codabench", 0.368),
        ("Tilimo", "Reproduced", 0.237),
        ("first_wewillwin", "Codabench", 0.485),
        ("first_wewillwin", "Reproduced", 0.191),
        ("TimeSeries Star", "Codabench", 0.299),
        ("TimeSeries Star", "Reproduced", 0.213),
        ("This project", "Local validation", 0.5538),
    ]
    df = pd.DataFrame(rows, columns=["team", "metric", "f1"])
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    sns.barplot(
        data=df,
        x="team",
        y="f1",
        hue="metric",
        palette={"Codabench": PALETTE["blue_light"], "Reproduced": PALETTE["green"], "Local validation": PALETTE["red"]},
        ax=ax,
    )
    ax.set_ylim(0, 0.70)
    ax.set_xlabel("")
    ax.set_ylabel("F1")
    ax.set_title("External Winners Reference")
    ax.tick_params(axis="x", rotation=24)
    ax.grid(axis="x", visible=False)
    ax.legend(title="")
    save(fig, TASK2_FIGURES / "task2_winners_reference.png")


def make_task2_figures() -> None:
    make_task2_model_comparison()
    make_task2_threshold_f1()
    make_task2_threshold_panels()
    make_stage2_label_distribution()
    make_agent_strategy_ablation()
    make_best_submission_scores()
    make_winners_reference()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["all", "task1", "task2"], default="all")
    args = parser.parse_args()
    setup_style()
    ensure_dirs()
    if args.scope in {"all", "task1"}:
        make_task1_figures()
    if args.scope in {"all", "task2"}:
        make_task2_figures()


if __name__ == "__main__":
    main()
