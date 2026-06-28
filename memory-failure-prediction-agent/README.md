# Memory Failure Prediction Agent

本项目用于在 SmartHW/SmartMem Stage2 官方预提取特征上训练 UE 预测模型、搜索
agent 策略并生成离线候选 submission。

## Scope

任务二主线使用官方预提取的 Stage2 表格特征。agent 搜索覆盖：

- 模型：RandomForest、HistGradientBoosting、LightGBM、XGBoost。
- 采样：不采样、balanced、hard negative。
- 特征组：全部特征、时间强度特征、空间/bit 特征。
- 决策：阈值搜索、top-k。
- 分类型：type_A/type_B per-type 模型和阈值。

## Layout

```text
memory-failure-prediction-agent/
├── src/                    # Stage2 pipeline 和 agent 搜索
├── common/                 # 配置路径解析工具
├── configs/                # Stage2 feature 配置
├── results/stage2_feature/ # 指标、阈值、agent 结果和 submission
├── reports/                # 报告、运行手册、PPT 大纲和图表
└── Makefile                # 推荐运行入口
```

## Data

不提交 GB 级原始数据。本目录默认通过配置中的相对路径读取：

```text
../data/stage2_feature/
├── failure_ticket.csv
├── type_A/*.feather
└── type_B/*.feather
```

换机器时使用：

```bash
make task2-agent MEMFAIL_DATA_ROOT=/path/to/data
```

`MEMFAIL_DATA_ROOT` 应指向包含 `stage2_feature/` 的 `data` 目录本身。

## Cached Features

本仓库提交了 `results/stage2_feature/stage2_train_features.csv.gz` 压缩训练缓存。
首次使用已有缓存前执行：

```bash
make restore-caches
```

如果有 Stage2 官方特征 feather 数据，也可以运行 `make task2-train` 重新构造。

## Commands

```bash
make help
make compile
make figures
make task2-agent-quick
```

完整流程：

```bash
make task2-train
make task2-agent
make task2-best-submission
```

`make task2-best-submission` 会重新训练最佳本地验证策略
`xgb_all_none_per_type`，遍历目标预测期并生成：

```text
results/stage2_feature/stage2_best_per_type_prediction_scores.csv
results/stage2_feature/stage2_best_per_type_submission.csv
```

## Main Results

| Method | Local validation F1 |
| --- | ---: |
| baseline XGBoost | 0.5405 |
| agent xgb_all_none_per_type | 0.5538 |

已生成 200 行离线候选 submission，格式检查通过：无空值、无重复
`serial_number`，时间戳位于目标预测期。目标期分数低于 per-type 验证阈值，因此
本次 submission 使用 top-k 兜底；该文件作为离线可提交格式和候选告警结果。
项目未提交 Codabench 获取隐藏测试 F1，效果结论以本地验证 F1 为准。

## Figures

关键图表位于 `reports/figures/`：

- `task2_model_comparison.png`
- `task2_threshold_f1.png`
- `task2_threshold_f1_panels.png`
- `stage2_label_distribution.png`
- `agent_strategy_ablation.png`
- `task2_best_submission_scores.png`
- `task2_winners_reference.png`

完整说明见 `docs/`、`reports/report.md` 和 `reports/runbook.md`。
