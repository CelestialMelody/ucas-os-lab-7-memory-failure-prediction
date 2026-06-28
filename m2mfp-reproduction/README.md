# M2-MFP Reproduction for Memory Failure Prediction

本项目用于复现并检验 M2-MFP 风格特征在 SmartMem Stage1 原始 CE 日志上的有效性。

## Scope

本项目复现和本地适配以下特征机制：

- BSFE：把 `RetryRdErrLogParity` 转成 8x4 DQ-Beat bit 矩阵并提取空间 bit 特征。
- time-patch：在 15m/1h/6h 多尺度窗口内聚合时间、空间和 bit 特征。
- time-point：提取故障前 CE 突发、最近间隔、短长窗口激增和空间集中规则。

评估器使用 Logistic Regression 和轻量 RandomForest。公开 M2-MFP 代码包含
自定义 decision tree/time-point 模块，但不包含完整私有训练流水线；本项目聚焦
特征思想的实现和本地验证。

## Layout

```text
m2mfp-reproduction/
├── src/                 # BSFE/time-patch/time-point 和消融脚本
├── common/              # 配置路径解析工具
├── configs/             # Stage1 smoke/type_A/type_B/A+B 配置
├── results/             # 指标、JSON 摘要和压缩后的特征缓存
├── reports/             # 报告、运行手册、PPT 大纲和图表
└── Makefile             # 推荐运行入口
```

## Data

不提交 GB 级原始数据。本目录默认通过配置中的相对路径读取：

```text
../data/stage1_feather/
├── ticket.csv
├── type_A/*.feather
└── type_B/*.feather
```

换机器时使用：

```bash
make task1-smoke MEMFAIL_DATA_ROOT=/path/to/data
```

`MEMFAIL_DATA_ROOT` 应指向包含 `stage1_feather/` 的 `data` 目录本身。

## Cached Features

本仓库提交了 `results/*.csv.gz` 压缩特征缓存，避免公开仓库出现单个超过
100MB 的 CSV。首次使用已有缓存前执行：

```bash
make restore-caches
```

如果有原始 feather 数据，也可以用 `--rebuild-features` 对应的 full 命令重新生成。

## Commands

```bash
make help
make compile
make figures
make task1-smoke MAX_FILES=300
make task1-ablation MAX_FILES=300
```

全量命令会读取 Stage1 原始日志，已完成并保留结果：

```bash
make task1-full-a
make task1-full-b
make task1-full-ab
```

## Main Results

| Dataset | Best ablation | Model | F1 |
| --- | --- | --- | ---: |
| type_A smoke | baseline_time_point | random_forest_small | 0.7270 |
| type_A full | baseline_time_point | random_forest_small | 0.7587 |
| type_B full | baseline_time_point | random_forest_small | 0.6479 |
| type_A + type_B full | baseline_time_point | random_forest_small | 0.7645 |

结论：time-point 在 Stage1 原始日志上最稳定有效；在本项目的 RandomForest
配置下，BSFE/time-patch 的叠加收益弱于单独 time-point。

## Figures

关键图表位于 `reports/figures/`：

- `task1_ablation_f1.png`
- `task1_type_a_full_ablation_f1.png`
- `task1_type_b_full_ablation_f1.png`
- `task1_type_ab_full_ablation_f1.png`

完整说明见 `docs/`、`reports/report.md` 和 `reports/runbook.md`。
