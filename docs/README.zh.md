# OS 实验七：内存故障预测

本仓库是 UCAS OS 课程实验七“内存故障预测”的课程项目整理版。项目被拆分为两个可复现的子项目：

- `m2mfp-reproduction/`：在 SmartMem Stage1 CE 原始日志上复现并检验 M2-MFP 风格的 raw-log 特征。
- `memory-failure-prediction-agent/`：在 SmartHW/SmartMem Stage2 官方预提取特征上训练模型、搜索 agent 策略，并生成离线候选提交文件。

仓库不包含 GB 级原始 feather 数据集。仓库中保留代码、配置、报告、图表、轻量指标结果和压缩后的中间特征缓存，使得不立即重新读取全部原始 feather 文件也能检查实验流程和结果。

## 快速开始

在仓库根目录使用 `uv` 安装依赖：

```bash
uv venv .venv
uv pip install -r requirements.txt
```

恢复仓库中提交的压缩特征缓存：

```bash
make restore-caches
```

`make restore-caches` 会把仓库中提交的 `.csv.gz` 缓存文件解压成轻量 CSV，供快速检查命令使用。它的作用是：在没有下载几十 GB 原始 feather 日志的情况下，也能查看已报告的指标并运行轻量验证流程。它不会下载原始数据，也不能替代全量 raw-data 实验。

运行静态编译检查：

```bash
make compile
```

运行快速 smoke 检查：

```bash
make task1-smoke MAX_FILES=300
make task2-agent-quick
```

## 文档入口

- `report.md`：研究报告正文，包含背景、M2-MFP 论文讲解、两个任务实现、结果和复现命令。
- `problem_solving_record.md`：公开版问题解决记录导览。
- `problems-solving-docs/`：一题一文档的问题解决记录，记录样本构造、特征实现、阈值、per-type、submission 等技术问题。
- `external_validation.md` / `external_validation.zh.md`：干净环境验证说明。

如果要运行全量实验，请把数据放在 Git 仓库之外，并将 `MEMFAIL_DATA_ROOT` 指向包含 `stage1_feather/`、`stage2_feature/` 和可选 `stage2_feather/` 的数据目录：

```bash
make task1-full-ab MEMFAIL_DATA_ROOT=/path/to/data
make task2-agent MEMFAIL_DATA_ROOT=/path/to/data
```

## 使用的数据

实验使用的本地数据目录如下：

| 数据目录                |      规模 | 用途                                    |
| ----------------------- | --------: | --------------------------------------- |
| `data/stage1_feather` |  约 33 GB | M2-MFP 原始日志复现                     |
| `data/stage2_feature` | 约 7.9 GB | Stage2 模型训练和 agent 搜索            |
| `data/stage2_feather` |  约 46 GB | 后续 raw-log 增强，可选；默认流程不使用 |

原始数据应从 SmartMem/SmartHW 发布源或比赛镜像下载。每个子项目的 `docs/data.md` 中记录了对应的数据说明。

## 结果摘要

任务一最佳结果：

| 数据集               | 最佳消融            | 模型                |     F1 |
| -------------------- | ------------------- | ------------------- | -----: |
| type_A smoke         | baseline_time_point | random_forest_small | 0.7270 |
| type_A full          | baseline_time_point | random_forest_small | 0.7587 |
| type_B full          | baseline_time_point | random_forest_small | 0.6479 |
| type_A + type_B full | baseline_time_point | random_forest_small | 0.7645 |

任务二最佳结果：

| 方法                           | 本地验证 F1 |
| ------------------------------ | ----------: |
| baseline XGBoost               |      0.5405 |
| agent `xgb_all_none_per_type` |      0.5538 |

Stage2 submission 文件是离线候选输出。当前 Codabench 评估入口关闭，项目没有 hidden-test F1。任务二效果结论以本地验证 F1 为准，winners 图表仅作为外部背景。

## 仓库结构

```text
oslab-7-memory-failure-prediction/
├── m2mfp-reproduction/
├── memory-failure-prediction-agent/
├── docs/
├── requirements.txt
├── Makefile
└── README.md
```

每个子项目都包含自己的 `README.md`、`docs/`、`configs/`、`src/`、`results/`、`reports/` 和 `Makefile`。
