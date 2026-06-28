# 外部验证指南

本文档说明如何在干净环境中验证项目。

## 仅使用仓库内容进行验证

Git 仓库包含代码、配置、文档、指标、图表、submission 示例和压缩后的特征缓存。即使机器上没有原始 feather 数据，也可以验证环境、检查缓存结果，并运行轻量流程。

```bash
uv venv .venv
uv pip install -r requirements.txt
make restore-caches
make compile
make figures
```

执行 `make restore-caches` 后，缓存特征 CSV 会从仓库中提交的 `.csv.gz` 文件恢复。恢复出的 CSV 文件已被 Git 忽略，不会污染版本控制。这个命令的目的，是避免为了做基础验证就必须先下载几十 GB 原始 feather 数据。它只解压仓库中已经提交的缓存产物，因此可以快速验证代码路径和缓存结果；它不会下载原始数据，也不能替代全量 raw-data 重建实验。

可用的仓库内轻量检查命令：

```bash
make -C m2mfp-reproduction task1-ablation
make -C memory-failure-prediction-agent task2-agent-quick
```

## 使用完整原始数据进行验证

全量重建需要 SmartMem/SmartHW 数据集。请将数据放在 Git 仓库之外，并设置 `MEMFAIL_DATA_ROOT`：

```bash
export MEMFAIL_DATA_ROOT=/path/to/data
```

期望的数据目录结构：

```text
data/
├── stage1_feather/
│   ├── ticket.csv
│   ├── type_A/*.feather
│   └── type_B/*.feather
├── stage2_feature/
│   ├── failure_ticket.csv
│   ├── type_A/*.feather
│   └── type_B/*.feather
└── stage2_feather/          # 后续扩展可选
```

全量运行命令：

```bash
make task1-full-a
make task1-full-b
make task1-full-ab
make task2-agent
make task2-best-submission
```

## 数据来源

主要公开入口：

- SmartMem / Codabench 比赛页面：`https://www.codabench.org/competitions/3586/`
- SmartMem 数据集 DOI：`https://zenodo.org/records/15516113`
- Stage2 预提取特征数据集：`https://www.kaggle.com/datasets/smartmem/smartmem-features`
- SmartHW 仓库：`https://github.com/hwcloud-RAS/SmartHW`
- M2-MFP 仓库：`https://github.com/hwcloud-RAS/M2-MFP`

Stage1 原始日志复现使用 `stage1_feather`。Stage2 模型训练和 agent 搜索使用 `stage2_feature`。`stage2_feather` 原始日志作为后续扩展路径保留，默认流程不重新处理这部分 46GB 原始数据。
