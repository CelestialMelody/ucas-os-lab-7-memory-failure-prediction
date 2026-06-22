# PPT 大纲

## 1. 标题页

题目：Memory Failure Prediction with M2-MFP Reproduction and SmartHW Agent Optimization

说明课程、实验七、组员和日期。

## 2. 问题定义

- 输入：CE 日志、DIMM 序列号、rank/device/bank/row/column、retry parity。
- 输出：未来预测窗口内可能故障的 DIMM 和预测时间戳。
- 难点：类别极不平衡、日志时间跨度长、故障前信号稀疏、不同 DIMM 类型分布不同。

## 3. 数据说明

- Stage1 原始 feather：用于 M2-MFP 复现检验。
- Stage2 官方特征：用于比赛式训练和提交。
- Stage2 原始 feather：作为后续增强数据。
- 展示数据规模和正负样本比例图，图中文字先用英语。

## 4. M2-MFP 方法复现

- BSFE：把 `RetryRdErrLogParity` 转成 8x4 DQ-Beat bit 矩阵。
- Time-patch：15m/1h/6h 多尺度时间窗口聚合。
- Time-point：CE storm 和空间聚集规则。
- 说明公开代码只提供核心模块，本项目做本地适配。

## 5. 任务一实验结果

- 展示 raw-log smoke 指标。
- 展示 baseline / BSFE / time-patch / time-point 消融。
- 300-file smoke 最好结果：baseline + time-point，RandomForest F1 = 0.7270。
- Stage1 type_A 全量最好结果：baseline + time-point，RandomForest F1 = 0.7587。
- type_A 全量完整特征组 F1 = 0.6455；time-point 规则在本地适配中最稳定，BSFE/time-patch 与本项目模型配置的叠加收益弱于单独 time-point。
- Stage1 type_B 全量最好结果：baseline + time-point，RandomForest F1 = 0.6479。
- Stage1 A+B 全量最好结果：baseline + time-point，RandomForest F1 = 0.7645。
- type_B 全量上 BSFE/time-patch/time-point 均超过 baseline，但正样本只有 68 个，需谨慎解释。
- 任务一使用 Logistic Regression 与轻量 RandomForest 做本地评估；公开 M2-MFP 代码包含自定义 decision tree/time-point 模块，但不提供完整论文训练流水线，因此不声称复现完全相同分类器配置。

## 6. SmartHW/SmartMem 预测系统

- 读取 Stage2 预提取特征。
- 构造正负训练样本。
- 时间切分验证。
- 训练 RF / HistGB / LightGBM / XGBoost。
- 阈值搜索和提交生成。
- 模型来源：SmartHW starter kit 演示 XGBoost，官方 baseline 使用 LightGBM；RF/HistGB 是本项目加入的对照模型。

## 7. Agent 自动实验优化

- 策略库：采样、模型、阈值、top-k、per-type、特征组合。
- 实验登记表：记录配置、指标、输出路径。
- 消融实验：判断哪些策略真正提升 F1/Recall。
- 默认 dry-run 只生成登记表，长训练由运行手册命令控制。

## 8. 任务二实验结果

- 展示模型对比图。
- 展示 threshold-F1 曲线。
- 展示 submission 生成结果。
- 说明 baseline 最好结果：XGBoost F1 = 0.5405。
- 说明 agent 最好结果：xgb_all_none_per_type，F1 = 0.5538。
- 展示最佳提交分数分布：200 行、无重复序列号，目标期阈值未命中后使用 top-k 兜底。
- 说明 submission 是离线候选告警文件；本地验证 F1、Codabench 隐藏测试 F1 和 winners reproduced F1 属于不同评估口径。

## 9. 问题与解决

- 大数据读取慢：使用 feather、缓存中间特征、先 smoke 后 full run。
- 论文仓库不完整：明确复现边界，复现核心模块和结论。
- 类别不平衡：class weight、阈值搜索、top-k、hard negative。
- GPU 不稳定或不可用：默认 CPU，可选 XGBoost CUDA。

## 10. 总结

- 完成 M2-MFP BSFE、time-patch、time-point 本地适配和 smoke 检验。
- 完成 SmartHW Stage2 训练、验证、agent 搜索和最佳 per-type 提交生成。
- 任务一 type_A smoke 最好 F1 = 0.7270，type_A full 最好 F1 = 0.7587，type_B full 最好 F1 = 0.6479，A+B full 最好 F1 = 0.7645，任务二本地验证最好 F1 = 0.5538。
- 报告、运行手册、图表和答辩材料完整；可继续扩展模型调参与 Stage2 原始日志增强。
