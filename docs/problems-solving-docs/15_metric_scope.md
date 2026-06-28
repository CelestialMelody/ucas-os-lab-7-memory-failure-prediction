# 15. 本地验证、Codabench 和 reproduced F1 的评估范围

## 问题背景

SmartHW winners 资料中同时出现 Codabench F1 和 reproduced F1。本项目产生的是本地验证 F1 和离线 submission。三类分数来自不同数据切分和评估入口。

## 处理方法

文档中将外部 winners 表标为 reference。任务二结果表只使用本地验证指标：

- baseline XGBoost：F1 = 0.5405。
- agent `xgb_all_none_per_type`：F1 = 0.5538。

submission 部分只描述格式、行数、时间戳范围、空值和重复值检查。

## 结果

项目结果分为两类：本地验证指标和离线候选文件。本地验证指标用于比较项目内部策略。离线候选文件用于展示目标期候选生成和提交格式。

## 结论

本地验证 F1、Codabench hidden-test F1 和 reproduced F1 属于不同评估范围。外部 winners 图表用于背景参照，不用于项目排名结论。
