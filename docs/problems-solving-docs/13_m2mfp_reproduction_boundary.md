# 13. M2-MFP 公开仓库缺少完整训练流程

## 问题背景

公开 M2-MFP 仓库提供 BSFE、time-patch、time-point 等模块代码。仓库内容覆盖特征机制，但没有提供完整的数据处理、训练、验证和论文表格复现实验流程。

## 处理方法

任务一的目标限定为特征机制复现和本地检验。实现范围包括：

- 适配 32 位 retry parity 到 8x4 DQ-Beat 矩阵。
- 实现 BSFE、time-patch、time-point 特征。
- 使用 Logistic Regression 和轻量 RandomForest 做本地评估。
- 用消融实验比较特征组贡献。

代码入口为 `m2mfp-reproduction/src/ablation.py`。该入口先生成包含全部特征组的缓存表，再按消融配置筛选特征列。

```python
full_config = replace(config, feature_groups=("baseline", "bsfe", "time_patch", "time_point"))
full_features = build_raw_smoke_features(full_config, max_files)

for name, groups in ABLATIONS.items():
    features = filter_feature_groups(full_features, groups)
    metrics = evaluate(features, replace(config, feature_groups=groups)).copy()
```

## 结果

任务一产出的是公开模块条件下的本地特征检验结果。最终报告引用 baseline、BSFE、time-patch、time-point 和 full 消融表，不把这些结果等同于论文私有训练配置。

## 结论

任务一复现 M2-MFP 风格特征机制，并在 Stage1 原始日志上完成本地消融检验。实验边界由公开代码可用内容和本项目评估器共同决定。
