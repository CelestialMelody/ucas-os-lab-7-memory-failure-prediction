# 问题解决记录

本文档汇总项目实现中的主要技术问题。详细记录按问题拆分到 `docs/problems-solving-docs/`。每个文档包含问题背景、定位过程、处理方法、代码位置、实验结果和结论。

| 编号 | 问题 | 文档 |
| ---: | --- | --- |
| 1 | 故障预测样本存在时间泄漏风险 | [01_time_leakage.md](problems-solving-docs/01_time_leakage.md) |
| 2 | Stage1 原始 CE 日志字段类型不稳定 | [02_stage1_schema_normalization.md](problems-solving-docs/02_stage1_schema_normalization.md) |
| 3 | `RetryRdErrLogParity` 的 bit 空间特征转换 | [03_retry_parity_bsfe.md](problems-solving-docs/03_retry_parity_bsfe.md) |
| 4 | CE burst 与普通计数的区分 | [04_ce_burst_features.md](problems-solving-docs/04_ce_burst_features.md) |
| 5 | full 特征组合低于 time-point | [05_full_features_vs_time_point.md](problems-solving-docs/05_full_features_vs_time_point.md) |
| 6 | type_B 正样本少，细小 F1 差异不能过度解释 | [06_type_b_positive_scarcity.md](problems-solving-docs/06_type_b_positive_scarcity.md) |
| 7 | Stage2 训练缓存容易被误解为原始数据 | [07_stage2_training_cache.md](problems-solving-docs/07_stage2_training_cache.md) |
| 8 | Stage2 type_A/type_B 列空间不一致 | [08_stage2_feature_alignment.md](problems-solving-docs/08_stage2_feature_alignment.md) |
| 9 | 阈值搜索和目标期分布存在漂移 | [09_threshold_target_shift.md](problems-solving-docs/09_threshold_target_shift.md) |
| 10 | per-type 策略的验证集可比性 | [10_per_type_comparability.md](problems-solving-docs/10_per_type_comparability.md) |
| 11 | 默认 0.5 阈值不适合低正样本比例 | [11_threshold_search_imbalance.md](problems-solving-docs/11_threshold_search_imbalance.md) |
| 12 | 大数据读取、缓存和分层验证 | [12_large_data_validation.md](problems-solving-docs/12_large_data_validation.md) |
| 13 | M2-MFP 公开仓库缺少完整训练流程 | [13_m2mfp_reproduction_boundary.md](problems-solving-docs/13_m2mfp_reproduction_boundary.md) |
| 14 | CPU 默认运行与 GPU 可选加速 | [14_cpu_default_gpu_optional.md](problems-solving-docs/14_cpu_default_gpu_optional.md) |
| 15 | 本地验证、Codabench 和 reproduced F1 不能混用 | [15_metric_scope.md](problems-solving-docs/15_metric_scope.md) |
| 16 | 深度学习模型的样本设计约束 | [16_deep_learning_scope.md](problems-solving-docs/16_deep_learning_scope.md) |
| 17 | 目标期分数整体低于验证阈值 | [17_target_period_low_scores.md](problems-solving-docs/17_target_period_low_scores.md) |
