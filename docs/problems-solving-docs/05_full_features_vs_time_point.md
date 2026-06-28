# 5. full 特征组合低于 time-point

## 问题背景

full 特征包含 baseline、BSFE、time-patch 和 time-point。任务一 full run 显示 full 组合低于 `baseline_time_point`。该现象直接影响任务一结论：特征机制检验和最终最佳消融结果必须分开记录。

## 现象

实验结果如下：

| Dataset | baseline_time_point F1 | full F1 |
| --- | ---: | ---: |
| type_A full | 0.7587 | 0.6455 |
| type_B full | 0.6479 | 0.6471 |
| A+B full | 0.7645 | 0.6358 |

## 分析

time-point 特征维度少，直接描述最近 CE、短长窗口比例、连续链和空间集中。BSFE/time-patch 引入大量窗口和 bit 特征，使 full 特征数提高到数百列。故障正样本仍然稀少，高维拼接会增加噪声和过拟合风险。

任务一消融入口先生成包含全部候选列的缓存表，再按前缀筛选特征组。这样 baseline、time-point 和 full 共享同一批样本。

```python
ABLATIONS = {
    "baseline": ("baseline",),
    "baseline_time_point": ("baseline", "time_point"),
    "baseline_bsfe_time_patch_time_point": (
        "baseline",
        "bsfe",
        "time_patch",
        "time_point",
    ),
}

full_features = build_raw_smoke_features(full_config, max_files)
features = filter_feature_groups(full_features, groups)
metrics = evaluate(features, replace(config, feature_groups=groups))
```

`filter_feature_groups()` 根据列名前缀切出特征。full 组保留所有前缀，`baseline_time_point` 只保留基础统计和预测点附近特征。二者差异来自特征列数量和特征类型。

## 处理方法

结果表中同时保留 full 和 `baseline_time_point`。结论区分两件事：BSFE/time-patch 存在增量信号；在本地 RandomForest 配置下，直接拼接全部特征没有取得最佳 F1。后续扩展需要特征选择、正则化或更大正样本规模支撑。

## 结果

任务一最终结论采用 `baseline_time_point` 作为最佳消融结果，同时保留 BSFE/time-patch 的解释价值。

## 结论

full 组合属于机制消融。任务一最佳消融结果为 `baseline_time_point`。M2-MFP 风格高维特征需要配合模型容量、正则化和样本规模。
