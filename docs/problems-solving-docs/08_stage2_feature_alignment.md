# 8. Stage2 type_A/type_B 列空间不一致

## 问题背景

Stage2 官方 feather 在 type_A/type_B 间列空间不完全一致。训练缓存由两类文件合并得到，包含完整列集。提交期逐个 feather 预测时，部分 type_B 文件缺少 6h 窗口列。

## 问题表现

生成 best submission 时曾出现缺列错误。模型训练时使用了完整列空间，预测时直接用单个 `target_df[cols]` 会在缺列文件上失败。

## 处理方法

`memory-failure-prediction-agent/src/stage2_feature_pipeline.py` 中加入特征对齐函数。该函数按训练列顺序重排预测期特征，缺失列补 0，多余列丢弃。

```python
def align_prediction_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.reindex(columns=cols, fill_value=0).fillna(0)
```

普通 submission 和 best per-type submission 共用该逻辑。

```python
pred_frame = align_prediction_features(target_df, cols)
score = model.predict_proba(pred_frame)[:, 1]
```

## 结果

`make task2-best-submission` 遍历目标期 feather，并生成 `stage2_best_per_type_prediction_scores.csv` 和 `stage2_best_per_type_submission.csv`。submission 为 200 行，无空值，无重复 `serial_number`。

## 结论

Stage2 官方特征文件的列空间存在类型差异。提交期按训练列显式对齐，保证预测矩阵与训练矩阵一致。
