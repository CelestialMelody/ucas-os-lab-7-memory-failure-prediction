# 7. Stage2 训练缓存容易被误解为原始数据

## 问题背景

`stage2_train_features.csv` 约 7.2 万行，远小于 Stage2 官方特征目录。agent 搜索运行较快，容易被误读为只使用一个人工整理的小 CSV。

## 定位过程

任务二的输入包括 `stage2_feature/type_A/*.feather`、`stage2_feature/type_B/*.feather` 和 `failure_ticket.csv`。CSV 由代码构造，是监督学习缓存表。

训练缓存由 `build_training_table()` 生成。该函数遍历官方 feather 文件，把每根 DIMM 的时间序列特征压成 1 到 2 条监督学习样本。

```python
for path, sn_type in iter_feature_files(config.feature_root):
    sn_name = path.stem
    df = pd.read_feather(path).sort_values("LogTime").reset_index(drop=True)
    if sn_name in ticket_map:
        alarm_time, ticket_type = ticket_map[sn_name]
        pos_start = alarm_time - lead - pred
        pos_end = alarm_time - lead
        pos_df = df[(df["LogTime"] >= pos_start) & (df["LogTime"] <= pos_end)]
        if not pos_df.empty:
            rows.append(add_meta(pos_df.iloc[-1], sn_name, sn_type, 1))
        neg_df = df[df["LogTime"] < pos_start - ONE_DAY]
        if not neg_df.empty:
            rows.append(add_meta(neg_df.iloc[-1], sn_name, sn_type, 0))
    else:
        rows.append(add_meta(df.iloc[-1], sn_name, sn_type, 0))
```

## 处理方法

文档中明确数据流：

```text
stage2_feature/type_A/*.feather
stage2_feature/type_B/*.feather
failure_ticket.csv
  -> build_training_table()
  -> stage2_train_features.csv
  -> model / threshold / agent search
```

缓存表保存正样本、故障 DIMM 负样本和无故障 DIMM 负样本。它用于加速复查和策略搜索。

## 结果

任务二通过缓存快速复查模型、阈值和 agent 策略。缓存重建入口为 `make task2-train`。

## 结论

Stage2 train CSV 是由官方 feather 特征构造出的监督学习缓存表。它保存样本构造结果，服务于模型训练、阈值搜索和 agent 搜索。
