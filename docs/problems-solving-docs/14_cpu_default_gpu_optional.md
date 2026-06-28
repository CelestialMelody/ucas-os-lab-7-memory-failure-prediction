# 14. CPU 默认运行与 GPU 可选加速

## 问题背景

本机可能有 GPU，但运行环境、助教机器或 CI 环境不一定能访问 CUDA。默认启用 GPU 会让复现结果依赖硬件和驱动状态。

## 处理方法

主流程采用 CPU 可运行配置：

- 任务一使用 Logistic Regression 和轻量 RandomForest。
- 任务二使用 RF、HGB、LightGBM、XGBoost。
- XGBoost GPU 参数只作为扩展加速方向。

模型构造中没有写入 CUDA 依赖参数。

```python
models["xgboost"] = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    n_jobs=-1,
    random_state=random_state,
)
```

## 结果

项目主要验证命令在普通 CPU 环境下运行。GPU 仅对应长任务加速方案。

## 结论

默认复现命令不依赖 CUDA。该设计降低了运行环境差异对课程交付的影响。
