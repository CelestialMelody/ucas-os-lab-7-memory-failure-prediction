# 12. 大数据读取慢，采用缓存和分层验证

## 问题背景

Stage1 原始日志约 33GB，Stage2 官方特征约 7.9GB，Stage2 原始日志约 46GB。每次修改后直接运行 full command 会消耗较长时间，也会拉长问题定位周期。

## 处理方法

项目采用分层验证流程。

1. `make compile` 做语法检查。
2. `make task1-smoke MAX_FILES=300` 做任务一小规模检查。
3. `make task2-agent-quick` 做任务二策略入口检查。
4. full run 只在结果更新时运行。
5. 提交压缩特征缓存 `.csv.gz`，避免公开仓库依赖原始 feather。

Makefile 将快速检查、图表生成和长任务拆成不同目标。这样公开项目在缺少 GB 级原始数据时仍能完成语法检查、图表复现和报告材料生成。

```makefile
compile:
	uv run python -m compileall m2mfp-reproduction memory-failure-prediction-agent scripts

figures:
	uv run python scripts/make_figures.py

task1-smoke:
	$(MAKE) -C m2mfp-reproduction smoke MAX_FILES=$(MAX_FILES)

task2-agent-quick:
	$(MAKE) -C memory-failure-prediction-agent agent-quick
```

## 结果

收口阶段先验证代码和图表，再运行长任务更新结果。公开仓库在无原始数据环境下仍能检查已有结果。

## 结论

GB 级原始数据不纳入 Git。轻量缓存用于复查结果和运行快速检查。
