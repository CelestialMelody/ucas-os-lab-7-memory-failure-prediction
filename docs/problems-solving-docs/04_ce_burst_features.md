# 4. CE burst 与普通计数的区分

## 问题背景

CE 总数只能说明某段时间内日志数量。故障前更有区分度的现象往往是短时间密集出现 CE。若只使用窗口总数，长时间均匀发生和短时间集中爆发可能得到相似的计数。

## 定位过程

M2-MFP time-point 思路强调故障前时点异常。SmartHW/SmartMem 的故障预测任务也关注未来短期风险。baseline 和 time-point 都需要加入局部 burst 描述。

## 处理方法

baseline 和 time-patch 使用同一类 CE storm 规则。它遍历按时间排序的 CE，如果相邻事件间隔小于阈值，则延长当前连续链。

```python
def ce_storm_count(times: pd.Series, interval_seconds: int = 60, threshold: int = 10) -> int:
    values = sorted(int(value) for value in times)
    storms = 0
    consecutive = 0
    for prev, cur in zip(values, values[1:]):
        if cur - prev <= interval_seconds:
            consecutive += 1
        else:
            consecutive = 0
        if consecutive > threshold:
            storms += 1
            consecutive = 0
    return storms
```

time-point 进一步记录短窗口内最长连续链。

```python
features.update(
    {
        "m2point_max_run_60s": float(_max_run_count(recent_times, 60)),
        "m2point_max_run_300s": float(_max_run_count(recent_times, 300)),
    }
)
```

## 结果

time-point 特征在任务一 full run 中表现最强。A+B full 的 `baseline_time_point + random_forest_small` F1 = 0.7645。

## 结论

CE burst 描述短时间密集错误，和普通窗口计数不同。time-point 的实验结果显示故障前突发和空间集中具有预测信号。
