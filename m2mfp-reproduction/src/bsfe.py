from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def bsfe_feature_names(prefix: str = "bsfe") -> list[str]:
    """生成 BSFE 输出列名。

    M2-MFP 论文 sec4.2 的核心做法是：先把 32 位错误 bit 拆成 8x4 矩阵，
    再分别沿行方向和列方向提取统计量。这里的列名保留 rowwise/columnwise、
    pooling 函数和 bit 统计函数，便于和论文模块逐项对应。
    """

    names: list[str] = []
    for row_column in ["row", "column"]:
        for g_function in ["max_pooling_F", "sum_pooling_F", "F_max_pooling"]:
            for f_function in [
                "bit_count",
                "bit_min_interval",
                "bit_max_interval",
                "bit_max_consecutive_length",
                "bit_consecutive_length",
            ]:
                names.append(f"{prefix}_{row_column}wise_{g_function}_{f_function}")
    return names


@dataclass
class BSFEExtractor:
    """Binary Spatial Feature Extractor，适配自 M2-MFP sec4.2。

    输入是 SmartMem CE 日志中的 `RetryRdErrLogParity`。该字段可解释为 32 位
    bit 图案，本项目按 8 行 x 4 列转换为 DQ-Beat 矩阵，然后复现论文中
    “按行/按列提取高阶空间 bit 特征”的思想。
    """

    row_count: int = 8
    column_count: int = 4
    _row_map: dict[str, tuple[int, int, int, int, int]] = field(init=False)
    _column_map: dict[str, tuple[int, int, int, int, int]] = field(init=False)

    def __post_init__(self) -> None:
        self._row_map = self._binary_string_map(self.column_count)
        self._column_map = self._binary_string_map(self.row_count)

    @staticmethod
    def _features_for_binary_string(binary_string: str) -> tuple[int, int, int, int, int]:
        """对一个二进制字符串提取 5 个 bit 结构特征。

        - bit_count：1 的个数。
        - bit_min_interval：相邻 1 的最小间隔。
        - bit_max_interval：首尾 1 的最大跨度。
        - bit_max_consecutive_length：最长连续 1 的长度。
        - bit_consecutive_length：连续聚集程度的近似计数。
        """

        bit_count = binary_string.count("1")
        bit_min_interval = len(binary_string)
        bit_max_interval = 0
        bit_max_consecutive_length = 0
        bit_consecutive_length = 0
        indices = [idx for idx, char in enumerate(binary_string) if char == "1"]
        if indices:
            bit_max_interval = indices[-1] - indices[0]
            bit_max_consecutive_length = max(len(part) for part in binary_string.split("0"))
            bit_consecutive_length = bit_count - sum(1 for part in binary_string.split("0") if part)
            if len(indices) > 1:
                bit_min_interval = int(np.diff(indices).min())
        return (
            bit_count,
            bit_min_interval,
            bit_max_interval,
            bit_max_consecutive_length,
            bit_consecutive_length,
        )

    @classmethod
    def _binary_string_map(cls, length: int) -> dict[str, tuple[int, int, int, int, int]]:
        """预先缓存所有可能 bit 串的特征，避免逐条日志重复计算。"""

        return {
            bin(value)[2:].zfill(length): cls._features_for_binary_string(bin(value)[2:].zfill(length))
            for value in range(2**length)
        }

    def _row_features(self, rows: list[str]) -> list[int]:
        """沿矩阵行方向聚合，模拟论文中的 row-wise BSFE。"""

        f_rows = [self._row_map[row] for row in rows]
        max_pool = [max(col) for col in zip(*f_rows)]
        sum_pool = [sum(col) for col in zip(*f_rows)]
        aggregate = 0
        for row in rows:
            aggregate |= int(row, 2)
        f_max_pool = list(self._row_map[bin(aggregate)[2:].zfill(self.column_count)])
        return max_pool + sum_pool + f_max_pool

    def _column_features(self, columns: list[str]) -> list[int]:
        """沿矩阵列方向聚合，模拟论文中的 column-wise BSFE。"""

        f_columns = [self._column_map[column] for column in columns]
        max_pool = [max(row) for row in zip(*f_columns)]
        sum_pool = [sum(row) for row in zip(*f_columns)]
        aggregate = 0
        for column in columns:
            aggregate |= int(column, 2)
        f_max_pool = list(self._column_map[bin(aggregate)[2:].zfill(self.row_count)])
        return max_pool + sum_pool + f_max_pool

    def transform_one(self, parity: object) -> list[int]:
        """把单条 RetryRdErrLogParity 转为一组 BSFE 特征。"""

        try:
            value = int(parity)
        except (TypeError, ValueError):
            value = 0
        bits = bin(max(value, 0))[2:].zfill(self.row_count * self.column_count)[-32:]
        rows = [bits[idx : idx + self.column_count] for idx in range(0, len(bits), self.column_count)]
        columns = [bits[idx:: self.column_count] for idx in range(self.column_count)]
        return self._row_features(rows) + self._column_features(columns)

    def transform_series(self, values: pd.Series, prefix: str = "bsfe") -> pd.DataFrame:
        """批量转换一个 Series，返回可直接拼接进特征表的 DataFrame。"""

        return pd.DataFrame([self.transform_one(value) for value in values], columns=bsfe_feature_names(prefix))


def add_bsfe_columns(df: pd.DataFrame, parity_col: str = "RetryRdErrLogParity", prefix: str = "bsfe") -> pd.DataFrame:
    """给原始日志表追加 BSFE 列。

    smoke pipeline 主要通过 time-patch 聚合调用 BSFE；这个函数保留给
    后续消融实验使用，例如比较“原始 CE 统计”与“追加逐条 BSFE 特征”的差异。
    """

    extractor = BSFEExtractor()
    bsfe_df = extractor.transform_series(df[parity_col].fillna(0), prefix=prefix)
    return pd.concat([df.reset_index(drop=True), bsfe_df], axis=1)
