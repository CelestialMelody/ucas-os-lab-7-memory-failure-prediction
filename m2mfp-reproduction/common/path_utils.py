from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def resolve_path(raw_path: str | Path, base: Path, prefix: Path | None = None) -> Path:
    """解析配置路径。

    - 绝对路径保持不变。
    - 相对路径优先拼到 prefix。
    - 没有 prefix 时拼到配置文件所在任务根目录。
    """

    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    if prefix is not None:
        parts = path.parts
        if "data" in parts:
            data_idx = parts.index("data")
            suffix = Path(*parts[data_idx + 1 :])
            return (prefix / suffix).resolve()
        return (prefix / path).resolve()
    root = prefix if prefix is not None else base
    return (root / path).resolve()


def config_path_prefix(raw: dict[str, Any], base: Path, env_name: str = "MEMFAIL_DATA_ROOT") -> Path | None:
    """读取数据路径 prefix。

    环境变量优先，便于换机器时不修改 JSON；否则使用 JSON 中的 `path_prefix`。
    """

    env_value = os.environ.get(env_name)
    if env_value:
        return resolve_path(env_value, base)
    if raw.get("path_prefix"):
        return resolve_path(raw["path_prefix"], base)
    return None
