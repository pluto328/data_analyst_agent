"""数据集注册：多文件上传后的统一描述与沙箱变量名生成。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

_RESERVED_KEYS: frozenset[str] = frozenset({"df", "pd", "np", "result"})


@dataclass
class DatasetInfo:
    """单张上传表的完整描述。"""

    key: str
    filename: str
    path: str
    preview: dict[str, Any]
    dataframe: pd.DataFrame = field(repr=False)

    def compact_schema(self) -> dict[str, Any]:
        """不含行数据的紧凑 schema，供规划与 LLM 上下文使用。"""
        return {
            "key": self.key,
            "filename": self.filename,
            "shape": self.preview.get("shape"),
            "columns": self.preview.get("columns"),
            "dtypes": self.preview.get("dtypes"),
            "null_counts": self.preview.get("null_counts"),
        }


def sanitize_filename_stem(filename: str) -> str:
    """从文件名生成合法 Python 标识符片段。"""
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    cleaned = re.sub(r"[^\w]", "_", stem, flags=re.UNICODE)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_").lower()
    if not cleaned:
        cleaned = "data"
    if cleaned[0].isdigit():
        cleaned = f"data_{cleaned}"
    return cleaned


def make_dataset_key(filename: str, existing_keys: set[str] | None = None) -> str:
    """生成 ``df_{stem}`` 形式的沙箱变量名，冲突时追加序号。"""
    existing = existing_keys or set()
    base = f"df_{sanitize_filename_stem(filename)}"
    if base in _RESERVED_KEYS:
        base = f"{base}_table"
    candidate = base
    counter = 2
    while candidate in existing or candidate in _RESERVED_KEYS:
        candidate = f"{base}_{counter}"
        counter += 1
    return candidate


def datasets_to_dict(datasets: list[DatasetInfo]) -> dict[str, pd.DataFrame]:
    """将 DatasetInfo 列表转为沙箱注入用的 dict。"""
    return {ds.key: ds.dataframe for ds in datasets}


def primary_dataset(datasets: list[DatasetInfo]) -> DatasetInfo:
    """返回第一个数据集（主表）。"""
    if not datasets:
        raise ValueError("datasets cannot be empty.")
    return datasets[0]


def merge_previews_for_legacy(datasets: list[DatasetInfo]) -> dict[str, Any]:
    """合并多表 preview 为单 dict（兼容旧接口，报告/图表用主表）。"""
    primary = primary_dataset(datasets)
    preview = dict(primary.preview)
    preview["filename"] = primary.filename
    preview["path"] = primary.path
    if len(datasets) > 1:
        preview["dataset_count"] = len(datasets)
        preview["dataset_keys"] = [ds.key for ds in datasets]
        preview["all_datasets"] = [
            {**ds.compact_schema(), "head": ds.preview.get("head")}
            for ds in datasets
        ]
    return preview


__all__ = [
    "DatasetInfo",
    "datasets_to_dict",
    "make_dataset_key",
    "merge_previews_for_legacy",
    "primary_dataset",
    "sanitize_filename_stem",
]
