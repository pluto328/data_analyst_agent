"""沙箱 result 解析为具名输出表（支持多表、自动命名与数量上限）。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config.settings import MAX_OUTPUT_TABLES
from utils.logger import get_logger
from utils.path_helper import sanitize_filename

log = get_logger()

_INVALID_LABEL_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_MULTI_UNDERSCORE = re.compile(r"_+")


@dataclass(frozen=True)
class ResultTable:
    """单张具名输出表。"""

    name: str
    filename: str
    dataframe: pd.DataFrame


def sanitize_table_label(label: str, *, fallback: str = "分析结果") -> str:
    """清洗表名/文件名标签（保留中文）。"""
    text = str(label or "").strip()
    if not text:
        text = fallback
    cleaned = _INVALID_LABEL_CHARS.sub("_", text)
    cleaned = _MULTI_UNDERSCORE.sub("_", cleaned).strip("._ ")
    if not cleaned:
        cleaned = fallback
    return cleaned[:80]


def _to_dataframe(value: object) -> pd.DataFrame | None:
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, pd.Series):
        return value.to_frame()
    return None


def _make_unique_filenames(tables: list[ResultTable]) -> list[ResultTable]:
    seen: set[str] = set()
    unique: list[ResultTable] = []
    for table in tables:
        base_stem = Path(table.filename).stem if "." in table.filename else table.filename
        candidate = sanitize_filename(f"{base_stem}.csv")
        stem = Path(candidate).stem
        suffix = Path(candidate).suffix or ".csv"
        final_name = candidate
        counter = 2
        while final_name in seen:
            final_name = sanitize_filename(f"{stem}_{counter}{suffix}")
            counter += 1
        seen.add(final_name)
        unique.append(
            ResultTable(name=table.name, filename=final_name, dataframe=table.dataframe)
        )
    return unique


def extract_result_tables(
    sandbox_result,
    *,
    max_tables: int | None = None,
) -> tuple[list[ResultTable], bool]:
    """从沙箱结果提取具名 DataFrame 列表。

    返回 ``(tables, truncated)``；``truncated`` 表示是否因上限被截断。
    """
    limit = max_tables if max_tables is not None else MAX_OUTPUT_TABLES
    limit = max(1, limit)

    if sandbox_result is None or not getattr(sandbox_result, "success", False):
        return [], False

    raw_tables: list[ResultTable] = []
    result = getattr(sandbox_result, "result", None)

    if isinstance(result, dict):
        for index, (key, value) in enumerate(result.items(), start=1):
            frame = _to_dataframe(value)
            if frame is None or frame.empty:
                continue
            label = sanitize_table_label(str(key), fallback=f"结果表{index}")
            filename = sanitize_filename(f"{label}.csv")
            raw_tables.append(ResultTable(name=label, filename=filename, dataframe=frame))
    elif isinstance(result, (pd.DataFrame, pd.Series)):
        frame = _to_dataframe(result)
        if frame is not None and not frame.empty:
            label = sanitize_table_label("分析结果")
            raw_tables.append(
                ResultTable(
                    name=label,
                    filename=sanitize_filename(f"{label}.csv"),
                    dataframe=frame,
                )
            )
    elif isinstance(result, list):
        for index, item in enumerate(result, start=1):
            frame = _to_dataframe(item)
            if frame is None or frame.empty:
                continue
            label = sanitize_table_label(f"结果表{index}")
            raw_tables.append(
                ResultTable(
                    name=label,
                    filename=sanitize_filename(f"{label}.csv"),
                    dataframe=frame,
                )
            )

    sandbox_df = getattr(sandbox_result, "df", None)
    if (
        not raw_tables
        and isinstance(sandbox_df, pd.DataFrame)
        and not sandbox_df.empty
    ):
        label = sanitize_table_label("输出数据表")
        raw_tables.append(
            ResultTable(
                name=label,
                filename=sanitize_filename(f"{label}.csv"),
                dataframe=sandbox_df,
            )
        )

    if not raw_tables:
        return [], False

    raw_tables = _make_unique_filenames(raw_tables)
    truncated = len(raw_tables) > limit
    if truncated:
        log.warning(
            "Result tables truncated from {} to {} (MAX_OUTPUT_TABLES)",
            len(raw_tables),
            limit,
        )
    return raw_tables[:limit], truncated


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """转为 CSV 字节流（utf-8-sig，便于 Excel 打开）。"""
    try:
        return df.to_csv(index=False).encode("utf-8-sig")
    except Exception as exc:
        raise ValueError("Failed to serialize DataFrame to CSV.") from exc


def tables_to_serializable(
    tables: list[ResultTable],
) -> list[dict[str, object]]:
    """转为 Streamlit session 可存储的结构。"""
    payload: list[dict[str, object]] = []
    for table in tables:
        try:
            payload.append(
                {
                    "name": table.name,
                    "filename": table.filename,
                    "csv_bytes": dataframe_to_csv_bytes(table.dataframe),
                    "shape": list(table.dataframe.shape),
                }
            )
        except ValueError as exc:
            log.warning("Skip table {} export: {}", table.name, exc)
    return payload


def primary_result_dataframe(tables: list[ResultTable]) -> pd.DataFrame | None:
    """返回第一张输出表，供绘图等默认使用。"""
    if not tables:
        return None
    return tables[0].dataframe


def _run_self_check() -> None:
    print("=== result_tables self-check ===")
    from sandbox.code_sandbox import SandboxResult

    single = SandboxResult(
        success=True,
        result=pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
    )
    tables, truncated = extract_result_tables(single, max_tables=5)
    assert len(tables) == 1 and tables[0].name == "分析结果"
    print(f"single table: {tables[0].filename}")

    multi = SandboxResult(
        success=True,
        result={
            "品类汇总": pd.DataFrame({"cat": ["A"], "amt": [10]}),
            "清洗明细": pd.DataFrame({"cat": ["A", "B"], "amt": [1, 2]}),
        },
    )
    tables, truncated = extract_result_tables(multi, max_tables=5)
    assert len(tables) == 2
    print(f"multi tables: {[t.name for t in tables]}")

    capped, truncated = extract_result_tables(multi, max_tables=1)
    assert len(capped) == 1 and truncated is True
    print("truncate: ok")
    print("=== done ===")


__all__ = [
    "ResultTable",
    "dataframe_to_csv_bytes",
    "extract_result_tables",
    "primary_result_dataframe",
    "sanitize_table_label",
    "tables_to_serializable",
]


if __name__ == "__main__":
    _run_self_check()
