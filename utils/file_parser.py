"""表格文件解析：CSV / Excel 读取、编码探测、数据预览。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agent.dataset_registry import DatasetInfo, make_dataset_key
from config.settings import (
    CSV_ENCODINGS,
    CSV_SUFFIX,
    MAX_UPLOAD_BYTES,
    XLS_SUFFIX,
    XLSX_SUFFIX,
)
from utils.logger import get_logger
from utils.path_helper import is_allowed_upload, to_path

log = get_logger()


def check_file_size(path: str | Path) -> int:
    """校验文件大小不超过 ``MAX_UPLOAD_BYTES``，返回字节数。"""
    file_path = to_path(path)
    try:
        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")
        size = file_path.stat().st_size
    except OSError as exc:
        raise OSError(f"Cannot read file size: {file_path}") from exc

    if size > MAX_UPLOAD_BYTES:
        limit_mb = MAX_UPLOAD_BYTES / (1024 * 1024)
        raise ValueError(
            f"File too large: {size} bytes. Limit is {limit_mb:.0f} MB."
        )
    return size


def read_csv_file(path: str | Path) -> pd.DataFrame:
    """依次尝试 utf-8 / gbk / gb2312 读取 CSV。"""
    file_path = to_path(path)
    check_file_size(file_path)
    last_error: Exception | None = None

    for encoding in CSV_ENCODINGS:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            log.info("CSV loaded with encoding {}: {}", encoding, file_path.name)
            return df
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except Exception as exc:
            raise ValueError(f"Failed to read CSV: {file_path}") from exc

    raise ValueError(
        f"Cannot decode CSV with encodings {CSV_ENCODINGS}: {file_path}"
    ) from last_error


def read_excel_file(path: str | Path, *, sheet_name: str | int = 0) -> pd.DataFrame:
    """读取 Excel：``.xls`` 用 xlrd，``.xlsx`` 用 openpyxl。"""
    file_path = to_path(path)
    check_file_size(file_path)
    suffix = file_path.suffix.lower()

    if suffix == XLS_SUFFIX:
        engine = "xlrd"
    elif suffix == XLSX_SUFFIX:
        engine = "openpyxl"
    else:
        raise ValueError(f"Unsupported Excel suffix: {suffix!r}")

    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, engine=engine)
        log.info("Excel loaded with engine {}: {}", engine, file_path.name)
        return df
    except Exception as exc:
        raise ValueError(f"Failed to read Excel: {file_path}") from exc


def read_table_file(path: str | Path, *, sheet_name: str | int = 0) -> pd.DataFrame:
    """按后缀自动选择 CSV 或 Excel 读取方式。"""
    file_path = to_path(path)
    if not is_allowed_upload(file_path.name):
        raise ValueError(f"Unsupported upload file: {file_path.name}")

    suffix = file_path.suffix.lower()
    if suffix == CSV_SUFFIX:
        return read_csv_file(file_path)
    if suffix in {XLS_SUFFIX, XLSX_SUFFIX}:
        return read_excel_file(file_path, sheet_name=sheet_name)
    raise ValueError(f"Unsupported file type: {suffix!r}")


def _json_safe(value: Any) -> Any:
    """将预览值转为 JSON 友好格式。"""
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def preview_dataframe(df: pd.DataFrame, *, head_rows: int = 5) -> dict[str, Any]:
    """生成供 Agent 使用的表格预览摘要。"""
    if head_rows < 1:
        raise ValueError("head_rows must be >= 1")

    preview_df = df.head(head_rows).copy()
    records: list[dict[str, Any]] = []
    for row in preview_df.to_dict(orient="records"):
        records.append({key: _json_safe(val) for key, val in row.items()})

    columns = [
        {"name": str(col), "dtype": str(dtype)}
        for col, dtype in df.dtypes.items()
    ]
    null_counts = {
        str(col): int(count) for col, count in df.isna().sum().items()
    }

    return {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": columns,
        "dtypes": {str(col): str(dtype) for col, dtype in df.dtypes.items()},
        "head": records,
        "null_counts": null_counts,
    }


def parse_uploaded_file(
    path: str | Path,
    *,
    sheet_name: str | int = 0,
    head_rows: int = 5,
) -> dict[str, Any]:
    """读取上传文件并返回预览摘要（含 ``dataframe`` 键）。"""
    file_path = to_path(path)
    try:
        df = read_table_file(file_path, sheet_name=sheet_name)
        preview = preview_dataframe(df, head_rows=head_rows)
    except Exception:
        log.exception("Failed to parse uploaded file: {}", file_path)
        raise

    return {
        "path": str(file_path),
        "filename": file_path.name,
        "file_size": check_file_size(file_path),
        **preview,
        "dataframe": df,
    }


def parse_uploaded_files(
    paths: list[str | Path],
    *,
    original_filenames: list[str] | None = None,
    sheet_name: str | int = 0,
    head_rows: int = 5,
) -> list[DatasetInfo]:
    """批量解析上传文件，为每张表分配唯一沙箱变量名。"""
    if not paths:
        raise ValueError("paths cannot be empty.")
    if original_filenames is not None and len(original_filenames) != len(paths):
        raise ValueError("original_filenames length must match paths.")

    datasets: list[DatasetInfo] = []
    existing_keys: set[str] = set()

    for index, path in enumerate(paths):
        parsed = parse_uploaded_file(path, sheet_name=sheet_name, head_rows=head_rows)
        display_name = (
            original_filenames[index]
            if original_filenames is not None
            else parsed["filename"]
        )
        key = make_dataset_key(index + 1, existing_keys)
        existing_keys.add(key)
        preview = {k: v for k, v in parsed.items() if k != "dataframe"}
        preview["filename"] = display_name
        datasets.append(
            DatasetInfo(
                key=key,
                filename=display_name,
                path=parsed["path"],
                preview=preview,
                dataframe=parsed["dataframe"],
            )
        )

    log.info("Parsed {} dataset(s): {}", len(datasets), [ds.key for ds in datasets])
    return datasets


def _run_self_check() -> None:
    """模块内置自检，便于 ``python -m utils.file_parser`` 调试。"""
    from utils.path_helper import OUTPUT_SUBDIR, build_temp_file_path, delete_path

    print("=== file_parser self-check ===")

    csv_path = build_temp_file_path(OUTPUT_SUBDIR, "self_check.csv", prefix="parser")
    try:
        csv_path.write_text(
            "name,amount,date\nAlice,100,2024-01-01\nBob,200,2024-01-02\n",
            encoding="utf-8",
        )
        print(f"sample csv: {csv_path}")

        parsed = parse_uploaded_file(csv_path, head_rows=2)
        print(f"filename:   {parsed['filename']}")
        print(f"shape:      {parsed['shape']}")
        print(f"columns:    {parsed['columns']}")
        print(f"head:       {parsed['head']}")
        print(f"null_counts:{parsed['null_counts']}")
        print(f"df rows:    {len(parsed['dataframe'])}")
    except Exception as exc:
        print(f"self-check failed: {exc}")
        raise
    finally:
        try:
            delete_path(csv_path)
        except OSError as exc:
            print(f"cleanup failed: {exc}")

    print("=== done ===")


__all__ = [
    "check_file_size",
    "parse_uploaded_file",
    "parse_uploaded_files",
    "preview_dataframe",
    "read_csv_file",
    "read_excel_file",
    "read_table_file",
]


if __name__ == "__main__":
    _run_self_check()
