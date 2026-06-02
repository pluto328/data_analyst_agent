"""跨平台路径工具与 temp_files 目录管理。"""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from pathlib import Path

from config.settings import (
    ALLOWED_UPLOAD_SUFFIXES,
    PROJECT_ROOT,
    TEMP_DIR,
    ensure_temp_dir,
)

UPLOAD_SUBDIR: str = "uploads"
CHART_SUBDIR: str = "charts"
OUTPUT_SUBDIR: str = "outputs"

_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_MULTI_UNDERSCORE = re.compile(r"_+")


def to_path(value: str | Path) -> Path:
    """将 str / Path 统一为绝对 Path。"""
    try:
        return Path(value).expanduser().resolve()
    except (TypeError, OSError) as exc:
        raise ValueError(f"Invalid path: {value!r}") from exc


def get_project_root() -> Path:
    """返回项目根目录。"""
    return PROJECT_ROOT


def get_temp_dir() -> Path:
    """返回 temp_files 目录；不存在则创建。"""
    try:
        return ensure_temp_dir()
    except OSError as exc:
        raise OSError(f"Cannot create temp directory: {TEMP_DIR}") from exc


def ensure_subdirectory(name: str) -> Path:
    """在 temp_files 下创建并返回子目录。"""
    safe_name = _sanitize_dir_name(name)
    subdir = get_temp_dir() / safe_name
    try:
        subdir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(f"Cannot create subdirectory: {safe_name}") from exc
    return subdir


def sanitize_filename(filename: str) -> str:
    """去除路径成分与非法字符，保留扩展名。"""
    if not filename or not str(filename).strip():
        raise ValueError("Filename cannot be empty.")

    name = Path(str(filename).strip()).name
    if not name or name in {".", ".."}:
        raise ValueError(f"Invalid filename: {filename!r}")

    stem = Path(name).stem
    suffix = Path(name).suffix.lower()
    safe_stem = _INVALID_FILENAME_CHARS.sub("_", stem)
    safe_stem = _MULTI_UNDERSCORE.sub("_", safe_stem).strip("._")
    if not safe_stem:
        safe_stem = "file"

    return f"{safe_stem}{suffix}"


def resolve_under_base(path: str | Path, base: str | Path) -> Path:
    """解析路径并校验其位于 base 目录内，防止路径穿越。"""
    base_path = to_path(base)
    candidate = Path(path)
    resolved = candidate if candidate.is_absolute() else (base_path / candidate)
    try:
        resolved = resolved.resolve()
        resolved.relative_to(base_path)
    except ValueError as exc:
        raise ValueError(
            f"Path {path!r} escapes base directory {base_path}"
        ) from exc
    except OSError as exc:
        raise ValueError(f"Cannot resolve path: {path!r}") from exc
    return resolved


def is_allowed_upload(filename: str) -> bool:
    """判断上传文件后缀是否在白名单内。"""
    try:
        suffix = Path(filename).suffix.lower()
    except (TypeError, ValueError):
        return False
    return suffix in ALLOWED_UPLOAD_SUFFIXES


def validate_upload_filename(filename: str) -> str:
    """校验并规范化上传文件名；非法后缀抛出 ValueError。"""
    safe_name = sanitize_filename(filename)
    if not is_allowed_upload(safe_name):
        allowed = ", ".join(sorted(ALLOWED_UPLOAD_SUFFIXES))
        raise ValueError(
            f"Unsupported file type: {Path(safe_name).suffix!r}. "
            f"Allowed: {allowed}"
        )
    return safe_name


def unique_filename(original_filename: str, *, prefix: str = "") -> str:
    """生成带时间戳与短 UUID 的唯一文件名。"""
    safe_name = sanitize_filename(original_filename)
    stem = Path(safe_name).stem
    suffix = Path(safe_name).suffix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = uuid.uuid4().hex[:8]
    safe_prefix = ""
    if prefix.strip():
        safe_prefix = _sanitize_dir_name(prefix) + "_"
    return f"{safe_prefix}{stem}_{timestamp}_{short_id}{suffix}"


def build_temp_file_path(
    subdir: str,
    original_filename: str,
    *,
    prefix: str = "",
) -> Path:
    """在 temp 子目录下生成唯一文件路径（不创建文件）。"""
    target_dir = ensure_subdirectory(subdir)
    filename = unique_filename(original_filename, prefix=prefix)
    return target_dir / filename


def delete_path(path: str | Path) -> bool:
    """删除文件，或删除空目录。路径不存在返回 False。"""
    target = Path(path)
    try:
        if not target.exists():
            return False
        if target.is_file():
            target.unlink()
            return True
        if target.is_dir():
            target.rmdir()
            return True
    except OSError:
        raise
    return False


def list_temp_files(subdir: str | None = None) -> list[Path]:
    """列出 temp 目录或指定子目录下的文件（不递归）。"""
    try:
        root = ensure_subdirectory(subdir) if subdir else get_temp_dir()
        if not root.is_dir():
            return []
        return sorted(p for p in root.iterdir() if p.is_file())
    except OSError:
        raise


def _sanitize_dir_name(name: str) -> str:
    cleaned = re.sub(r"[^\w\-]", "_", name.strip())
    cleaned = _MULTI_UNDERSCORE.sub("_", cleaned).strip("_")
    if not cleaned:
        raise ValueError("Subdirectory name cannot be empty.")
    return cleaned


def _run_self_check() -> None:
    """模块内置自检，便于 ``python -m utils.path_helper`` 调试。"""
    print("=== path_helper self-check ===")
    print(f"project_root: {get_project_root()}")
    print(f"temp_dir:     {get_temp_dir()}")

    upload_dir = ensure_subdirectory(UPLOAD_SUBDIR)
    print(f"upload_dir:   {upload_dir}")

    samples = [
        "sales_data.csv",
        r"..\..\etc\passwd.csv",
        "报表 2024.xlsx",
        "bad.exe",
    ]
    for name in samples:
        try:
            safe = sanitize_filename(name)
            allowed = is_allowed_upload(safe)
            print(f"  sanitize {name!r} -> {safe!r}, allowed={allowed}")
        except ValueError as exc:
            print(f"  sanitize {name!r} -> ValueError: {exc}")

    try:
        validate_upload_filename("demo.csv")
        print("validate_upload_filename('demo.csv'): ok")
    except ValueError as exc:
        print(f"validate_upload_filename('demo.csv'): {exc}")

    try:
        validate_upload_filename("demo.exe")
    except ValueError as exc:
        print(f"validate_upload_filename('demo.exe'): expected error -> {exc}")

    temp_file = build_temp_file_path(UPLOAD_SUBDIR, "demo.csv", prefix="test")
    print(f"build_temp_file_path: {temp_file}")
    try:
        temp_file.write_text("hello", encoding="utf-8")
        print(f"write test file: ok ({temp_file.stat().st_size} bytes)")
        deleted = delete_path(temp_file)
        print(f"delete_path: {deleted}")
    except OSError as exc:
        print(f"file io error: {exc}")

    try:
        resolve_under_base("uploads/demo.csv", get_temp_dir())
        print("resolve_under_base (valid): ok")
    except ValueError as exc:
        print(f"resolve_under_base (valid): {exc}")

    try:
        resolve_under_base("../.env", get_temp_dir())
        print("resolve_under_base (escape): FAIL - should have raised")
    except ValueError:
        print("resolve_under_base (escape): blocked as expected")

    print("=== done ===")


__all__ = [
    "CHART_SUBDIR",
    "OUTPUT_SUBDIR",
    "UPLOAD_SUBDIR",
    "build_temp_file_path",
    "delete_path",
    "ensure_subdirectory",
    "get_project_root",
    "get_temp_dir",
    "is_allowed_upload",
    "list_temp_files",
    "resolve_under_base",
    "sanitize_filename",
    "to_path",
    "unique_filename",
    "validate_upload_filename",
]


if __name__ == "__main__":
    _run_self_check()