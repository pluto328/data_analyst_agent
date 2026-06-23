"""项目全局配置：路径常量、环境变量、Windows 兼容项。

通过 ``python-dotenv`` 加载项目根目录 ``.env``；业务模块应从此处读取配置，
勿在其它文件中硬编码路径或魔法数字。
"""

from __future__ import annotations

import os
from pathlib import Path

# Windows：避免 libiomp5md.dll 重复初始化导致 Streamlit/图表进程异常退出
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ---------------------------------------------------------------------------
# 路径（相对本文件定位项目根目录）
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
TEMP_DIR: Path = PROJECT_ROOT / "temp_files"
ENV_FILE: Path = PROJECT_ROOT / ".env"

# ---------------------------------------------------------------------------
# 文件与编码（Windows 兼容）
# ---------------------------------------------------------------------------
CSV_ENCODINGS: tuple[str, ...] = ("utf-8", "gbk", "gb2312")
ALLOWED_UPLOAD_SUFFIXES: frozenset[str] = frozenset({".csv", ".xls", ".xlsx"})
XLS_SUFFIX: str = ".xls"
XLSX_SUFFIX: str = ".xlsx"
CSV_SUFFIX: str = ".csv"

# ---------------------------------------------------------------------------
# Matplotlib（Windows 中文显示）
# ---------------------------------------------------------------------------
MPL_FONT_FAMILY: str = "SimHei"
MPL_UNICODE_MINUS: bool = False

# ---------------------------------------------------------------------------
# 默认值（可被 .env 覆盖）
# ---------------------------------------------------------------------------
_DEFAULT_SANDBOX_TIMEOUT_SEC: int = 30
_DEFAULT_MAX_UPLOAD_MB: int = 20
_DEFAULT_MAX_UPLOAD_FILES: int = 10
_DEFAULT_MAX_TOTAL_UPLOAD_MB: int = 50
_DEFAULT_OPENAI_MODEL: str = "gpt-4o-mini"
_DEFAULT_LOG_LEVEL: str = "INFO"
_DEFAULT_CORRECTION_MAX_RECORDS: int = 200
_DEFAULT_CORRECTION_TOP_K: int = 2
_DEFAULT_MAX_OUTPUT_TABLES: int = 5


def _load_dotenv() -> None:
    """加载 .env；文件缺失或解析失败时不抛错。"""
    try:
        from dotenv import load_dotenv

        if ENV_FILE.is_file():
            load_dotenv(ENV_FILE, override=False)
        else:
            load_dotenv(override=False)
    except Exception:
        pass


def _get_env_str(key: str, default: str = "") -> str:
    value = os.getenv(key)
    if value is None or not value.strip():
        return default
    return value.strip()


def _require_env_str(key: str) -> str:
    """读取必填环境变量；缺失或仅空白时抛出 ValueError。"""
    value = os.getenv(key)
    if value is None or not value.strip():
        raise ValueError(
            f"Missing required environment variable: {key}. "
            f"Set it in {ENV_FILE} or the process environment."
        )
    return value.strip()


def _get_env_int(key: str, default: int, *, minimum: int = 1) -> int:
    raw = os.getenv(key)
    if raw is None or not str(raw).strip():
        return default
    try:
        parsed = int(str(raw).strip())
    except ValueError:
        return default
    return max(minimum, parsed)


def _get_env_bool(key: str, default: bool = True) -> bool:
    raw = os.getenv(key)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


_load_dotenv()

# ---------------------------------------------------------------------------
# LLM / LangChain（OpenAI 兼容接口）
# ---------------------------------------------------------------------------
OPENAI_API_KEY: str = _require_env_str("OPENAI_API_KEY")
OPENAI_API_BASE: str = _require_env_str("OPENAI_API_BASE")
OPENAI_MODEL: str = _get_env_str("OPENAI_MODEL", _DEFAULT_OPENAI_MODEL)

# ---------------------------------------------------------------------------
# 沙箱与上传限制
# ---------------------------------------------------------------------------
SANDBOX_TIMEOUT_SEC: int = _get_env_int(
    "SANDBOX_TIMEOUT_SEC",
    _DEFAULT_SANDBOX_TIMEOUT_SEC,
    minimum=1,
)
MAX_UPLOAD_MB: int = _get_env_int(
    "MAX_UPLOAD_MB",
    _DEFAULT_MAX_UPLOAD_MB,
    minimum=1,
)
MAX_UPLOAD_BYTES: int = MAX_UPLOAD_MB * 1024 * 1024
MAX_UPLOAD_FILES: int = _get_env_int(
    "MAX_UPLOAD_FILES",
    _DEFAULT_MAX_UPLOAD_FILES,
    minimum=1,
)
MAX_TOTAL_UPLOAD_MB: int = _get_env_int(
    "MAX_TOTAL_UPLOAD_MB",
    _DEFAULT_MAX_TOTAL_UPLOAD_MB,
    minimum=1,
)
MAX_TOTAL_UPLOAD_BYTES: int = MAX_TOTAL_UPLOAD_MB * 1024 * 1024

# ---------------------------------------------------------------------------
# 改错记录（few-shot 自我迭代）
# ---------------------------------------------------------------------------
CORRECTION_ENABLED: bool = _get_env_bool("CORRECTION_ENABLED", True)
CORRECTION_MAX_RECORDS: int = _get_env_int(
    "CORRECTION_MAX_RECORDS",
    _DEFAULT_CORRECTION_MAX_RECORDS,
    minimum=10,
)
CORRECTION_TOP_K: int = _get_env_int(
    "CORRECTION_TOP_K",
    _DEFAULT_CORRECTION_TOP_K,
    minimum=1,
)

# ---------------------------------------------------------------------------
# 输出结果表
# ---------------------------------------------------------------------------
MAX_OUTPUT_TABLES: int = _get_env_int(
    "MAX_OUTPUT_TABLES",
    _DEFAULT_MAX_OUTPUT_TABLES,
    minimum=1,
)

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
LOG_LEVEL: str = _get_env_str("LOG_LEVEL", _DEFAULT_LOG_LEVEL).upper()


def ensure_temp_dir() -> Path:
    """确保临时目录存在；失败时抛出 OSError。"""
    try:
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise
    return TEMP_DIR


def configure_matplotlib() -> None:
    """应用 Windows 中文字体与负号显示设置。"""
    try:
        import matplotlib as mpl

        mpl.rcParams["font.sans-serif"] = [MPL_FONT_FAMILY]
        mpl.rcParams["axes.unicode_minus"] = MPL_UNICODE_MINUS
    except Exception:
        pass


__all__ = [
    "ALLOWED_UPLOAD_SUFFIXES",
    "CSV_ENCODINGS",
    "CSV_SUFFIX",
    "CORRECTION_ENABLED",
    "CORRECTION_MAX_RECORDS",
    "CORRECTION_TOP_K",
    "ENV_FILE",
    "LOG_LEVEL",
    "MAX_OUTPUT_TABLES",
    "MAX_TOTAL_UPLOAD_BYTES",
    "MAX_TOTAL_UPLOAD_MB",
    "MAX_UPLOAD_BYTES",
    "MAX_UPLOAD_FILES",
    "MAX_UPLOAD_MB",
    "MPL_FONT_FAMILY",
    "MPL_UNICODE_MINUS",
    "OPENAI_API_BASE",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "PROJECT_ROOT",
    "SANDBOX_TIMEOUT_SEC",
    "TEMP_DIR",
    "XLS_SUFFIX",
    "XLSX_SUFFIX",
    "configure_matplotlib",
    "ensure_temp_dir",
]
