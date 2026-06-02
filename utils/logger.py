"""基于 loguru 的统一日志初始化。"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from config.settings import LOG_LEVEL, TEMP_DIR, ensure_temp_dir

LOG_SUBDIR: str = "logs"
_DEFAULT_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)
_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
    "{name}:{function}:{line} | {message}"
)
_VALID_LEVELS: frozenset[str] = frozenset(
    {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}
)

_configured: bool = False


def normalize_log_level(level: str | None = None) -> str:
    """规范化日志级别；非法值回退为 INFO。"""
    candidate = (level or LOG_LEVEL or "INFO").strip().upper()
    if candidate not in _VALID_LEVELS:
        return "INFO"
    return candidate


def get_log_dir() -> Path:
    """返回日志目录 ``temp_files/logs``，不存在则创建。"""
    log_dir = ensure_temp_dir() / LOG_SUBDIR
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(f"Cannot create log directory: {log_dir}") from exc
    return log_dir


def setup_logger(
    *,
    log_to_file: bool = True,
    log_level: str | None = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> None:
    """配置 loguru：控制台 + 可选文件输出。重复调用会先移除已有 handler。"""
    global _configured

    level = normalize_log_level(log_level)
    try:
        logger.remove()
    except Exception:
        pass

    try:
        logger.add(
            sys.stderr,
            format=_DEFAULT_FORMAT,
            level=level,
            colorize=True,
            enqueue=True,
        )
    except Exception as exc:
        raise RuntimeError("Failed to configure console logging") from exc

    if log_to_file:
        try:
            log_path = get_log_dir() / "app_{time:YYYY-MM-DD}.log"
            logger.add(
                str(log_path),
                format=_FILE_FORMAT,
                level=level,
                rotation=rotation,
                retention=retention,
                encoding="utf-8",
                enqueue=True,
            )
        except Exception as exc:
            raise RuntimeError("Failed to configure file logging") from exc

    _configured = True
    logger.debug("Logger initialized at level {}", level)


def get_logger():
    """返回 loguru logger；若尚未配置则自动 ``setup_logger()``。"""
    if not _configured:
        setup_logger()
    return logger


def _run_self_check() -> None:
    """模块内置自检，便于 ``python -m utils.logger`` 调试。"""
    print("=== logger self-check ===")
    print(f"configured before: {_configured}")
    print(f"log_level:         {normalize_log_level()}")
    print(f"log_dir:           {get_log_dir()}")

    setup_logger(log_to_file=True)
    log = get_logger()

    log.trace("trace message (may be hidden)")
    log.debug("debug message")
    log.info("info message")
    log.warning("warning message")
    log.error("error message")

    log_files = sorted(get_log_dir().glob("app_*.log"))
    print(f"log files:         {len(log_files)}")
    if log_files:
        latest = log_files[-1]
        print(f"latest log file:   {latest}")
        try:
            tail = latest.read_text(encoding="utf-8").splitlines()[-3:]
            for line in tail:
                print(f"  | {line}")
        except OSError as exc:
            print(f"  read log failed: {exc}")

    print(f"configured after:  {_configured}")
    print("=== done ===")


__all__ = [
    "LOG_SUBDIR",
    "get_log_dir",
    "get_logger",
    "normalize_log_level",
    "setup_logger",
]


if __name__ == "__main__":
    _run_self_check()
