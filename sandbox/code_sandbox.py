"""沙箱核心：代码校验、RestrictedPython 编译、子进程隔离与超时控制。"""

from __future__ import annotations

import multiprocessing as mp
import pickle
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from RestrictedPython import compile_restricted

from config.settings import SANDBOX_TIMEOUT_SEC
from sandbox.safe_globals import (
    SecurityError,
    build_execution_globals,
    extract_outputs,
    validate_code_security,
)
from utils.logger import get_logger
from utils.path_helper import delete_path, ensure_subdirectory

log = get_logger()

SANDBOX_SUBDIR: str = "sandbox"


class SandboxTimeoutError(TimeoutError):
    """沙箱执行超时。"""


class SandboxExecutionError(RuntimeError):
    """沙箱执行失败。"""


@dataclass
class SandboxResult:
    success: bool
    result: Any = None
    df: pd.DataFrame | None = None
    stdout: str = ""
    error: str | None = None
    error_type: str | None = None
    timed_out: bool = False


def compile_user_code(code: str) -> Any:
    """使用 RestrictedPython 编译用户代码。"""
    validate_code_security(code)
    try:
        byte_code = compile_restricted(code, "<sandbox>", "exec")
    except SyntaxError as exc:
        raise SecurityError(f"RestrictedPython rejected code: {exc}") from exc
    if byte_code is None:
        raise SecurityError("RestrictedPython returned empty bytecode.")
    return byte_code


def _sandbox_child_entry(code: str, datasets_path: str, conn) -> None:
    """子进程入口：二次校验后在隔离环境中执行代码。"""
    try:
        validate_code_security(code)
        byte_code = compile_restricted(code, "<sandbox>", "exec")
        if byte_code is None:
            raise SecurityError("RestrictedPython returned empty bytecode.")

        try:
            datasets_bytes = Path(datasets_path).read_bytes()
            datasets = pickle.loads(datasets_bytes)
        except Exception as exc:
            raise RuntimeError(f"Failed to load datasets: {exc}") from exc

        if not isinstance(datasets, dict):
            raise RuntimeError("datasets payload must be a dict.")

        namespace = build_execution_globals(datasets=datasets)
        exec(byte_code, namespace)
        payload = extract_outputs(namespace)
        conn.send(
            {
                "success": True,
                "result": payload.get("result"),
                "df": payload.get("df"),
                "stdout": payload.get("stdout", ""),
            }
        )
    except Exception as exc:
        conn.send(
            {
                "success": False,
                "error": str(exc),
                "error_type": type(exc).__name__,
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _terminate_process(process: mp.Process) -> None:
    """终止子进程：terminate 后必要时 kill。"""
    if not process.is_alive():
        return
    try:
        process.terminate()
        process.join(5)
    except Exception:
        log.exception("Failed to terminate sandbox process")
    if process.is_alive():
        try:
            process.kill()
            process.join(5)
        except Exception:
            log.exception("Failed to kill sandbox process")


def execute_code(
    code: str,
    df: pd.DataFrame | None = None,
    *,
    datasets: dict[str, pd.DataFrame] | None = None,
    timeout_sec: int | None = None,
) -> SandboxResult:
    """在子进程中安全执行用户代码，并返回 ``result`` / ``df``。"""
    if df is None and not datasets:
        raise ValueError("Either df or datasets must be provided.")
    if datasets is None:
        if df is None:
            raise ValueError("df cannot be None when datasets is not provided.")
        datasets = {"df": df}

    timeout = timeout_sec if timeout_sec is not None else SANDBOX_TIMEOUT_SEC
    compile_user_code(code)

    sandbox_dir = ensure_subdirectory(SANDBOX_SUBDIR)
    datasets_path = sandbox_dir / f"datasets_{uuid.uuid4().hex}.pkl"
    try:
        try:
            datasets_path.write_bytes(
                pickle.dumps(datasets, protocol=pickle.HIGHEST_PROTOCOL)
            )
        except Exception as exc:
            raise SandboxExecutionError(f"Failed to serialize datasets: {exc}") from exc

        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        process = ctx.Process(
            target=_sandbox_child_entry,
            args=(code, str(datasets_path), child_conn),
            daemon=True,
        )
        try:
            process.start()
        except Exception as exc:
            raise SandboxExecutionError(f"Failed to start sandbox process: {exc}") from exc
        finally:
            try:
                child_conn.close()
            except Exception:
                pass

        try:
            if parent_conn.poll(timeout):
                payload = parent_conn.recv()
            else:
                _terminate_process(process)
                log.warning("Sandbox execution timed out after {}s", timeout)
                return SandboxResult(
                    success=False,
                    error=f"Execution timed out after {timeout} seconds.",
                    error_type="SandboxTimeoutError",
                    timed_out=True,
                )
        finally:
            try:
                parent_conn.close()
            except Exception:
                pass

        process.join(10)
        if process.is_alive():
            _terminate_process(process)

        if not payload.get("success"):
            return SandboxResult(
                success=False,
                error=payload.get("error"),
                error_type=payload.get("error_type"),
            )

        result_df = payload.get("df")
        if result_df is not None and not isinstance(result_df, pd.DataFrame):
            result_df = None

        return SandboxResult(
            success=True,
            result=payload.get("result"),
            df=result_df,
            stdout=str(payload.get("stdout") or ""),
        )
    finally:
        try:
            delete_path(datasets_path)
        except OSError:
            log.exception("Failed to delete sandbox datasets pickle: {}", datasets_path)


def run_security_audit() -> list[dict[str, Any]]:
    """运行恶意代码样本集，验证沙箱拦截能力。"""
    blocked_cases: list[tuple[str, str]] = [
        ("import_os", "import os\nos.system('echo pwned')"),
        ("import_subprocess", "import subprocess\nsubprocess.run(['echo'])"),
        ("open_file", "open('secret.txt', 'w')"),
        ("eval_call", "eval('1+1')"),
        ("exec_call", "exec('x=1')"),
        ("dunder_escape", "().__class__.__bases__[0].__subclasses__()"),
        ("import_requests", "import requests"),
        ("os_system_text", "os.system('echo')"),
        ("socket_use", "import socket"),
    ]
    allowed_cases: list[tuple[str, str]] = [
        (
            "pandas_groupby",
            "result = df.groupby('category', as_index=False)['value'].sum()",
        ),
        (
            "pandas_dropna",
            "df = df.dropna()\nresult = df.shape",
        ),
        (
            "pandas_setitem",
            "df['total'] = df['value'] * 2\nresult = df.shape",
        ),
        (
            "pandas_loc_assign",
            "df.loc[0, 'value'] = 99\nresult = df.loc[0, 'value']",
        ),
    ]

    sample_df = pd.DataFrame(
        {
            "category": ["A", "A", "B"],
            "value": [1, 2, 3],
        }
    )
    report: list[dict[str, Any]] = []

    for name, code in blocked_cases:
        entry = {"name": name, "kind": "blocked", "passed": False, "detail": ""}
        try:
            validate_code_security(code)
            compile_user_code(code)
            entry["detail"] = "Expected SecurityError but code was accepted."
        except SecurityError as exc:
            entry["passed"] = True
            entry["detail"] = str(exc)
        except Exception as exc:
            entry["detail"] = f"Unexpected error: {type(exc).__name__}: {exc}"
        report.append(entry)

    for name, code in allowed_cases:
        entry = {"name": name, "kind": "allowed", "passed": False, "detail": ""}
        try:
            outcome = execute_code(code, sample_df, timeout_sec=10)
            if outcome.success:
                entry["passed"] = True
                entry["detail"] = f"result={outcome.result!r}"
            else:
                entry["detail"] = f"Execution failed: {outcome.error_type}: {outcome.error}"
        except Exception as exc:
            entry["detail"] = f"Unexpected error: {type(exc).__name__}: {exc}"
        report.append(entry)

    timeout_code = "while True:\n    pass"
    timeout_entry = {
        "name": "infinite_loop_timeout",
        "kind": "blocked",
        "passed": False,
        "detail": "",
    }
    try:
        validate_code_security(timeout_code)
        outcome = execute_code(timeout_code, sample_df, timeout_sec=2)
        if outcome.timed_out and not outcome.success:
            timeout_entry["passed"] = True
            timeout_entry["detail"] = outcome.error or "timed out"
        else:
            timeout_entry["detail"] = "Expected timeout but execution succeeded."
    except SecurityError as exc:
        timeout_entry["passed"] = True
        timeout_entry["detail"] = f"Blocked before execution: {exc}"
    except Exception as exc:
        timeout_entry["detail"] = f"Unexpected error: {type(exc).__name__}: {exc}"
    report.append(timeout_entry)

    return report


def _run_self_check() -> None:
    """模块内置自检 + 安全审计。"""
    print("=== code_sandbox self-check ===")
    report = run_security_audit()
    passed = sum(1 for item in report if item["passed"])
    total = len(report)
    print(f"security audit: {passed}/{total} passed")
    for item in report:
        status = "PASS" if item["passed"] else "FAIL"
        print(f"  [{status}] {item['name']} ({item['kind']}): {item['detail'][:120]}")

    if passed != total:
        raise RuntimeError(f"Security audit failed: {passed}/{total} passed")
    print("=== done ===")


__all__ = [
    "SANDBOX_SUBDIR",
    "SandboxExecutionError",
    "SandboxResult",
    "SandboxTimeoutError",
    "compile_user_code",
    "execute_code",
    "run_security_audit",
]


if __name__ == "__main__":
    _run_self_check()
