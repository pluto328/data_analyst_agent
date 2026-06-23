"""沙箱白名单与静态安全校验。"""

from __future__ import annotations

import ast
import re
from typing import Any

import numpy as np
import pandas as pd

from RestrictedPython import safe_builtins
from RestrictedPython.Guards import (
    full_write_guard,
    guarded_unpack_sequence,
    safer_getattr,
)
from RestrictedPython.PrintCollector import PrintCollector

# ---------------------------------------------------------------------------
# 安全策略常量
# ---------------------------------------------------------------------------
ALLOWED_MODULES: frozenset[str] = frozenset({"pandas", "numpy", "pd", "np"})

FORBIDDEN_ROOT_MODULES: frozenset[str] = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "shutil",
        "socket",
        "pathlib",
        "io",
        "builtins",
        "importlib",
        "ctypes",
        "multiprocessing",
        "threading",
        "pickle",
        "marshal",
        "shelve",
        "sqlite3",
        "http",
        "urllib",
        "ftplib",
        "requests",
        "code",
        "inspect",
        "ast",
        "dis",
        "tempfile",
        "glob",
        "webbrowser",
        "signal",
        "pty",
        "platform",
        "runpy",
        "builtins",
    }
)

FORBIDDEN_BUILTINS: frozenset[str] = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "input",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "dir",
        "help",
        "breakpoint",
        "exit",
        "quit",
        "memoryview",
        "format",
    }
)

FORBIDDEN_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "__class__",
        "__bases__",
        "__subclasses__",
        "__globals__",
        "__code__",
        "__builtins__",
        "__import__",
        "__dict__",
        "__getattribute__",
        "__reduce__",
        "__reduce_ex__",
    }
)

FORBIDDEN_NAME_PREFIXES: tuple[str, ...] = ("__",)

FORBIDDEN_TEXT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bos\s*\.\s*system\b", re.IGNORECASE),
    re.compile(r"\bsubprocess\b", re.IGNORECASE),
    re.compile(r"\b__import__\s*\(", re.IGNORECASE),
    re.compile(r"\bopen\s*\(", re.IGNORECASE),
    re.compile(r"\bsocket\b", re.IGNORECASE),
    re.compile(r"\bshutil\b", re.IGNORECASE),
    re.compile(r"\brequests\s*\.\s*get\b", re.IGNORECASE),
)


class SecurityError(ValueError):
    """用户代码未通过安全校验。"""


_SANDBOX_UNWRAPPED_WRITE_TYPES: tuple[type, ...] = (
    dict,
    list,
    pd.DataFrame,
    pd.Series,
    np.ndarray,
)


def _allow_direct_write(ob: Any) -> bool:
    """判断是否跳过 RestrictedPython Wrapper（允许 subscript / loc 写入）。"""
    if isinstance(ob, _SANDBOX_UNWRAPPED_WRITE_TYPES) or hasattr(ob, "_guarded_writes"):
        return True
    module = getattr(type(ob), "__module__", "") or ""
    return module.startswith(("pandas.", "numpy."))


def sandbox_write_guard(ob: Any) -> Any:
    """对 pandas/numpy 对象跳过 Wrapper，允许列赋值与 loc/iloc 写入。"""
    if _allow_direct_write(ob):
        return ob
    return full_write_guard(ob)


class _SecurityVisitor(ast.NodeVisitor):
    """AST 静态扫描：拦截 import、危险调用与 dunder 访问。"""

    def __init__(self) -> None:
        self.violations: list[str] = []

    def _add(self, message: str) -> None:
        self.violations.append(message)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in FORBIDDEN_ROOT_MODULES:
                self._add(f"Forbidden import: {alias.name}")
            else:
                self._add(f"Import not allowed in sandbox: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        root = module.split(".")[0] if module else ""
        if root in FORBIDDEN_ROOT_MODULES:
            self._add(f"Forbidden import from: {module}")
        else:
            self._add(f"Import-from not allowed in sandbox: {module or '*'}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in FORBIDDEN_BUILTINS:
            self._add(f"Forbidden builtin reference: {node.id}")
        if any(node.id.startswith(prefix) for prefix in FORBIDDEN_NAME_PREFIXES):
            if node.id not in {"_print"}:
                self._add(f"Forbidden name: {node.id}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in FORBIDDEN_ATTRIBUTES:
            self._add(f"Forbidden attribute access: {node.attr}")
        if node.attr.startswith("__"):
            self._add(f"Forbidden dunder attribute: {node.attr}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_BUILTINS:
            self._add(f"Forbidden call: {node.func.id}()")
        self.generic_visit(node)


def scan_forbidden_text(code: str) -> list[str]:
    """基于正则的二次文本扫描（防御编码绕过）。"""
    violations: list[str] = []
    for pattern in FORBIDDEN_TEXT_PATTERNS:
        if pattern.search(code):
            violations.append(f"Forbidden pattern matched: {pattern.pattern}")
    return violations


def validate_code_security(code: str) -> None:
    """静态校验用户代码；不通过则抛出 ``SecurityError``。"""
    if not code or not code.strip():
        raise SecurityError("Code cannot be empty.")

    text_hits = scan_forbidden_text(code)
    if text_hits:
        raise SecurityError("; ".join(text_hits))

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise SecurityError(f"Invalid Python syntax: {exc}") from exc

    visitor = _SecurityVisitor()
    visitor.visit(tree)
    if visitor.violations:
        raise SecurityError("; ".join(visitor.violations))


def build_execution_globals(
    df: pd.DataFrame | None = None,
    *,
    datasets: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """构建 RestrictedPython 执行环境（仅注入白名单模块与副本数据）。"""
    namespace: dict[str, Any] = {
        "__builtins__": safe_builtins,
        "_print_": PrintCollector,
        "_getattr_": safer_getattr,
        "_write_": sandbox_write_guard,
        "_getitem_": lambda obj, index: obj[index],
        "_getiter_": iter,
        "_iter_unpack_sequence_": guarded_unpack_sequence,
        "pd": pd,
        "np": np,
    }

    if datasets:
        for key, frame in datasets.items():
            namespace[key] = frame.copy()
        if len(datasets) == 1:
            namespace["df"] = next(iter(datasets.values())).copy()
        elif "df" not in namespace and datasets:
            namespace["df"] = next(iter(datasets.values())).copy()
    elif df is not None:
        namespace["df"] = df.copy()
    else:
        raise ValueError("Either df or datasets must be provided.")

    return namespace


def extract_outputs(namespace: dict[str, Any]) -> dict[str, Any]:
    """从执行命名空间提取 ``result`` 与 ``df``。"""
    outputs: dict[str, Any] = {}
    if "result" in namespace:
        outputs["result"] = namespace["result"]
    if "df" in namespace:
        outputs["df"] = namespace["df"]

    print_collector = namespace.get("_print")
    if callable(print_collector):
        try:
            outputs["stdout"] = print_collector()
        except Exception:
            outputs["stdout"] = ""
    else:
        outputs["stdout"] = ""
    return outputs


__all__ = [
    "ALLOWED_MODULES",
    "FORBIDDEN_ATTRIBUTES",
    "FORBIDDEN_BUILTINS",
    "FORBIDDEN_ROOT_MODULES",
    "SecurityError",
    "build_execution_globals",
    "extract_outputs",
    "sandbox_write_guard",
    "scan_forbidden_text",
    "validate_code_security",
]