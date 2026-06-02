"""根据用户需求与表格预览，生成可在沙箱中执行的 Pandas 代码。"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config.settings import OPENAI_API_BASE, OPENAI_API_KEY, OPENAI_MODEL
from sandbox.code_sandbox import compile_user_code
from sandbox.safe_globals import SecurityError, validate_code_security
from utils.logger import get_logger

log = get_logger()

_CODE_BLOCK_PATTERN = re.compile(
    r"```(?:python)?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

SYSTEM_PROMPT = """你是数据分析代码生成助手，专门为 RestrictedPython 沙箱编写 Pandas 数据处理代码。

## 硬性约束（必须遵守）
1. 只输出可执行的 Python 代码，不要解释文字。
2. 禁止 import / from 语句（沙箱已预置 pd、np、df）。
3. 禁止 os、subprocess、open、eval、exec、网络与文件读写。
4. 输入数据变量名固定为 df（DataFrame 副本）。
5. 最终结果必须赋值给 result（可以是 DataFrame、Series、dict、list 或标量）。
6. 如需保留清洗后的表，可更新 df；但 result 必须能代表本次分析结论。
7. 仅使用 pandas / numpy 常见数据清洗、聚合、筛选、统计方法。
8. 代码应简洁、可一次执行成功，避免假设不存在的列名。

## 输出格式
仅输出一个 ```python 代码块，或纯 Python 代码。"""


class CodeGenerationError(RuntimeError):
    """代码生成或校验失败。"""


@dataclass
class CodeGenerationResult:
    code: str
    raw_response: str
    model: str


def format_data_context(data_preview: dict[str, Any]) -> str:
    """将 file_parser 预览摘要格式化为 LLM 可读上下文。"""
    if not data_preview:
        raise ValueError("data_preview cannot be empty.")

    context = {
        "filename": data_preview.get("filename"),
        "shape": data_preview.get("shape"),
        "columns": data_preview.get("columns"),
        "dtypes": data_preview.get("dtypes"),
        "null_counts": data_preview.get("null_counts"),
        "head": data_preview.get("head"),
    }
    try:
        return json.dumps(context, ensure_ascii=False, indent=2)
    except Exception as exc:
        raise ValueError("Failed to serialize data preview.") from exc


def build_user_message(
    user_request: str,
    data_preview: dict[str, Any],
    *,
    previous_code: str = "",
    previous_error: str = "",
    previous_error_type: str = "",
    retry_count: int = 0,
) -> str:
    """组装用户侧提示词；重试时附带上一轮失败代码与报错。"""
    if not user_request or not user_request.strip():
        raise ValueError("user_request cannot be empty.")
    context = format_data_context(data_preview)
    message = (
        f"用户需求：{user_request.strip()}\n\n"
        f"数据表预览（JSON）：\n{context}\n\n"
    )
    if previous_code.strip() and previous_error.strip():
        message += (
            f"## 上次执行失败（第 {retry_count} 次失败后请求修复）\n"
            f"- 错误类型：{previous_error_type or 'Unknown'}\n"
            f"- 错误信息：{previous_error.strip()}\n\n"
            f"失败代码：\n```python\n{previous_code.strip()}\n```\n\n"
            "请根据错误修正代码，仍须满足沙箱约束，并赋值 result。\n\n"
        )
    message += "请生成满足需求的 Pandas 代码。"
    return message


def extract_python_code(text: str) -> str:
    """从 LLM 回复中提取 Python 代码。"""
    if not text or not str(text).strip():
        raise CodeGenerationError("LLM response is empty.")

    content = str(text).strip()
    match = _CODE_BLOCK_PATTERN.search(content)
    if match:
        return match.group(1).strip()

    if any(keyword in content for keyword in ("df", "pd.", "np.", "result")):
        return content
    raise CodeGenerationError("No Python code found in LLM response.")


def sanitize_generated_code(code: str) -> str:
    """移除 import 语句并清理空白。"""
    cleaned_lines: list[str] = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            continue
        cleaned_lines.append(line.rstrip())
    cleaned = "\n".join(cleaned_lines).strip()
    if not cleaned:
        raise CodeGenerationError("Generated code is empty after sanitization.")
    return cleaned


def validate_generated_code(code: str) -> None:
    """静态校验 + RestrictedPython 编译。"""
    try:
        validate_code_security(code)
        compile_user_code(code)
    except SecurityError as exc:
        raise CodeGenerationError(f"Generated code failed security check: {exc}") from exc
    except Exception as exc:
        raise CodeGenerationError(f"Generated code failed compilation: {exc}") from exc


def create_chat_model(
    *,
    model: str | None = None,
    temperature: float = 0.2,
) -> ChatOpenAI:
    """创建 LangChain ChatOpenAI 客户端（OpenAI 兼容接口）。"""
    try:
        return ChatOpenAI(
            model=model or OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_API_BASE,
            temperature=temperature,
        )
    except Exception as exc:
        raise CodeGenerationError(f"Failed to create chat model: {exc}") from exc


def generate_pandas_code(
    user_request: str,
    data_preview: dict[str, Any],
    *,
    llm: ChatOpenAI | None = None,
    model: str | None = None,
    previous_code: str = "",
    previous_error: str = "",
    previous_error_type: str = "",
    retry_count: int = 0,
) -> CodeGenerationResult:
    """调用 LLM 生成并通过沙箱规则校验的 Pandas 代码。"""
    user_message = build_user_message(
        user_request,
        data_preview,
        previous_code=previous_code,
        previous_error=previous_error,
        previous_error_type=previous_error_type,
        retry_count=retry_count,
    )
    chat = llm or create_chat_model(model=model)
    model_name = getattr(chat, "model_name", None) or model or OPENAI_MODEL

    try:
        response = chat.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_message),
            ]
        )
    except Exception as exc:
        log.exception("LLM invocation failed")
        raise CodeGenerationError(f"LLM invocation failed: {exc}") from exc

    raw_content = getattr(response, "content", "")
    if isinstance(raw_content, list):
        raw_text = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in raw_content
        )
    else:
        raw_text = str(raw_content)

    try:
        code = sanitize_generated_code(extract_python_code(raw_text))
        validate_generated_code(code)
    except CodeGenerationError:
        raise
    except Exception as exc:
        raise CodeGenerationError(f"Failed to process LLM response: {exc}") from exc

    log.info(
        "Generated pandas code ({} chars), model={}, retry_count={}",
        len(code),
        model_name,
        retry_count,
    )
    return CodeGenerationResult(code=code, raw_response=raw_text, model=model_name)


def _run_self_check(*, live: bool = False) -> None:
    """模块内置自检。"""
    print("=== code_generator self-check ===")

    preview = {
        "filename": "demo.csv",
        "shape": [3, 3],
        "columns": [
            {"name": "name", "dtype": "object"},
            {"name": "amount", "dtype": "int64"},
            {"name": "date", "dtype": "object"},
        ],
        "dtypes": {"name": "object", "amount": "int64", "date": "object"},
        "null_counts": {"name": 0, "amount": 1, "date": 0},
        "head": [
            {"name": "Alice", "amount": 100, "date": "2024-01-01"},
            {"name": "Bob", "amount": None, "date": "2024-01-02"},
        ],
    }

    sample_response = (
        "```python\n"
        "import pandas as pd\n"
        "df = df.dropna(subset=['amount'])\n"
        "result = df['amount'].sum()\n"
        "```"
    )
    extracted = sanitize_generated_code(extract_python_code(sample_response))
    print(f"extract/sanitize: {extracted!r}")
    validate_generated_code(extracted)
    print("static validation: ok")

    context = format_data_context(preview)
    print(f"context length: {len(context)} chars")

    if live:
        print("live LLM test...")
        outcome = generate_pandas_code(
            "删除 amount 为空的行，并计算 amount 总和",
            preview,
        )
        print(f"model: {outcome.model}")
        print(f"code:\n{outcome.code}")
    else:
        print("live LLM test skipped (use --live to enable)")

    print("=== done ===")


__all__ = [
    "CodeGenerationError",
    "CodeGenerationResult",
    "build_user_message",
    "create_chat_model",
    "extract_python_code",
    "format_data_context",
    "generate_pandas_code",
    "sanitize_generated_code",
    "validate_generated_code",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="code_generator self-check")
    parser.add_argument(
        "--live",
        action="store_true",
        help="call real LLM API (requires valid .env)",
    )
    args = parser.parse_args()
    _run_self_check(live=args.live)
