"""根据沙箱运行结果与图表，生成 Markdown 数据分析报告。"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from agent.code_generator import CodeGenerationError, create_chat_model
from config.settings import OPENAI_MODEL
from sandbox.code_sandbox import SandboxResult
from utils.logger import get_logger
from utils.path_helper import OUTPUT_SUBDIR, build_temp_file_path, delete_path

log = get_logger()
# 匹配 Markdown 代码块
_MARKDOWN_BLOCK_PATTERN = re.compile(
    r"```(?:markdown|md)?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


SYSTEM_PROMPT = """你是专业的数据分析师，负责撰写清晰、结构化的 Markdown 数据分析报告。

## 写作要求
1. 使用中文，语气专业但易读。
2. 只输出 Markdown 正文，不要包裹在代码块中（除非用户明确要求）。
3. 必须包含以下二级标题（按顺序）：
   - ## 分析概要
   - ## 数据概况
   - ## 处理与结果
   - ## 可视化说明（若无图表则写“本次未生成图表”）
   - ## 结论与建议
4. 结合用户原始需求、数据预览、执行代码与运行结果撰写，不要编造不存在的列或数值。
5. 若执行失败，说明失败原因并给出可操作的排查建议。
6. 结果表格可用 Markdown 表格呈现；数值保留合理精度。
7. 不要输出 HTML 或 JavaScript。"""


class ReportGenerationError(RuntimeError):
    """报告生成失败。"""


@dataclass
class ReportGenerationResult:
    markdown: str
    raw_response: str
    model: str
    saved_path: Path | None = None

# 将值转换为 JSON 安全的格式
def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, pd.DataFrame):
        preview = value.head(10)
        return {
            "type": "DataFrame",
            "shape": list(value.shape),
            "preview": preview.to_dict(orient="records"),
        }
    if isinstance(value, pd.Series):
        return {
            "type": "Series",
            "name": str(value.name),
            "preview": value.head(20).to_dict(),
        }
    if isinstance(value, (dict, list, str, int, float, bool)):
        return value
    return str(value)


def format_execution_context(
    sandbox_result: SandboxResult,
    *,
    generated_code: str = "",
    stdout: str = "",
) -> str:
    """将沙箱执行结果格式化为 LLM 可读 JSON。"""
    payload = {
        "success": sandbox_result.success,
        "timed_out": sandbox_result.timed_out,
        "error_type": sandbox_result.error_type,
        "error": sandbox_result.error,
        "stdout": stdout or sandbox_result.stdout,
        "generated_code": generated_code.strip(),
        "result": _json_safe(sandbox_result.result),
        "output_shape": (
            list(sandbox_result.df.shape) if sandbox_result.df is not None else None
        ),
    }
    try:
        return json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    except Exception as exc:
        raise ValueError("Failed to serialize execution context.") from exc


def format_chart_context(chart_paths: list[str | Path] | None) -> str:
    """格式化图表路径列表。"""
    if not chart_paths:
        return "[]"
    names = [Path(p).name for p in chart_paths]
    try:
        return json.dumps(names, ensure_ascii=False, indent=2)
    except Exception as exc:
        raise ValueError("Failed to serialize chart paths.") from exc


def build_report_user_message(
    user_request: str,
    data_preview: dict[str, Any],
    sandbox_result: SandboxResult,
    *,
    generated_code: str = "",
    chart_paths: list[str | Path] | None = None,
) -> str:
    """组装报告生成提示词。"""
    if not user_request or not user_request.strip():
        raise ValueError("user_request cannot be empty.")
    if not data_preview:
        raise ValueError("data_preview cannot be empty.")

    preview_payload = {
        "filename": data_preview.get("filename"),
        "shape": data_preview.get("shape"),
        "columns": data_preview.get("columns"),
        "dtypes": data_preview.get("dtypes"),
        "null_counts": data_preview.get("null_counts"),
        "head": data_preview.get("head"),
    }
    try:
        preview_json = json.dumps(preview_payload, ensure_ascii=False, indent=2)
    except Exception as exc:
        raise ValueError("Failed to serialize data preview.") from exc

    execution_json = format_execution_context(
        sandbox_result,
        generated_code=generated_code,
    )
    charts_json = format_chart_context(chart_paths)

    return (
        f"用户原始需求：{user_request.strip()}\n\n"
        f"数据预览（JSON）：\n{preview_json}\n\n"
        f"沙箱执行结果（JSON）：\n{execution_json}\n\n"
        f"图表文件（JSON 文件名列表）：\n{charts_json}\n\n"
        "请撰写完整 Markdown 分析报告。"
    )


def extract_markdown(text: str) -> str:
    """从 LLM 回复中提取 Markdown 正文。"""
    if not text or not str(text).strip():
        raise ReportGenerationError("LLM response is empty.")

    content = str(text).strip()
    match = _MARKDOWN_BLOCK_PATTERN.search(content)
    if match:
        return match.group(1).strip()
    return content


def build_fallback_report(
    user_request: str,
    data_preview: dict[str, Any],
    sandbox_result: SandboxResult,
    *,
    generated_code: str = "",
    chart_paths: list[str | Path] | None = None,
) -> str:
    """LLM 不可用时的模板化 Markdown 报告（离线/降级）。"""
    filename = data_preview.get("filename", "unknown")
    shape = data_preview.get("shape", ["?", "?"])
    chart_names = [Path(p).name for p in (chart_paths or [])]

    lines = [
        "## 分析概要",
        f"- 用户需求：{user_request.strip()}",
        f"- 数据文件：`{filename}`",
        f"- 执行状态：{'成功' if sandbox_result.success else '失败'}",
        "",
        "## 数据概况",
        f"- 行列规模：{shape[0]} 行 × {shape[1]} 列",
        f"- 字段：{', '.join(str(c.get('name', c)) for c in data_preview.get('columns', []))}",
        "",
        "## 处理与结果",
    ]

    if generated_code.strip():
        lines.extend(["### 执行代码", "```python", generated_code.strip(), "```", ""])

    if sandbox_result.success:
        lines.append(f"- 分析结果：`{_json_safe(sandbox_result.result)}`")
        if sandbox_result.stdout.strip():
            lines.append(f"- 标准输出：{sandbox_result.stdout.strip()}")
        if sandbox_result.df is not None:
            lines.append(f"- 输出表规模：{sandbox_result.df.shape[0]} 行 × {sandbox_result.df.shape[1]} 列")
    else:
        lines.append(f"- 错误类型：{sandbox_result.error_type or 'Unknown'}")
        lines.append(f"- 错误信息：{sandbox_result.error or '无'}")
        if sandbox_result.timed_out:
            lines.append("- 备注：执行超时，请简化代码或提高超时阈值。")

    lines.extend(["", "## 可视化说明"])
    if chart_names:
        for name in chart_names:
            lines.append(f"- 图表：`{name}`")
    else:
        lines.append("- 本次未生成图表")

    lines.extend(
        [
            "",
            "## 结论与建议",
            "- 以上为系统自动汇总；如需更深度解读，请启用 LLM 报告生成。",
        ]
    )
    return "\n".join(lines)


def save_report_markdown(markdown: str, *, title: str = "analysis_report") -> Path:
    """将报告保存到 ``temp_files/outputs``。"""
    if not markdown or not markdown.strip():
        raise ValueError("markdown cannot be empty.")
    output_path = build_temp_file_path(OUTPUT_SUBDIR, f"{title}.md", prefix="report")
    try:
        output_path.write_text(markdown.strip() + "\n", encoding="utf-8")
        log.info("Saved markdown report: {}", output_path.name)
        return output_path
    except OSError as exc:
        raise OSError(f"Failed to save report: {output_path}") from exc


def generate_markdown_report(
    user_request: str,
    data_preview: dict[str, Any],
    sandbox_result: SandboxResult,
    *,
    generated_code: str = "",
    chart_paths: list[str | Path] | None = None,
    llm=None,
    model: str | None = None,
    save_to_file: bool = False,
    report_title: str = "analysis_report",
) -> ReportGenerationResult:
    """调用 LLM 生成 Markdown 分析报告。"""
    user_message = build_report_user_message(
        user_request,
        data_preview,
        sandbox_result,
        generated_code=generated_code,
        chart_paths=chart_paths,
    )
    chat = llm or create_chat_model(model=model, temperature=0.4)
    model_name = getattr(chat, "model_name", None) or model or OPENAI_MODEL

    try:
        response = chat.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_message),
            ]
        )
    except Exception as exc:
        log.exception("LLM report generation failed")
        raise ReportGenerationError(f"LLM invocation failed: {exc}") from exc

    raw_content = getattr(response, "content", "")
    if isinstance(raw_content, list):
        raw_text = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in raw_content
        )
    else:
        raw_text = str(raw_content)

    try:
        markdown = extract_markdown(raw_text).strip()
        if not markdown:
            raise ReportGenerationError("Extracted markdown is empty.")
    except ReportGenerationError:
        raise
    except Exception as exc:
        raise ReportGenerationError(f"Failed to process LLM response: {exc}") from exc

    saved_path = None
    if save_to_file:
        try:
            saved_path = save_report_markdown(markdown, title=report_title)
        except OSError as exc:
            raise ReportGenerationError(f"Failed to save report: {exc}") from exc

    log.info("Generated markdown report ({} chars), model={}", len(markdown), model_name)
    return ReportGenerationResult(
        markdown=markdown,
        raw_response=raw_text,
        model=model_name,
        saved_path=saved_path,
    )


def _run_self_check(*, live: bool = False) -> None:
    """模块内置自检。"""
    print("=== report_generator self-check ===")

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
    code = "df = df.dropna(subset=['amount'])\nresult = df['amount'].sum()"
    sandbox_ok = SandboxResult(
        success=True,
        result=300,
        df=pd.DataFrame({"name": ["Alice"], "amount": [100], "date": ["2024-01-01"]}),
        stdout="",
    )
    charts = ["temp_files/charts/chart_demo.png"]

    fallback = build_fallback_report(
        "删除空值并求和",
        preview,
        sandbox_ok,
        generated_code=code,
        chart_paths=charts,
    )
    print(f"fallback report: {len(fallback)} chars")
    assert "## 分析概要" in fallback
    print("fallback template: ok")

    saved = save_report_markdown(fallback, title="self_check")
    print(f"saved report: {saved.name}")
    # try:
    #     delete_path(saved)
    # except OSError as exc:
    #     print(f"cleanup failed: {exc}")

    if live:
        print("live LLM test...")
        outcome = generate_markdown_report(
            "删除 amount 空值并计算总和",
            preview,
            sandbox_ok,
            generated_code=code,
            chart_paths=charts,
            save_to_file=True,
            report_title="live_report",
        )
        print(f"model: {outcome.model}")
        print(f"report preview:\n{outcome.markdown[:400]}...")
        if outcome.saved_path:
            print(f"saved: {outcome.saved_path.name}")
        #     try:
        #         delete_path(outcome.saved_path)
        #     except OSError as exc:
        #         print(f"cleanup failed: {exc}")
    else:
        print("live LLM test skipped (use --live to enable)")

    print("=== done ===")


__all__ = [
    "ReportGenerationError",
    "ReportGenerationResult",
    "build_fallback_report",
    "build_report_user_message",
    "extract_markdown",
    "format_chart_context",
    "format_execution_context",
    "generate_markdown_report",
    "save_report_markdown",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="report_generator self-check")
    parser.add_argument(
        "--live",
        action="store_true",
        help="call real LLM API (requires valid .env)",
    )
    args = parser.parse_args()
    _run_self_check(live=args.live)
