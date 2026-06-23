"""LangGraph 工作流：代码生成 ↔ 沙箱执行，失败时带错误上下文回溯重试。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

import pandas as pd
from langgraph.graph import END, StateGraph

from agent.code_generator import CodeGenerationError, generate_pandas_code
from agent.correction_store import save_correction_from_retry_success
from sandbox.code_sandbox import SandboxResult, execute_code
from utils.logger import get_logger

log = get_logger()

# 最多重试 3 次（不含首次，共最多 4 轮 生成+执行）
MAX_CODE_RETRIES: int = 3


class RetryRecord(TypedDict):
    attempt: int
    code: str
    error_type: str | None
    error: str | None


class AnalysisState(TypedDict, total=False):
    user_request: str
    data_preview: dict[str, Any]
    datasets: dict[str, pd.DataFrame]
    max_retries: int

    retry_count: int
    previous_code: str
    previous_error: str
    previous_error_type: str
    retry_history: list[RetryRecord]

    generated_code: str
    model: str
    raw_response: str

    sandbox_result: SandboxResult | None
    code_generation_error: str

    status: Literal["running", "success", "execution_failed", "generation_failed"]


@dataclass
class AnalysisGraphResult:
    success: bool
    generated_code: str
    model: str
    sandbox_result: SandboxResult | None
    retry_history: list[RetryRecord] = field(default_factory=list)
    code_generation_error: str | None = None
    total_attempts: int = 0
    status: str = "running"


def _generate_code_node(state: AnalysisState) -> AnalysisState:
    """调用 code_generator；重试时附带上一轮失败代码与报错。"""
    try:
        outcome = generate_pandas_code(
            state["user_request"],
            state["data_preview"],
            previous_code=state.get("previous_code", ""),
            previous_error=state.get("previous_error", ""),
            previous_error_type=state.get("previous_error_type", ""),
            retry_count=state.get("retry_count", 0),
        )
    except CodeGenerationError as exc:
        log.exception("Code generation failed in graph node")
        return {
            "code_generation_error": str(exc),
            "status": "generation_failed",
        }

    attempt_no = state.get("retry_count", 0) + 1
    log.info("Graph generate node attempt {}", attempt_no)
    return {
        "generated_code": outcome.code,
        "model": outcome.model,
        "raw_response": outcome.raw_response,
        "code_generation_error": "",
    }


def _execute_code_node(state: AnalysisState) -> AnalysisState:
    """在沙箱中执行当前 generated_code。"""
    code = state.get("generated_code", "")
    datasets = state["datasets"]
    try:
        sandbox_result = execute_code(code, datasets=datasets)
    except Exception as exc:
        log.exception("Sandbox execution raised in graph node")
        sandbox_result = SandboxResult(
            success=False,
            error=str(exc),
            error_type=type(exc).__name__,
        )

    if sandbox_result.success:
        return {
            "sandbox_result": sandbox_result,
            "status": "success",
        }

    retry_count = state.get("retry_count", 0) + 1
    history = list(state.get("retry_history", []))
    history.append(
        {
            "attempt": retry_count,
            "code": code,
            "error_type": sandbox_result.error_type,
            "error": sandbox_result.error,
        }
    )
    return {
        "sandbox_result": sandbox_result,
        "retry_count": retry_count,
        "previous_code": code,
        "previous_error": sandbox_result.error or "",
        "previous_error_type": sandbox_result.error_type or "ExecutionError",
        "retry_history": history,
        "status": "execution_failed",
    }


def _route_after_generate(state: AnalysisState) -> Literal["execute", "finish"]:
    if state.get("status") == "generation_failed":
        return "finish"
    if not state.get("generated_code", "").strip():
        return "finish"
    return "execute"


def _route_after_execute(state: AnalysisState) -> Literal["retry", "finish"]:
    """执行成功后结束；失败且未达重试上限则回到生成节点。"""
    if state.get("status") == "success":
        return "finish"
    if state.get("status") == "generation_failed":
        return "finish"
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", MAX_CODE_RETRIES)
    if retry_count > max_retries:
        log.warning(
            "Max retries reached ({}/{}), stopping graph",
            retry_count,
            max_retries,
        )
        return "finish"
    return "retry"


def build_analysis_graph():
    """构建 生成 → 执行 →（失败则回溯重试）状态图。"""
    graph = StateGraph(AnalysisState)
    graph.add_node("generate_code", _generate_code_node)
    graph.add_node("execute_code", _execute_code_node)
    graph.set_entry_point("generate_code")
    graph.add_conditional_edges(
        "generate_code",
        _route_after_generate,
        {
            "execute": "execute_code",
            "finish": END,
        },
    )
    graph.add_conditional_edges(
        "execute_code",
        _route_after_execute,
        {
            "retry": "generate_code",
            "finish": END,
        },
    )
    return graph.compile()


def run_analysis_graph(
    user_request: str,
    data_preview: dict[str, Any],
    datasets: dict[str, pd.DataFrame],
    *,
    max_retries: int = MAX_CODE_RETRIES,
) -> AnalysisGraphResult:
    """运行完整 生成-执行-回溯 工作流。"""
    if not user_request.strip():
        raise ValueError("user_request cannot be empty.")
    if not datasets:
        raise ValueError("datasets cannot be empty.")
    for key, frame in datasets.items():
        if frame is None or frame.empty:
            raise ValueError(f"dataset {key!r} cannot be empty.")

    initial: AnalysisState = {
        "user_request": user_request.strip(),
        "data_preview": data_preview,
        "datasets": datasets,
        "max_retries": max_retries,
        "retry_count": 0,
        "previous_code": "",
        "previous_error": "",
        "previous_error_type": "",
        "retry_history": [],
        "generated_code": "",
        "model": "",
        "status": "running",
    }

    try:
        app = build_analysis_graph()
        final_state = app.invoke(initial)
    except Exception as exc:
        log.exception("Analysis graph invocation failed")
        raise RuntimeError(f"Analysis graph failed: {exc}") from exc

    status = final_state.get("status", "execution_failed")
    sandbox_result = final_state.get("sandbox_result")
    retry_history = final_state.get("retry_history", [])
    retry_count = final_state.get("retry_count", 0)
    if status == "success":
        total_attempts = retry_count + 1
    elif status == "execution_failed":
        total_attempts = retry_count if retry_count else 1
    else:
        total_attempts = 0

    if status == "generation_failed":
        return AnalysisGraphResult(
            success=False,
            generated_code=final_state.get("generated_code", ""),
            model=final_state.get("model", ""),
            sandbox_result=sandbox_result,
            retry_history=retry_history,
            code_generation_error=final_state.get("code_generation_error"),
            total_attempts=total_attempts,
            status="generation_failed",
        )

    success = status == "success" and sandbox_result is not None and sandbox_result.success
    generated_code = final_state.get("generated_code", "")

    if success and retry_history:
        try:
            save_correction_from_retry_success(
                user_request=user_request.strip(),
                data_preview=data_preview,
                retry_history=retry_history,
                correct_code=generated_code,
            )
        except Exception as exc:
            log.exception("Failed to persist correction record: {}", exc)

    return AnalysisGraphResult(
        success=success,
        generated_code=generated_code,
        model=final_state.get("model", ""),
        sandbox_result=sandbox_result,
        retry_history=retry_history,
        code_generation_error=None,
        total_attempts=total_attempts,
        status="success" if success else "execution_failed",
    )


def _run_self_check() -> None:
    """离线自检：验证图结构可编译。"""
    print("=== analysis_graph self-check ===")
    app = build_analysis_graph()
    print(f"graph nodes: {list(app.get_graph().nodes.keys())}")
    print(f"MAX_CODE_RETRIES: {MAX_CODE_RETRIES}")
    print("=== done ===")


__all__ = [
    "MAX_CODE_RETRIES",
    "AnalysisGraphResult",
    "AnalysisState",
    "RetryRecord",
    "build_analysis_graph",
    "run_analysis_graph",
]


if __name__ == "__main__":
    _run_self_check()