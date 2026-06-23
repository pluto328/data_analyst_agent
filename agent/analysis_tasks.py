"""分析任务队列：排队、执行、取消与结果快照。"""

from __future__ import annotations

import threading
import time
import uuid
from copy import deepcopy
from typing import Any, Literal

import pandas as pd

from agent.analysis_graph import run_analysis_graph
from agent.dataset_registry import DatasetInfo, datasets_to_dict, merge_previews_for_legacy
from agent.report_generator import (
    ReportGenerationError,
    build_fallback_report,
    generate_markdown_report,
    save_report_markdown,
)
from utils.result_tables import dataframe_to_csv_bytes, extract_result_tables

TaskStatus = Literal["queued", "running", "completed", "failed", "cancelled"]

STATUS_LABELS: dict[str, str] = {
    "queued": "排队中",
    "running": "处理中",
    "completed": "已完成",
    "failed": "失败",
    "cancelled": "已取消",
}


def empty_task_result() -> dict[str, Any]:
    return {
        "generated_code": "",
        "sandbox_ok": None,
        "retry_history": [],
        "graph_attempts": 0,
        "report_markdown": "",
        "report_path": None,
        "result_tables": [],
        "result_truncated": False,
        "result_scalar": None,
        "error_message": "",
        "success": False,
    }


def init_task_state(session_state: Any) -> None:
    if "analysis_tasks" not in session_state:
        session_state.analysis_tasks = []
    if "selected_task_id" not in session_state:
        session_state.selected_task_id = None
    if "pending_close_task_id" not in session_state:
        session_state.pending_close_task_id = None
    if "pending_cancel_task_id" not in session_state:
        session_state.pending_cancel_task_id = None
    if "task_queue_dirty" not in session_state:
        session_state.task_queue_dirty = False


def get_task_by_id(tasks: list[dict[str, Any]], task_id: str) -> dict[str, Any] | None:
    for task in tasks:
        if task["id"] == task_id:
            return task
    return None


def get_running_task(tasks: list[dict[str, Any]]) -> dict[str, Any] | None:
    for task in tasks:
        if task["status"] == "running":
            return task
    return None


def get_selected_task(tasks: list[dict[str, Any]], selected_id: str | None) -> dict[str, Any] | None:
    if not selected_id:
        return None
    return get_task_by_id(tasks, selected_id)


def enqueue_analysis_task(
    *,
    user_request: str,
    options: dict[str, Any],
    datasets: list[DatasetInfo],
) -> dict[str, Any]:
    preview = merge_previews_for_legacy(datasets)
    task = {
        "id": uuid.uuid4().hex[:8],
        "user_request": user_request,
        "status": "queued",
        "options": deepcopy(options),
        "preview": preview,
        "dataset_filenames": [ds.filename for ds in datasets],
        "created_at": time.time(),
        "started_at": None,
        "finished_at": None,
        "cancel_requested": False,
        "holder": None,
        "result": empty_task_result(),
    }
    return task


def _run_analysis_worker(
    user_request: str,
    options: dict[str, Any],
    preview: dict[str, Any],
    df_dict: dict[str, pd.DataFrame],
    result_holder: dict[str, Any],
) -> None:
    try:
        graph_result = run_analysis_graph(user_request, preview, df_dict)
        result_holder["status"] = "complete"
        result_holder["payload"] = {
            "user_request": user_request,
            "preview": preview,
            "graph_result": graph_result,
            "options": options,
        }
    except Exception as exc:
        result_holder["status"] = "error"
        result_holder["error"] = str(exc)


def _extract_sandbox_outputs(sandbox_result) -> tuple[list[dict[str, Any]], bool, Any]:
    tables, truncated = extract_result_tables(sandbox_result)
    stored: list[dict[str, Any]] = []
    for table in tables:
        try:
            stored.append(
                {
                    "name": table.name,
                    "filename": table.filename,
                    "csv_bytes": dataframe_to_csv_bytes(table.dataframe),
                    "dataframe": table.dataframe,
                    "shape": list(table.dataframe.shape),
                }
            )
        except ValueError:
            continue
    scalar = None
    if sandbox_result is not None and sandbox_result.success and not stored:
        raw = sandbox_result.result
        if not isinstance(raw, (pd.DataFrame, pd.Series, dict, list)):
            scalar = raw
    return stored, truncated, scalar


def _maybe_generate_report_for_task(
    *,
    user_request: str,
    preview: dict[str, Any],
    sandbox_result,
    generated_code: str,
    options: dict[str, Any],
    result: dict[str, Any],
) -> None:
    if not options.get("enable_report"):
        return
    try:
        report = generate_markdown_report(
            user_request,
            preview,
            sandbox_result,
            generated_code=generated_code,
            chart_paths=[],
            save_to_file=True,
            report_title="analysis_report",
        )
        result["report_markdown"] = report.markdown
        result["report_path"] = str(report.saved_path) if report.saved_path else None
    except ReportGenerationError as exc:
        if options.get("use_fallback_report"):
            fallback = build_fallback_report(
                user_request,
                preview,
                sandbox_result,
                generated_code=generated_code,
                chart_paths=[],
            )
            try:
                saved = save_report_markdown(fallback, title="analysis_report")
                result["report_markdown"] = fallback
                result["report_path"] = str(saved)
                result["error_message"] = f"LLM 报告失败，已使用模板报告：{exc}"
            except Exception as save_exc:
                result["report_markdown"] = fallback
                result["error_message"] = f"报告保存失败：{save_exc}"
        else:
            result["error_message"] = f"报告生成失败：{exc}"


def _populate_task_result(
    task: dict[str, Any],
    *,
    user_request: str,
    preview: dict[str, Any],
    graph_result,
    options: dict[str, Any],
) -> None:
    result = task["result"]
    result["generated_code"] = graph_result.generated_code or ""
    result["retry_history"] = list(graph_result.retry_history or [])
    result["graph_attempts"] = graph_result.total_attempts
    result["sandbox_ok"] = graph_result.sandbox_result

    if graph_result.code_generation_error:
        result["error_message"] = f"代码生成失败：{graph_result.code_generation_error}"
        result["success"] = False
        task["status"] = "failed"
        return

    sandbox_result = graph_result.sandbox_result
    if sandbox_result is None:
        result["error_message"] = "沙箱未返回执行结果。"
        result["success"] = False
        task["status"] = "failed"
        return

    if not graph_result.success:
        result["error_message"] = (
            f"执行失败 [{sandbox_result.error_type}]：{sandbox_result.error}"
        )
        result["success"] = False
        task["status"] = "failed"
        _maybe_generate_report_for_task(
            user_request=user_request,
            preview=preview,
            sandbox_result=sandbox_result,
            generated_code=graph_result.generated_code,
            options=options,
            result=result,
        )
        return

    stored, truncated, scalar = _extract_sandbox_outputs(sandbox_result)
    result["result_tables"] = stored
    result["result_truncated"] = truncated
    result["result_scalar"] = scalar
    result["success"] = True
    task["status"] = "completed"
    _maybe_generate_report_for_task(
        user_request=user_request,
        preview=preview,
        sandbox_result=sandbox_result,
        generated_code=graph_result.generated_code,
        options=options,
        result=result,
    )


def start_next_queued_task(
    task: dict[str, Any],
    datasets: list[DatasetInfo],
) -> None:
    preview = task["preview"]
    df_dict = datasets_to_dict(datasets)
    holder: dict[str, Any] = {"status": "running"}
    task["holder"] = holder
    task["status"] = "running"
    task["started_at"] = time.time()
    worker = threading.Thread(
        target=_run_analysis_worker,
        args=(
            task["user_request"],
            task["options"],
            preview,
            df_dict,
            holder,
        ),
        daemon=True,
    )
    worker.start()


def finalize_running_task(task: dict[str, Any]) -> bool:
    """若运行中任务已结束则写入结果。返回是否发生状态变化。"""
    holder = task.get("holder")
    if not holder:
        return False
    worker_status = holder.get("status", "running")
    if worker_status == "running":
        return False

    task["finished_at"] = time.time()
    task["holder"] = None

    if task.get("cancel_requested"):
        task["status"] = "cancelled"
        task["result"]["error_message"] = "任务已取消。"
        return True

    if worker_status == "error":
        task["status"] = "failed"
        task["result"]["error_message"] = str(holder.get("error", "未知错误"))
        return True

    payload = holder.get("payload") or {}
    _populate_task_result(
        task,
        user_request=str(payload.get("user_request", task["user_request"])),
        preview=payload.get("preview") or task["preview"],
        graph_result=payload["graph_result"],
        options=payload.get("options") or task["options"],
    )
    return True


def advance_task_queue(session_state: Any, datasets: list[DatasetInfo] | None) -> bool:
    """推进队列：收尾运行中任务、启动下一个排队任务。返回是否需要刷新页面。"""
    tasks: list[dict[str, Any]] = session_state.analysis_tasks
    changed = False

    running = get_running_task(tasks)
    if running is not None:
        if finalize_running_task(running):
            changed = True
            if running["status"] in ("completed", "failed"):
                if session_state.selected_task_id is None:
                    session_state.selected_task_id = running["id"]
        return changed

    if datasets is None:
        return changed

    for task in tasks:
        if task["status"] != "queued":
            continue
        if task.get("cancel_requested"):
            task["status"] = "cancelled"
            task["finished_at"] = time.time()
            changed = True
            continue
        start_next_queued_task(task, datasets)
        changed = True
        break
    return changed


def request_cancel_task(task: dict[str, Any]) -> None:
    task["cancel_requested"] = True
    if task["status"] == "queued":
        task["status"] = "cancelled"
        task["finished_at"] = time.time()
        task["result"]["error_message"] = "任务已取消。"


def remove_task(session_state: Any, task_id: str) -> None:
    tasks: list[dict[str, Any]] = session_state.analysis_tasks
    session_state.analysis_tasks = [t for t in tasks if t["id"] != task_id]
    if session_state.selected_task_id == task_id:
        session_state.selected_task_id = None
        for task in reversed(session_state.analysis_tasks):
            if task["status"] in ("completed", "failed"):
                session_state.selected_task_id = task["id"]
                break


def task_summary_label(task: dict[str, Any], max_len: int = 28) -> str:
    text = task["user_request"].strip().replace("\n", " ")
    if len(text) > max_len:
        text = text[: max_len - 1] + "…"
    return text or "（空需求）"


__all__ = [
    "STATUS_LABELS",
    "advance_task_queue",
    "enqueue_analysis_task",
    "get_running_task",
    "get_selected_task",
    "get_task_by_id",
    "init_task_state",
    "remove_task",
    "request_cancel_task",
    "task_summary_label",
]
