"""Streamlit 入口：上传表格、生成代码、沙箱执行、可视化与报告。"""

from __future__ import annotations

import os

# Windows/Conda：numpy、matplotlib 等重复链接 OpenMP 时的兼容项（见 Intel OMP Error #15）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from agent.analysis_tasks import (
    STATUS_LABELS,
    advance_task_queue,
    enqueue_analysis_task,
    get_running_task,
    get_selected_task,
    init_task_state,
    remove_task,
    request_cancel_task,
    task_summary_label,
)
from agent.dataset_registry import DatasetInfo
from config.settings import (
    MAX_OUTPUT_TABLES,
    MAX_TOTAL_UPLOAD_BYTES,
    MAX_TOTAL_UPLOAD_MB,
    MAX_UPLOAD_FILES,
    MAX_UPLOAD_MB,
    OPENAI_MODEL,
    SANDBOX_TIMEOUT_SEC,
    ensure_temp_dir,
)
from agent.correction_store import correction_record_count, ensure_correction_records_loaded
from utils.file_parser import parse_uploaded_files
from utils.logger import setup_logger
from utils.path_helper import (
    UPLOAD_SUBDIR,
    build_temp_file_path,
    validate_upload_filename,
)
from visualization.chart_builder import SUPPORTED_CHART_TYPES, build_chart
from visualization.chart_save import save_matplotlib_figure

setup_logger()
ensure_temp_dir()
try:
    ensure_correction_records_loaded()
except Exception:
    pass

_CHART_TYPE_LABELS: dict[str, str] = {
    "line": "折线图",
    "bar": "柱状图",
    "box": "箱线图",
}

if hasattr(st, "fragment"):
    _st_fragment = st.fragment
elif hasattr(st, "experimental_fragment"):
    _st_fragment = st.experimental_fragment
else:

    def _st_fragment(**_kwargs: Any):
        def decorator(func):
            return func

        return decorator

if hasattr(st, "dialog"):
    _st_dialog = st.dialog
elif hasattr(st, "experimental_dialog"):
    _st_dialog = st.experimental_dialog
else:

    def _st_dialog(_title: str):
        def decorator(func):
            return func

        return decorator


def _init_session_state() -> None:
    defaults = {
        "datasets": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    init_task_state(st.session_state)


def _dataset_filenames(datasets: list[DatasetInfo]) -> str:
    return "、".join(f"「{ds.filename}」" for ds in datasets)


def _select_chart_dataframe(
    datasets: list[DatasetInfo],
    *,
    panel_key: str,
) -> pd.DataFrame:
    if len(datasets) == 1:
        return datasets[0].dataframe
    labels = [ds.filename for ds in datasets]
    selected = st.selectbox(
        "选择要绘图的文件",
        labels,
        key=f"chart_file_{panel_key}",
    )
    for ds in datasets:
        if ds.filename == selected:
            return ds.dataframe
    return datasets[0].dataframe


def _chart_cache_key(panel_key: str) -> str:
    return f"chart_cache_{panel_key}"


def _chart_gen_id_key(panel_key: str) -> str:
    return f"chart_gen_id_{panel_key}"


def _render_result_tables_section(result: dict[str, Any]) -> None:
    """展示多张输出表及各自 CSV 下载。"""
    tables: list[dict[str, Any]] = result.get("result_tables") or []
    truncated = result.get("result_truncated", False)
    scalar = result.get("result_scalar")

    if truncated:
        st.warning(
            f"输出表超过上限，仅展示前 {MAX_OUTPUT_TABLES} 张"
            f"（可在 .env 调整 MAX_OUTPUT_TABLES）。"
        )

    if tables:
        st.caption(f"共 {len(tables)} 张输出表（命名来自代码 result 字典键名）")
        if len(tables) == 1:
            item = tables[0]
            st.dataframe(item["dataframe"], use_container_width=True)
            st.download_button(
                label=f"下载 {item['filename']}",
                data=item["csv_bytes"],
                file_name=item["filename"],
                mime="text/csv",
                key=f"download_result_{item['filename']}",
            )
        else:
            tabs = st.tabs([str(item["name"]) for item in tables])
            for tab, item in zip(tabs, tables):
                with tab:
                    shape = item.get("shape") or item["dataframe"].shape
                    st.caption(f"{item['filename']} · {shape[0]} 行 × {shape[1]} 列")
                    st.dataframe(item["dataframe"], use_container_width=True)
                    st.download_button(
                        label=f"下载 {item['filename']}",
                        data=item["csv_bytes"],
                        file_name=item["filename"],
                        mime="text/csv",
                        key=f"download_result_{item['filename']}",
                    )
    elif scalar is not None:
        st.write(scalar)


def _result_chart_dataframe(result: dict[str, Any]) -> pd.DataFrame | None:
    """从任务结果中的输出表选取可绘图 DataFrame。"""
    tables: list[dict[str, Any]] = result.get("result_tables") or []
    if not tables:
        return None
    if len(tables) == 1:
        return tables[0]["dataframe"]
    labels = [str(item["name"]) for item in tables]
    selected = st.selectbox(
        "选择要绘图的输出表",
        labels,
        key="result_chart_table_pick",
    )
    for item in tables:
        if item["name"] == selected:
            return item["dataframe"]
    return tables[0]["dataframe"]


def _save_uploaded_file(uploaded_file) -> Path:
    safe_name = validate_upload_filename(uploaded_file.name)
    if uploaded_file.size > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"文件超过 {MAX_UPLOAD_MB} MB 限制。")
    target = build_temp_file_path(UPLOAD_SUBDIR, safe_name, prefix="upload")
    try:
        target.write_bytes(uploaded_file.getvalue())
    except OSError as exc:
        raise OSError(f"无法保存上传文件: {target}") from exc
    return target


def _pick_default_columns(df: pd.DataFrame) -> tuple[str, str]:
    columns = list(df.columns)
    if len(columns) < 2:
        raise ValueError("至少需要 2 列才能绘图。")
    numeric = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
    non_numeric = [c for c in columns if c not in numeric]
    x_col = non_numeric[0] if non_numeric else columns[0]
    y_col = numeric[0] if numeric else columns[1]
    if x_col == y_col and len(columns) > 1:
        y_col = columns[1]
    return str(x_col), str(y_col)


def _build_chart(
    df: pd.DataFrame,
    chart_type: str,
    x_col: str,
    y_col: str,
    title: str,
) -> Path | None:
    try:
        fig = build_chart(chart_type, df, x_col, y_col, title=title)
        return save_matplotlib_figure(fig, chart_type, title=title)
    except Exception as exc:
        st.warning(f"图表生成失败：{exc}")
        return None


def _show_chart_image(chart_path: Path) -> None:
    """展示图表 PNG；兼容 Streamlit 1.36（无 use_container_width）。"""
    try:
        st.image(str(chart_path), use_container_width=True)
    except TypeError:
        st.image(str(chart_path), use_column_width=True)


def _show_chart_with_actions(cache: dict[str, Any]) -> None:
    """展示图表，并提供 PNG 下载（浏览器中可右键复制图片）。"""
    path = Path(cache["path"])
    png_bytes = cache.get("bytes")
    if png_bytes is None and path.is_file():
        png_bytes = path.read_bytes()
    filename = str(cache.get("filename") or path.name)
    if path.is_file():
        _show_chart_image(path)
    if png_bytes:
        st.download_button(
            label=f"下载图表 {filename}",
            data=png_bytes,
            file_name=filename,
            mime="image/png",
            key=f"download_chart_{filename}_{cache.get('gen_id', 0)}",
        )
        st.caption("提示：可下载 PNG，或在图表上右键「复制图片」/「另存为」。")


def _render_standalone_chart_panel(
    *,
    panel_key: str,
    df: pd.DataFrame,
    title: str,
) -> None:
    """独立图表区：手动点击生成，不自动出图；可随时清除或重新选坐标。"""
    columns = [str(c) for c in df.columns]
    if len(columns) < 2:
        st.caption("至少需要 2 列才能绘图。")
        return

    st.caption("选择类型与坐标后点击「生成图表」；不会自动出图，与分析互不阻塞。")
    x_default, y_default = _pick_default_columns(df)
    col1, col2, col3 = st.columns(3)
    with col1:
        chart_type = st.selectbox(
            "图表类型",
            SUPPORTED_CHART_TYPES,
            index=1,
            format_func=lambda value: _CHART_TYPE_LABELS.get(value, value),
            key=f"chart_type_{panel_key}",
        )
    with col2:
        x_col = st.selectbox(
            "X 轴列",
            columns,
            index=columns.index(x_default),
            key=f"chart_x_{panel_key}",
        )
    with col3:
        y_col = st.selectbox(
            "Y 轴列",
            columns,
            index=columns.index(y_default),
            key=f"chart_y_{panel_key}",
        )

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        generate = st.button("生成图表", key=f"chart_generate_{panel_key}")
    with btn_col2:
        clear = st.button("清除图表", key=f"chart_clear_{panel_key}")

    gen_id_key = _chart_gen_id_key(panel_key)
    cache_key = _chart_cache_key(panel_key)

    if clear:
        st.session_state[gen_id_key] = st.session_state.get(gen_id_key, 0) + 1
        st.session_state.pop(cache_key, None)

    if generate:
        st.session_state[gen_id_key] = st.session_state.get(gen_id_key, 0) + 1
        current_gen = st.session_state[gen_id_key]
        with st.spinner("正在生成图表…"):
            chart_path = _build_chart(df, chart_type, x_col, y_col, title=title)
        if current_gen != st.session_state.get(gen_id_key):
            return
        if chart_path is not None and chart_path.is_file():
            try:
                st.session_state[cache_key] = {
                    "path": str(chart_path),
                    "bytes": chart_path.read_bytes(),
                    "filename": chart_path.name,
                    "gen_id": current_gen,
                }
            except OSError as exc:
                st.warning(f"图表读取失败：{exc}")

    cache = st.session_state.get(cache_key)
    if cache:
        _show_chart_with_actions(cache)


def _render_upload_chart_section(datasets: list[DatasetInfo]) -> None:
    st.subheader("2. 数据可视化")
    chart_df = _select_chart_dataframe(datasets, panel_key="upload")
    _render_standalone_chart_panel(
        panel_key="upload",
        df=chart_df,
        title="原始数据图表",
    )


def _render_sidebar_settings() -> dict[str, Any]:
    st.sidebar.caption(f"模型：`{OPENAI_MODEL}` · 沙箱超时：{SANDBOX_TIMEOUT_SEC}s")
    st.sidebar.caption(f"已加载改错记录：{correction_record_count()} 条")
    st.sidebar.caption(f"输出表上限：{MAX_OUTPUT_TABLES} 张")
    enable_report = st.sidebar.checkbox("生成 Markdown 报告", value=True)
    use_fallback_report = st.sidebar.checkbox(
        "报告 LLM 失败时使用模板降级",
        value=True,
    )
    return {
        "enable_report": enable_report,
        "use_fallback_report": use_fallback_report,
    }


@_st_dialog("确认关闭任务")
def _confirm_close_task_dialog(task_id: str) -> None:
    tasks: list[dict[str, Any]] = st.session_state.analysis_tasks
    task = next((item for item in tasks if item["id"] == task_id), None)
    if task is None:
        st.session_state.pending_close_task_id = None
        return
    st.write(f"确定从列表中移除此任务？")
    st.caption(f"「{task_summary_label(task, 60)}」")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("确认关闭", type="primary", use_container_width=True, key="dialog_close_ok"):
            remove_task(st.session_state, task_id)
            st.session_state.pending_close_task_id = None
            st.rerun()
    with col2:
        if st.button("取消", use_container_width=True, key="dialog_close_cancel"):
            st.session_state.pending_close_task_id = None
            st.rerun()


def _render_sidebar_tasks() -> None:
    tasks: list[dict[str, Any]] = st.session_state.analysis_tasks
    st.sidebar.header("分析任务")
    if not tasks:
        st.sidebar.caption("暂无任务。填写需求并点击「开始分析」后任务会出现在这里。")
        return

    for task in reversed(tasks):
        status = str(task["status"])
        label = STATUS_LABELS.get(status, status)
        if task.get("cancel_requested") and status == "running":
            label = "取消中"
        summary = task_summary_label(task)
        item_key = task["id"]

        if status in ("completed", "failed"):
            selected_mark = " ✓" if st.session_state.selected_task_id == item_key else ""
            if st.sidebar.button(
                f"{label} · {summary}{selected_mark}",
                key=f"task_select_{item_key}",
                use_container_width=True,
            ):
                st.session_state.selected_task_id = item_key
                st.rerun()
        else:
            st.sidebar.markdown(f"**{label}** · {summary}")

        action_col1, action_col2 = st.sidebar.columns(2)
        with action_col1:
            if status in ("queued", "running"):
                if action_col1.button("取消", key=f"task_cancel_{item_key}", use_container_width=True):
                    request_cancel_task(task)
                    st.rerun()
        with action_col2:
            if status in ("completed", "failed", "cancelled"):
                if action_col2.button("关闭", key=f"task_close_{item_key}", use_container_width=True):
                    st.session_state.pending_close_task_id = item_key
                    _confirm_close_task_dialog(item_key)
        st.sidebar.divider()


def _render_sidebar() -> dict[str, Any]:
    _render_sidebar_tasks()
    st.sidebar.header("分析设置")
    options = _render_sidebar_settings()
    return options


@_st_fragment(run_every=1.5)
def _poll_task_queue() -> None:
    datasets: list[DatasetInfo] | None = st.session_state.datasets
    changed = advance_task_queue(st.session_state, datasets)
    if changed:
        st.rerun()


def _render_current_task_status() -> None:
    tasks: list[dict[str, Any]] = st.session_state.analysis_tasks
    running = get_running_task(tasks)
    queued_count = sum(
        1 for task in tasks if task["status"] == "queued" and not task.get("cancel_requested")
    )
    if running:
        st.info(f"**正在处理：** {task_summary_label(running, 80)}")
        st.caption(
            "主界面仍可修改需求、继续排队；上方「数据可视化」与分析入口互不阻塞。"
        )
    if queued_count:
        st.caption(f"排队等待：{queued_count} 个任务")


def _render_upload_section() -> None:
    st.subheader("1. 上传数据")
    uploaded_list = st.file_uploader(
        "支持 CSV / XLS / XLSX（可多选，用于多表关联分析）",
        type=["csv", "xls", "xlsx"],
        accept_multiple_files=True,
        help=(
            f"单文件不超过 {MAX_UPLOAD_MB} MB，"
            f"最多 {MAX_UPLOAD_FILES} 个文件，"
            f"总大小不超过 {MAX_TOTAL_UPLOAD_MB} MB"
        ),
    )

    if uploaded_list:
        if len(uploaded_list) > MAX_UPLOAD_FILES:
            st.error(f"最多上传 {MAX_UPLOAD_FILES} 个文件。")
            return

        try:
            paths: list[Path] = []
            original_names: list[str] = []
            total_size = 0
            for uploaded in uploaded_list:
                if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
                    raise ValueError(f"「{uploaded.name}」超过 {MAX_UPLOAD_MB} MB 限制。")
                total_size += uploaded.size
                original_names.append(uploaded.name)
                paths.append(_save_uploaded_file(uploaded))
            if total_size > MAX_TOTAL_UPLOAD_BYTES:
                raise ValueError(f"总大小超过 {MAX_TOTAL_UPLOAD_MB} MB 限制。")

            datasets = parse_uploaded_files(paths, original_filenames=original_names)
            st.session_state.datasets = datasets

            if len(datasets) == 1:
                ds = datasets[0]
                st.success(
                    f"已加载：`{ds.filename}`（"
                    f"{ds.preview['shape'][0]} 行 × {ds.preview['shape'][1]} 列）"
                )
            else:
                summary = "、".join(f"`{ds.filename}`" for ds in datasets)
                st.success(f"已加载 {len(datasets)} 个文件：{summary}")

        except Exception as exc:
            st.session_state.datasets = None
            st.error(f"文件解析失败：{exc}")
            return

    datasets: list[DatasetInfo] | None = st.session_state.datasets
    if not datasets:
        return

    with st.expander("数据预览", expanded=True):
        if len(datasets) == 1:
            st.dataframe(datasets[0].dataframe.head(15), use_container_width=True)
        else:
            tabs = st.tabs([ds.filename for ds in datasets])
            for tab, ds in zip(tabs, datasets):
                with tab:
                    shape = ds.preview["shape"]
                    st.caption(f"{ds.filename} · {shape[0]} 行 × {shape[1]} 列")
                    st.dataframe(ds.dataframe.head(15), use_container_width=True)

    _render_upload_chart_section(datasets)


def _render_analysis_section(options: dict[str, Any]) -> None:
    datasets: list[DatasetInfo] | None = st.session_state.datasets
    if not datasets:
        st.info("请先上传并解析数据文件。")
        return

    _poll_task_queue()
    _render_current_task_status()

    st.subheader("3. 描述分析需求")
    if len(datasets) > 1:
        f1, f2 = datasets[0].filename, datasets[1].filename
        placeholder = (
            f"例如：将「{f1}」与「{f2}」按 id 关联，"
            "输出「品类汇总」和「清洗明细」两张表"
        )
        st.caption(f"已加载文件：{_dataset_filenames(datasets)}")
    else:
        placeholder = "例如：删除 amount 为空的行，按 category 汇总 value 并求和"
        st.caption(f"已加载文件：{datasets[0].filename}")

    user_request = st.text_area(
        "用自然语言描述你想做的清洗或分析（可直接写文件名）",
        placeholder=placeholder,
        height=120,
        key="user_analysis_request",
    )

    if st.button("开始分析", type="primary", use_container_width=True):
        if not user_request.strip():
            st.error("请填写分析需求。")
        else:
            task = enqueue_analysis_task(
                user_request=user_request.strip(),
                options=options,
                datasets=datasets,
            )
            st.session_state.analysis_tasks.append(task)
            advance_task_queue(st.session_state, datasets)
            st.rerun()


def _render_execution_results(sandbox_result) -> None:
    if sandbox_result.stdout:
        st.text("标准输出")
        st.code(sandbox_result.stdout)


def _render_results_panel() -> None:
    tasks: list[dict[str, Any]] = st.session_state.analysis_tasks
    selected = get_selected_task(tasks, st.session_state.selected_task_id)
    if selected is None or selected["status"] not in ("completed", "failed"):
        return

    result = selected["result"]
    st.subheader("4. 分析结果")
    st.caption(
        f"任务：{task_summary_label(selected, 60)} · "
        f"{STATUS_LABELS.get(selected['status'], selected['status'])}"
    )

    if result.get("error_message") and not result.get("success"):
        st.error(result["error_message"])

    code = result.get("generated_code") or ""
    if code:
        st.markdown("**生成的代码**")
        st.code(code, language="python")

    retry_history = result.get("retry_history") or []
    if retry_history:
        with st.expander("回溯重试历史", expanded=not result.get("success")):
            for record in retry_history:
                st.markdown(f"**第 {record['attempt']} 次失败**")
                st.caption(f"{record.get('error_type')} - {record.get('error')}")
                st.code(record.get("code", ""), language="python")

    sandbox_result = result.get("sandbox_ok")
    if sandbox_result is None:
        return

    _render_execution_results(sandbox_result)

    if sandbox_result.success:
        st.markdown("**执行结果**")
        _render_result_tables_section(result)

        chart_df = _result_chart_dataframe(result)
        if chart_df is not None:
            st.markdown("**处理结果可视化**")
            st.caption("选择坐标后点击「生成图表」，与分析流程无关。")
            _render_standalone_chart_panel(
                panel_key="result",
                df=chart_df,
                title="处理结果图表",
            )

    report_md = result.get("report_markdown") or ""
    if report_md:
        st.markdown("**分析报告**")
        st.markdown(report_md)
        st.download_button(
            label="下载 Markdown 报告",
            data=report_md,
            file_name="analysis_report.md",
            mime="text/markdown",
            key=f"download_report_{selected['id']}",
        )


def main() -> None:
    st.set_page_config(
        page_title="数据分析师 AI Agent",
        page_icon="📊",
        layout="wide",
    )
    _init_session_state()

    st.title("数据分析师 AI Agent")
    st.caption(
        "上传表格（可多表）→ 可选坐标即时出图 → 自然语言分析（支持排队）→ 侧边栏查看历史任务"
    )

    options = _render_sidebar()
    _render_upload_section()
    _render_analysis_section(options)

    if st.session_state.selected_task_id:
        _render_results_panel()


if __name__ == "__main__":
    main()
