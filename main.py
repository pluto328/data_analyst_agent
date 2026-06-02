"""Streamlit 入口：上传表格、生成代码、沙箱执行、可视化与报告。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from agent.analysis_graph import run_analysis_graph
from agent.report_generator import (
    ReportGenerationError,
    build_fallback_report,
    generate_markdown_report,
    save_report_markdown,
)
from config.settings import (
    MAX_UPLOAD_MB,
    OPENAI_MODEL,
    SANDBOX_TIMEOUT_SEC,
    ensure_temp_dir,
)
from utils.file_parser import parse_uploaded_file
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


def _init_session_state() -> None:
    defaults = {
        "parsed": None,
        "upload_path": None,
        "generated_code": "",
        "sandbox_ok": None,
        "chart_paths": [],
        "report_markdown": "",
        "report_path": None,
        "retry_history": [],
        "graph_attempts": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _preview_without_df(parsed: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in parsed.items() if k != "dataframe"}


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


def _render_sidebar() -> dict[str, Any]:
    st.sidebar.header("分析设置")
    st.sidebar.caption(f"模型：`{OPENAI_MODEL}` · 沙箱超时：{SANDBOX_TIMEOUT_SEC}s")
    enable_chart = st.sidebar.checkbox("生成可视化图表", value=True)
    chart_type = st.sidebar.selectbox("图表类型", SUPPORTED_CHART_TYPES, index=1)
    enable_report = st.sidebar.checkbox("生成 Markdown 报告", value=True)
    use_fallback_report = st.sidebar.checkbox(
        "报告 LLM 失败时使用模板降级",
        value=True,
    )
    return {
        "enable_chart": enable_chart,
        "chart_type": chart_type,
        "enable_report": enable_report,
        "use_fallback_report": use_fallback_report,
    }


def _render_upload_section() -> None:
    st.subheader("1. 上传数据")
    uploaded = st.file_uploader(
        "支持 CSV / XLS / XLSX",
        type=["csv", "xls", "xlsx"],
        help=f"单文件不超过 {MAX_UPLOAD_MB} MB",
    )
    if uploaded is None:
        return

    try:
        path = _save_uploaded_file(uploaded)
        parsed = parse_uploaded_file(path)
        st.session_state.upload_path = str(path)
        st.session_state.parsed = parsed
        st.success(f"已加载：`{parsed['filename']}`（{parsed['shape'][0]} 行 × {parsed['shape'][1]} 列）")
    except Exception as exc:
        st.session_state.parsed = None
        st.error(f"文件解析失败：{exc}")
        return

    preview = _preview_without_df(st.session_state.parsed)
    with st.expander("数据预览", expanded=True):
        #只展示前15行
        st.dataframe(st.session_state.parsed["dataframe"].head(15), use_container_width=True)


def _render_analysis_section(options: dict[str, Any]) -> None:
    parsed = st.session_state.parsed
    if not parsed:
        st.info("请先上传并解析数据文件。")
        return

    st.subheader("2. 描述分析需求")
    user_request = st.text_area(
        "用自然语言描述你想做的清洗或分析",
        placeholder="例如：删除 amount 为空的行，按 category 汇总 value 并求和",
        height=120,
    )

    df: pd.DataFrame = parsed["dataframe"]
    columns = [str(c) for c in df.columns]
    x_col, y_col = columns[0], columns[0]
    if len(columns) >= 2:
        x_col, y_col = _pick_default_columns(df)
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("图表 X 轴列", columns, index=columns.index(x_col))
        with col2:
            y_col = st.selectbox("图表 Y 轴列", columns, index=columns.index(y_col))
    else:
        st.caption("数据仅 1 列，将跳过图表绘制。")

    if st.button("开始分析", type="primary", use_container_width=True):
        if not user_request.strip():
            st.error("请填写分析需求。")
            return
        _run_analysis_pipeline(
            user_request=user_request.strip(),
            options=options,
            x_col=x_col,
            y_col=y_col,
        )


def _run_analysis_pipeline(
    *,
    user_request: str,
    options: dict[str, Any],
    x_col: str,
    y_col: str,
) -> None:
    parsed = st.session_state.parsed
    preview = _preview_without_df(parsed)
    df: pd.DataFrame = parsed["dataframe"]

    st.session_state.chart_paths = []
    st.session_state.report_markdown = ""
    st.session_state.report_path = None

    with st.status("分析进行中...", expanded=True) as status:
        st.write("运行 LangGraph 工作流（生成代码 ↔ 沙箱执行，失败最多回溯 3 次）...")
        try:
            graph_result = run_analysis_graph(user_request, preview, df)
        except Exception as exc:
            status.update(label="分析失败", state="error")
            st.error(f"工作流异常：{exc}")
            return

        st.session_state.generated_code = graph_result.generated_code
        st.session_state.sandbox_ok = graph_result.sandbox_result
        st.session_state.retry_history = graph_result.retry_history
        st.session_state.graph_attempts = graph_result.total_attempts

        if graph_result.model:
            st.write(f"模型：{graph_result.model} · 共尝试 {graph_result.total_attempts} 轮")

        if graph_result.retry_history:
            # #region agent log
            try:
                import json
                import time
                from pathlib import Path as _Path

                _Path("debug-014dc4.log").open("a", encoding="utf-8").write(
                    json.dumps(
                        {
                            "sessionId": "014dc4",
                            "hypothesisId": "A",
                            "location": "main.py:retry-in-status",
                            "message": "render retry summary inside st.status (no nested expander)",
                            "data": {"retry_count": len(graph_result.retry_history)},
                            "timestamp": int(time.time() * 1000),
                            "runId": "post-fix",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            except Exception:
                pass
            # #endregion
            st.warning(
                f"经历 {len(graph_result.retry_history)} 次执行失败后回溯重试"
            )
            # st.status 内部也是 expander，不可再嵌套 st.expander
            for record in graph_result.retry_history:
                st.caption(
                    f"第 {record['attempt']} 次失败："
                    f"{record.get('error_type')} - {record.get('error')}"
                )

        if graph_result.code_generation_error:
            status.update(label="代码生成失败", state="error")
            st.error(f"代码生成失败：{graph_result.code_generation_error}")
            return

        if graph_result.generated_code:
            st.code(graph_result.generated_code, language="python")

        sandbox_result = graph_result.sandbox_result
        if sandbox_result is None:
            status.update(label="分析失败", state="error")
            st.error("沙箱未返回执行结果。")
            return

        if not graph_result.success:
            status.update(label="执行失败", state="error")
            st.error(
                f"已达最大重试次数，仍执行失败 "
                f"[{sandbox_result.error_type}]：{sandbox_result.error}"
            )
            _render_execution_results(sandbox_result)
            _maybe_generate_report(
                user_request,
                preview,
                sandbox_result,
                graph_result.generated_code,
                options,
            )
            return

        st.write("代码执行成功")
        chart_paths: list[str] = []

        # 3. 可视化
        if options["enable_chart"] and len(df.columns) >= 2:
            st.write("生成图表...")
            chart_df = sandbox_result.df if sandbox_result.df is not None else df
            try:
                chart_path = _build_chart(
                    chart_df,
                    options["chart_type"],
                    x_col,
                    y_col,
                    title="分析图表",
                )
                if chart_path:
                    chart_paths.append(str(chart_path))
            except ValueError as exc:
                st.warning(str(exc))

        st.session_state.chart_paths = chart_paths

        # 4. 报告
        if options["enable_report"]:
            st.write("生成分析报告...")
            _maybe_generate_report(
                user_request,
                preview,
                sandbox_result,
                graph_result.generated_code,
                options,
                chart_paths=chart_paths,
            )

        status.update(label="分析完成", state="complete")

    _render_results_panel()


def _maybe_generate_report(
    user_request: str,
    preview: dict[str, Any],
    sandbox_result,
    generated_code: str,
    options: dict[str, Any],
    *,
    chart_paths: list[str] | None = None,
) -> None:
    paths = chart_paths if chart_paths is not None else st.session_state.chart_paths
    try:
        report = generate_markdown_report(
            user_request,
            preview,
            sandbox_result,
            generated_code=generated_code,
            chart_paths=paths,
            save_to_file=True,
            report_title="analysis_report",
        )
        st.session_state.report_markdown = report.markdown
        st.session_state.report_path = (
            str(report.saved_path) if report.saved_path else None
        )
    except ReportGenerationError as exc:
        if options.get("use_fallback_report"):
            fallback = build_fallback_report(
                user_request,
                preview,
                sandbox_result,
                generated_code=generated_code,
                chart_paths=paths,
            )
            try:
                saved = save_report_markdown(fallback, title="analysis_report")
                st.session_state.report_markdown = fallback
                st.session_state.report_path = str(saved)
                st.warning(f"LLM 报告失败，已使用模板报告：{exc}")
            except Exception as save_exc:
                st.session_state.report_markdown = fallback
                st.warning(f"报告保存失败：{save_exc}")
        else:
            st.error(f"报告生成失败：{exc}")


def _render_execution_results(sandbox_result) -> None:
    if sandbox_result.stdout:
        st.text("标准输出")
        st.code(sandbox_result.stdout)


def _render_results_panel() -> None:
    st.subheader("3. 分析结果")

    code = st.session_state.generated_code
    if code:
        st.markdown("**生成的代码**")
        st.code(code, language="python")

    retry_history = st.session_state.get("retry_history") or []
    if retry_history:
        with st.expander("回溯重试历史", expanded=False):
            for record in retry_history:
                st.markdown(f"**第 {record['attempt']} 次失败**")
                st.caption(
                    f"{record.get('error_type')} - {record.get('error')}"
                )
                st.code(record.get("code", ""), language="python")

    sandbox_result = st.session_state.sandbox_ok
    if sandbox_result is None:
        return

    _render_execution_results(sandbox_result)

    if sandbox_result.success:
        st.markdown("**执行结果**")
        result = sandbox_result.result
        if isinstance(result, pd.DataFrame):
            st.dataframe(result, use_container_width=True)
        elif isinstance(result, pd.Series):
            st.dataframe(result.to_frame(), use_container_width=True)
        else:
            st.write(result)

        if sandbox_result.df is not None:
            st.markdown("**输出数据表（df）**")
            st.dataframe(sandbox_result.df, use_container_width=True)

    chart_paths = st.session_state.chart_paths or []
    if chart_paths:
        st.markdown("**可视化图表**")
        for path_str in chart_paths:
            path = Path(path_str)
            if path.suffix.lower() == ".png" and path.is_file():
                st.image(str(path), caption=path.name)
            else:
                st.write(path.name)

    report_md = st.session_state.report_markdown
    if report_md:
        st.markdown("**分析报告**")
        st.markdown(report_md)
        st.download_button(
            label="下载 Markdown 报告",
            data=report_md,
            file_name="analysis_report.md",
            mime="text/markdown",
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
        "上传表格 → 自然语言描述需求 → 自动生成并安全执行 Pandas 代码 → 图表与报告"
    )

    options = _render_sidebar()
    _render_upload_section()
    _render_analysis_section(options)


if __name__ == "__main__":
    main()