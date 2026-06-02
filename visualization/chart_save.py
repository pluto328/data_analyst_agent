"""图表持久化：Matplotlib PNG 与 Plotly HTML。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from utils.logger import get_logger
from utils.path_helper import CHART_SUBDIR, build_temp_file_path, delete_path, ensure_subdirectory

log = get_logger()

PNG_SUFFIX: str = ".png"
HTML_SUFFIX: str = ".html"
DEFAULT_DPI: int = 150


def get_chart_dir() -> Path:
    """返回图表输出目录 ``temp_files/charts``。"""
    try:
        return ensure_subdirectory(CHART_SUBDIR)
    except OSError as exc:
        raise OSError(f"Cannot create chart directory: {CHART_SUBDIR}") from exc


def build_chart_output_path(
    chart_name: str,
    *,
    suffix: str = PNG_SUFFIX,
    prefix: str = "chart",
) -> Path:
    """在 charts 目录下生成唯一输出路径。"""
    safe_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    filename = chart_name if chart_name.endswith(safe_suffix) else f"{chart_name}{safe_suffix}"
    return build_temp_file_path(CHART_SUBDIR, filename, prefix=prefix)


def close_figure(fig: Figure) -> None:
    """关闭 Matplotlib 图像，释放 Windows 文件句柄。"""
    try:
        plt.close(fig)
    except Exception:
        log.exception("Failed to close matplotlib figure")


def save_matplotlib_figure(
    fig: Figure,
    chart_name: str,
    *,
    title: str | None = None,
    dpi: int = DEFAULT_DPI,
    prefix: str = "chart",
) -> Path:
    """保存 Matplotlib 图表为 PNG，并在保存后关闭 figure。"""
    if not isinstance(fig, Figure):
        raise TypeError("fig must be a matplotlib Figure.")

    output_name = title or chart_name
    output_path = build_chart_output_path(output_name, suffix=PNG_SUFFIX, prefix=prefix)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        log.info("Saved matplotlib chart: {}", output_path.name)
        return output_path
    except Exception as exc:
        raise OSError(f"Failed to save matplotlib chart: {output_path}") from exc
    finally:
        close_figure(fig)


def save_plotly_figure(
    fig: Any,
    chart_name: str,
    *,
    title: str | None = None,
    prefix: str = "chart",
) -> Path:
    """保存 Plotly 图表为 HTML。"""
    if not hasattr(fig, "write_html"):
        raise TypeError("fig must be a Plotly figure with write_html().")

    output_name = title or chart_name
    output_path = build_chart_output_path(output_name, suffix=HTML_SUFFIX, prefix=prefix)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path), include_plotlyjs="cdn")
        log.info("Saved plotly chart: {}", output_path.name)
        return output_path
    except Exception as exc:
        raise OSError(f"Failed to save plotly chart: {output_path}") from exc


def save_figure(
    fig: Any,
    chart_name: str,
    *,
    title: str | None = None,
    dpi: int = DEFAULT_DPI,
    prefix: str = "chart",
) -> Path:
    """自动识别 Matplotlib / Plotly 并保存。"""
    if isinstance(fig, Figure):
        return save_matplotlib_figure(
            fig,
            chart_name,
            title=title,
            dpi=dpi,
            prefix=prefix,
        )
    if hasattr(fig, "write_html"):
        return save_plotly_figure(fig, chart_name, title=title, prefix=prefix)
    raise TypeError("Unsupported figure type for save_figure().")


def _run_self_check() -> None:
    """模块内置自检。"""
    import pandas as pd
    import plotly.express as px

    from visualization.chart_builder import build_line_chart

    print("=== chart_save self-check ===")
    print(f"chart_dir: {get_chart_dir()}")

    sample = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    fig = build_line_chart(sample, "x", "y", title="Plotly前置测试")
    png_path = save_matplotlib_figure(fig, "line_demo", title="折线保存测试")
    print(f"matplotlib: {png_path.name} ({png_path.stat().st_size} bytes)")

    plotly_fig = px.line(sample, x="x", y="y", title="交互折线")
    html_path = save_plotly_figure(plotly_fig, "plotly_demo", title="交互图测试")
    print(f"plotly:     {html_path.name} ({html_path.stat().st_size} bytes)")

    auto_path = save_figure(
        build_line_chart(sample, "x", "y", title="自动识别"),
        "auto_demo",
        title="自动保存测试",
    )
    print(f"auto:       {auto_path.name}")

    for path in {png_path, html_path, auto_path}:
        try:
            delete_path(path)
        except OSError as exc:
            print(f"cleanup failed: {path} -> {exc}")

    print("=== done ===")


__all__ = [
    "DEFAULT_DPI",
    "HTML_SUFFIX",
    "PNG_SUFFIX",
    "build_chart_output_path",
    "close_figure",
    "get_chart_dir",
    "save_figure",
    "save_matplotlib_figure",
    "save_plotly_figure",
]


if __name__ == "__main__":
    _run_self_check()
