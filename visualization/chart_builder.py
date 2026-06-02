"""Matplotlib / Seaborn 图表构建（Windows 中文兼容）。"""

from __future__ import annotations

from typing import Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from config.settings import MPL_FONT_FAMILY, MPL_UNICODE_MINUS, configure_matplotlib
from utils.logger import get_logger

log = get_logger()

ChartType = Literal["line", "bar", "box"]
SUPPORTED_CHART_TYPES: tuple[str, ...] = ("line", "bar", "box")


def apply_chart_style() -> None:
    """应用 SimHei 字体（含 Windows 回退）与 Seaborn 主题。"""
    try:
        configure_matplotlib()
        import matplotlib as mpl
        from matplotlib import font_manager

        sns.set_theme(style="whitegrid")

        candidates = [
            MPL_FONT_FAMILY,
            "Microsoft YaHei",
            "SimSun",
            "Arial Unicode MS",
            "DejaVu Sans",
        ]
        available = {item.name for item in font_manager.fontManager.ttflist}
        chosen = [name for name in candidates if name in available]
        if chosen:
            mpl.rcParams["font.sans-serif"] = chosen
            mpl.rcParams["font.family"] = "sans-serif"
        mpl.rcParams["axes.unicode_minus"] = MPL_UNICODE_MINUS
    except Exception:
        log.exception("Failed to apply chart style")


def _validate_dataframe(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        raise ValueError("DataFrame cannot be empty.")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")


def _validate_columns(df: pd.DataFrame, x: str, y: str | None = None) -> None:
    _validate_dataframe(df)
    if x not in df.columns:
        raise ValueError(f"Column not found for x: {x!r}")
    if y is not None and y not in df.columns:
        raise ValueError(f"Column not found for y: {y!r}")


def _create_figure(
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] = (10.0, 6.0),
) -> tuple[Figure, plt.Axes]:
    apply_chart_style()
    fig, ax = plt.subplots(figsize=figsize)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    return fig, ax


def build_line_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] = (10.0, 6.0),
) -> Figure:
    """构建折线图。"""
    _validate_columns(df, x, y)
    fig, ax = _create_figure(
        title=title,
        xlabel=xlabel or x,
        ylabel=ylabel or y,
        figsize=figsize,
    )
    try:
        plot_df = df[[x, y]].dropna()
        ax.plot(plot_df[x], plot_df[y], marker="o", linewidth=2)
        fig.tight_layout()
        log.info("Built line chart: x={}, y={}", x, y)
        return fig
    except Exception as exc:
        plt.close(fig)
        raise ValueError(f"Failed to build line chart: {exc}") from exc


def build_bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] = (10.0, 6.0),
) -> Figure:
    """构建柱状图。"""
    _validate_columns(df, x, y)
    fig, ax = _create_figure(
        title=title,
        xlabel=xlabel or x,
        ylabel=ylabel or y,
        figsize=figsize,
    )
    try:
        plot_df = df[[x, y]].dropna()
        ax.bar(plot_df[x].astype(str), plot_df[y])
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        log.info("Built bar chart: x={}, y={}", x, y)
        return fig
    except Exception as exc:
        plt.close(fig)
        raise ValueError(f"Failed to build bar chart: {exc}") from exc


def build_box_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] = (10.0, 6.0),
) -> Figure:
    """构建箱线图（x 为分类，y 为数值）。"""
    _validate_columns(df, x, y)
    fig, ax = _create_figure(
        title=title,
        xlabel=xlabel or x,
        ylabel=ylabel or y,
        figsize=figsize,
    )
    try:
        plot_df = df[[x, y]].dropna()
        sns.boxplot(data=plot_df, x=x, y=y, ax=ax)
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        log.info("Built box chart: x={}, y={}", x, y)
        return fig
    except Exception as exc:
        plt.close(fig)
        raise ValueError(f"Failed to build box chart: {exc}") from exc


def build_chart(
    chart_type: ChartType,
    df: pd.DataFrame,
    x: str,
    y: str,
    **kwargs,
) -> Figure:
    """按类型分发构建图表。"""
    builders = {
        "line": build_line_chart,
        "bar": build_bar_chart,
        "box": build_box_chart,
    }
    if chart_type not in builders:
        allowed = ", ".join(SUPPORTED_CHART_TYPES)
        raise ValueError(f"Unsupported chart type: {chart_type!r}. Allowed: {allowed}")
    return builders[chart_type](df, x, y, **kwargs)


def _run_self_check() -> None:
    """模块内置自检。"""
    from visualization.chart_save import save_matplotlib_figure

    print("=== chart_builder self-check ===")
    sample = pd.DataFrame(
        {
            "月份": ["1月", "2月", "3月", "4月"],
            "销售额": [120, 150, 90, 180],
            "类别": ["A", "A", "B", "B"],
        }
    )

    checks = [
        ("line", "月份", "销售额", "销售趋势"),
        ("bar", "月份", "销售额", "销售对比"),
        ("box", "类别", "销售额", "类别分布"),
    ]
    saved: list = []
    for chart_type, x_col, y_col, title in checks:
        fig = build_chart(chart_type, sample, x_col, y_col, title=title)
        path = save_matplotlib_figure(fig, chart_type, title=title)
        saved.append(path)
        print(f"  [{chart_type}] saved -> {path.name} ({path.stat().st_size} bytes)")

    for path in saved:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            print(f"  cleanup failed: {path} -> {exc}")

    print("=== done ===")


__all__ = [
    "ChartType",
    "SUPPORTED_CHART_TYPES",
    "apply_chart_style",
    "build_bar_chart",
    "build_box_chart",
    "build_chart",
    "build_line_chart",
]


if __name__ == "__main__":
    _run_self_check()
