"""可视化图表模块。

基于 Matplotlib / Seaborn / Plotly 生成分析图表，统一处理 Windows 下 SimHei 中文显示；
负责将图表保存至 ``temp_files`` 供 Streamlit 展示与报告引用。

- ``chart_builder``：折线图、柱状图、箱线图等常用图表构建
- ``chart_save``：静态图片与交互式图表持久化
"""

from __future__ import annotations

__all__ = [
    "chart_builder",
    "chart_save",
]


def __getattr__(name: str):
    """延迟加载子模块，避免 ``python -m visualization.chart_builder`` 双重导入警告。"""
    if name == "chart_builder":
        from . import chart_builder as module

        return module
    if name == "chart_save":
        from . import chart_save as module

        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
