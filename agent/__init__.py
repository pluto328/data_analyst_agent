"""AI 智能体核心。

根据用户自然语言提示与已上传表格的 schema / 预览信息，驱动 LangChain 工作流：

- ``code_generator``：生成可在沙箱中执行的 Pandas 数据清洗与分析代码
- ``report_generator``：结合沙箱运行结果与可视化产出，生成 Markdown 数据报告
"""

from __future__ import annotations

__all__ = [
    "analysis_graph",
    "code_generator",
    "report_generator",
]


def __getattr__(name: str):
    """延迟加载子模块，避免 ``python -m agent.*`` 双重导入警告。"""
    if name == "analysis_graph":
        from . import analysis_graph as module

        return module
    if name == "code_generator":
        from . import code_generator as module

        return module
    if name == "report_generator":
        from . import report_generator as module

        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
