"""代码沙箱模块（安全执行层）。

在 RestrictedPython 校验与白名单 globals 约束下，通过子进程隔离运行 AI 生成的代码，
并支持超时终止。禁止执行系统调用、网络访问及非数据处理类操作。

- ``safe_globals``：允许 / 禁用的内置函数与第三方模块白名单
- ``code_sandbox``：代码过滤、编译、子进程执行与结果回收
"""

from __future__ import annotations

from . import code_sandbox, safe_globals

__all__ = [
    "code_sandbox",
    "safe_globals",
]
