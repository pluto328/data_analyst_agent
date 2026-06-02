"""全局配置模块。

集中管理项目常量：路径、沙箱超时、上传文件大小限制、Matplotlib 字体等。
业务代码应通过 ``from config import settings`` 或 ``from config.settings import ...`` 读取配置，
避免在模块内硬编码路径与魔法数字。
"""

from __future__ import annotations

from . import settings

__all__ = [
    "settings",
]
