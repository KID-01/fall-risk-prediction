"""
萤石开放平台 SDK 集成模块

提供 EzvizClient 类，封装：
- AccessToken 自动获取与刷新
- 设备列表查询
- 视频流地址获取
- 云台控制
"""
from .client import EzvizClient

__all__ = ["EzvizClient"]
