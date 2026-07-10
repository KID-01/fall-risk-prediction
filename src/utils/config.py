"""
配置加载工具 — 全局单例,使用 OmegaConf 加载 YAML 配置
"""
from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "base.yaml"
_config = None


def get_config():
    """获取全局配置单例"""
    global _config
    if _config is None:
        _config = OmegaConf.load(_CONFIG_PATH)
    return _config


def reload_config():
    """重新加载配置(修改yaml后调用)"""
    global _config
    _config = OmegaConf.load(_CONFIG_PATH)
    return _config
