"""
配置加载工具 — 全局单例,使用 OmegaConf 加载 YAML 配置
"""
from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
_CONFIG_PATH = _CONFIG_DIR / "base.yaml"
_EZVIZ_CONFIG_PATH = _CONFIG_DIR / "ezviz.yaml"
_config = None
_ezviz_config = None


def get_config():
    """获取全局配置单例（base.yaml）"""
    global _config
    if _config is None:
        _config = OmegaConf.load(_CONFIG_PATH)
    return _config


def get_ezviz_config():
    """
    获取萤石配置单例（ezviz.yaml）

    如果 ezviz.yaml 不存在，返回空配置（不会报错，方便在没有密钥时开发其他模块）
    """
    global _ezviz_config
    if _ezviz_config is None:
        if _EZVIZ_CONFIG_PATH.exists():
            _ezviz_config = OmegaConf.load(_EZVIZ_CONFIG_PATH)
        else:
            # 返回空配置，避免阻塞其他模块的开发
            _ezviz_config = OmegaConf.create({"ezviz": {"app_key": "", "app_secret": ""}})
    return _ezviz_config


def reload_config():
    """重新加载配置(修改yaml后调用)"""
    global _config, _ezviz_config
    _config = OmegaConf.load(_CONFIG_PATH)
    _ezviz_config = None  # 下次调用 get_ezviz_config 时重新加载
    return _config
