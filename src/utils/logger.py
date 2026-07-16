"""
结构化日志系统 — 基于 loguru
按天轮转,同时输出到文件和控制台
格式: 时间 | 级别 | 模块 | 消息
"""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from src.utils.config import get_config

_initialized = False


def setup_logging():
    """初始化全局日志配置(幂等,多次调用安全)"""
    global _initialized
    if _initialized:
        return

    config = get_config()
    log_dir = Path(config.paths.logs)
    log_dir.mkdir(parents=True, exist_ok=True)

    # 移除默认handler
    logger.remove()

    # 控制台输出: 彩色,精简格式
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> | <level>{message}</level>",
        colorize=True,
    )

    # 文件输出: 按天轮转,保留30天,JSON结构化
    logger.add(
        log_dir / "app_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="00:00",          # 每天午夜轮转
        retention="30 days",       # 保留30天
        compression="zip",         # 旧日志压缩
        encoding="utf-8",
        enqueue=True,              # 线程安全
    )

    # 错误日志单独文件
    logger.add(
        log_dir / "error_{time:YYYY-MM-DD}.log",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="00:00",
        retention="30 days",
        compression="zip",
        encoding="utf-8",
        enqueue=True,
    )

    _initialized = True
    logger.info("日志系统初始化完成")


def get_logger(name: str = __name__):
    """获取带模块名的logger"""
    if not _initialized:
        setup_logging()
    return logger.bind(name=name)
