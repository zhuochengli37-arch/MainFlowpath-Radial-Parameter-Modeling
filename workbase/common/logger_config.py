"""
AIR2 Project1 日志配置。

本模块为整个项目提供统一的日志初始化逻辑。
日志会同时写入控制台和文件，并分别使用适合阅读和调试的格式。
"""

import logging
import os
from datetime import datetime
from pathlib import Path


def setup_logger(
    name: str = "AIR2",
    log_dir: str = "./logs",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    log_to_file: bool = True
) -> logging.Logger:
    """
    创建同时支持控制台和文件输出的 logger。

    参数:
        name: logger 名称
        log_dir: 日志文件保存目录
        console_level: 控制台输出日志级别，默认 INFO
        file_level: 文件输出日志级别，默认 DEBUG
        log_to_file: 是否写入日志文件，默认 True

    返回:
        配置完成的 logger 实例
    """
    logger = logging.getLogger(name)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # 控制台 handler：使用更适合用户阅读的简洁格式
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_format = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # 文件 handler：使用更适合调试的详细格式
    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"{name}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(file_level)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

        logger.info(f"Log file created: {log_file}")

    return logger


def get_logger(name: str = "AIR2") -> logging.Logger:
    """
    获取已有 logger；如果不存在则按默认配置创建。

    参数:
        name: logger 名称

    返回:
        logger 实例
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger
