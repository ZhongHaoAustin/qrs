"""
Logging configuration utilities.
"""

import sys

from loguru import logger


def setup_logger(module_name: str):
    """
    Set up logger for a specific module.

    Args:
        module_name (str): Name of the module

    Returns:
        logger: Configured logger instance
    """
    # Remove default handlers
    logger.remove()

    # Add stdout handler with custom format
    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}",
        level="INFO",
    )

    # Add file handler
    logger.add(
        f"logs/{module_name}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="10 days",
    )

    return logger
