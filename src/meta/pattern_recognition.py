"""
模式识别核心模块 - Core Pattern Recognition Module

提供模式识别的基础框架和通用功能。
"""

from typing import Callable, List

from loguru import logger
import pandas as pd


def identify_patterns(
    df: pd.DataFrame, pattern_functions: List[Callable]
) -> pd.DataFrame:
    """
    使用提供的模式函数识别订单簿数据中的模式。

    Args:
        df: 包含订单簿数据的DataFrame
        pattern_functions: 识别特定模式的函数列表

    Returns:
        包含模式识别结果的DataFrame
    """
    result_df = df.copy()

    for pattern_func in pattern_functions:
        try:
            result_df = pattern_func(result_df)
        except Exception as e:
            logger.error(f"应用模式函数 {pattern_func.__name__} 时出错: {e}")

    return result_df


def register_pattern(name: str = None):
    """
    注册模式函数的装饰器。

    Args:
        name: 模式名称，如果为None则使用函数名

    Returns:
        装饰器函数
    """

    def decorator(pattern_func: Callable) -> Callable:
        pattern_func.pattern_name = name or pattern_func.__name__
        return pattern_func

    return decorator
