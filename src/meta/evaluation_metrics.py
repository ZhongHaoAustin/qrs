"""
评估指标核心模块 - Core Evaluation Metrics Module

提供评估指标的基础框架和通用功能。
"""

from typing import Any, Callable, Dict, List

from loguru import logger
import pandas as pd


def calculate_metrics(
    df: pd.DataFrame, metric_functions: List[Callable]
) -> Dict[str, float]:
    """
    使用提供的指标函数计算评估指标。

    Args:
        df: 包含数据和结果的DataFrame
        metric_functions: 计算指标的函数列表

    Returns:
        包含指标名称和值的字典
    """
    metrics = {}

    for metric_func in metric_functions:
        try:
            result = metric_func(df)
            if isinstance(result, dict):
                metrics.update(result)
            else:
                metrics[metric_func.__name__] = result
        except Exception as e:
            logger.error(f"应用指标函数 {metric_func.__name__} 时出错: {e}")

    return metrics


def register_metric(name: str = None):
    """
    注册指标函数的装饰器。

    Args:
        name: 指标名称，如果为None则使用函数名

    Returns:
        装饰器函数
    """

    def decorator(metric_func: Callable) -> Callable:
        metric_func.metric_name = name or metric_func.__name__
        return metric_func

    return decorator


def validate_metric_data(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    验证指标计算所需的数据列是否存在。

    Args:
        df: 数据DataFrame
        required_columns: 必需的列名列表

    Returns:
        如果所有必需列都存在则返回True，否则返回False
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.warning(f"缺少计算指标所需的数据列: {missing_columns}")
        return False
    return True


def calculate_metric_summary(metrics: Dict[str, float]) -> Dict[str, Any]:
    """
    计算指标汇总统计。

    Args:
        metrics: 指标字典

    Returns:
        包含汇总统计的字典
    """
    import numpy as np

    values = list(metrics.values())

    summary = {
        "total_metrics": len(metrics),
        "mean_value": np.mean(values),
        "std_value": np.std(values),
        "min_value": np.min(values),
        "max_value": np.max(values),
        "median_value": np.median(values),
    }

    return summary
