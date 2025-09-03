"""
Evaluation metrics module for orderbook data analysis.
"""

from typing import Callable, Dict, List

from loguru import logger
import pandas as pd


def calculate_metrics(
    df: pd.DataFrame, metric_functions: List[Callable]
) -> Dict[str, float]:
    """
    Calculate evaluation metrics using provided metric functions.

    Args:
        df: DataFrame with orderbook data and results
        metric_functions: List of functions that calculate metrics

    Returns:
        Dictionary with metric names and values
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
            logger.error(f"Error applying metric function {metric_func.__name__}: {e}")

    return metrics


def register_metric(name: str, metric_func: Callable) -> Callable:
    """
    Decorator to register metric functions.

    Args:
        name: Name of the metric
        metric_func: Function that calculates the metric

    Returns:
        The metric function with metadata
    """
    metric_func.metric_name = name
    return metric_func
