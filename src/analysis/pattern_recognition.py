"""
Pattern recognition module for orderbook data analysis.
"""

from typing import Any, Callable, Dict, List

from loguru import logger
import pandas as pd


def identify_patterns(
    df: pd.DataFrame, pattern_functions: List[Callable]
) -> pd.DataFrame:
    """
    Identify patterns in orderbook data using provided pattern functions.

    Args:
        df: DataFrame with orderbook data
        pattern_functions: List of functions that identify specific patterns

    Returns:
        DataFrame with pattern identification results
    """
    result_df = df.copy()

    for pattern_func in pattern_functions:
        try:
            result_df = pattern_func(result_df)
        except Exception as e:
            logger.error(
                f"Error applying pattern function {pattern_func.__name__}: {e}"
            )

    return result_df


def register_pattern(name: str, pattern_func: Callable) -> Callable:
    """
    Decorator to register pattern functions.

    Args:
        name: Name of the pattern
        pattern_func: Function that identifies the pattern

    Returns:
        The pattern function with metadata
    """
    pattern_func.pattern_name = name
    return pattern_func
