"""
特征构建核心模块 - Core Feature Construction Module

提供特征构建的基础框架和通用功能。
"""

from typing import Any, Callable, Dict, List

from loguru import logger
import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame, feature_functions: List[Callable]) -> pd.DataFrame:
    """
    使用提供的特征函数从订单簿数据构建特征。

    Args:
        df: 包含订单簿数据的DataFrame
        feature_functions: 构建特征的函数列表

    Returns:
        包含构建特征的DataFrame
    """
    result_df = df.copy()

    for feature_func in feature_functions:
        try:
            result_df = feature_func(result_df)
        except Exception as e:
            logger.error(f"应用特征函数 {feature_func.__name__} 时出错: {e}")

    return result_df


def register_feature(name: str = None):
    """
    注册特征函数的装饰器。

    Args:
        name: 特征名称，如果为None则使用函数名

    Returns:
        装饰器函数
    """

    def decorator(feature_func: Callable) -> Callable:
        feature_func.feature_name = name or feature_func.__name__
        return feature_func

    return decorator


def validate_feature_data(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    验证特征构建所需的数据列是否存在。

    Args:
        df: 数据DataFrame
        required_columns: 必需的列名列表

    Returns:
        如果所有必需列都存在则返回True，否则返回False
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.warning(f"缺少必需的数据列: {missing_columns}")
        return False
    return True


def create_lag_features(
    df: pd.DataFrame, columns: List[str], lags: List[int]
) -> pd.DataFrame:
    """
    创建滞后特征。

    Args:
        df: 数据DataFrame
        columns: 需要创建滞后特征的列名列表
        lags: 滞后期数列表

    Returns:
        添加了滞后特征的DataFrame
    """
    result_df = df.copy()

    for col in columns:
        if col in df.columns:
            for lag in lags:
                result_df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    return result_df


def create_rolling_features(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int],
    functions: List[str] = ["mean", "std", "min", "max"],
) -> pd.DataFrame:
    """
    创建滚动窗口特征。

    Args:
        df: 数据DataFrame
        columns: 需要创建滚动特征的列名列表
        windows: 滚动窗口大小列表
        functions: 统计函数列表

    Returns:
        添加了滚动特征的DataFrame
    """
    result_df = df.copy()

    for col in columns:
        if col in df.columns:
            for window in windows:
                for func in functions:
                    if func == "mean":
                        result_df[f"{col}_rolling_mean_{window}"] = (
                            df[col].rolling(window=window).mean()
                        )
                    elif func == "std":
                        result_df[f"{col}_rolling_std_{window}"] = (
                            df[col].rolling(window=window).std()
                        )
                    elif func == "min":
                        result_df[f"{col}_rolling_min_{window}"] = (
                            df[col].rolling(window=window).min()
                        )
                    elif func == "max":
                        result_df[f"{col}_rolling_max_{window}"] = (
                            df[col].rolling(window=window).max()
                        )
                    elif func == "sum":
                        result_df[f"{col}_rolling_sum_{window}"] = (
                            df[col].rolling(window=window).sum()
                        )

    return result_df
