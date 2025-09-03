"""
Meta模块使用示例 - Meta Module Usage Example

展示如何使用meta模块提供的核心框架功能。
"""

from typing import Callable, Dict, List

from loguru import logger
import pandas as pd

from src.meta import (
    build_features,
    calculate_metrics,
    identify_patterns,
    register_feature,
    register_metric,
    register_pattern,
)


# 示例：注册一个自定义模式识别函数
@register_pattern("custom_pattern")
def identify_custom_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """
    自定义模式识别函数示例。

    Args:
        df: 数据DataFrame

    Returns:
        添加了自定义模式标识的DataFrame
    """
    df = df.copy()

    # 示例：识别价格突破模式
    if "last_price" in df.columns:
        df["price_breakout"] = df["last_price"] > df["last_price"].rolling(
            20
        ).max().shift(1)

    return df


# 示例：注册一个自定义特征构建函数
@register_feature("custom_feature")
def construct_custom_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    自定义特征构建函数示例。

    Args:
        df: 数据DataFrame

    Returns:
        添加了自定义特征的DataFrame
    """
    df = df.copy()

    # 示例：构建价格动量特征
    if "last_price" in df.columns:
        df["price_momentum_ratio"] = df["last_price"] / df["last_price"].shift(5)
        df["price_acceleration"] = df["last_price"].diff().diff()

    return df


# 示例：注册一个自定义评估指标函数
@register_metric("custom_metric")
def calculate_custom_metric(df: pd.DataFrame) -> float:
    """
    自定义评估指标函数示例。

    Args:
        df: 数据DataFrame

    Returns:
        自定义指标值
    """
    if "price_breakout" in df.columns and "price_momentum_ratio" in df.columns:
        # 示例：计算突破模式与动量的相关性
        breakout_events = df[df["price_breakout"] == True]
        if len(breakout_events) > 0:
            avg_momentum = breakout_events["price_momentum_ratio"].mean()
            return avg_momentum
    return 0.0


def example_meta_analysis_pipeline(df: pd.DataFrame) -> Dict[str, any]:
    """
    使用meta模块进行完整分析流水线的示例。

    Args:
        df: 原始数据DataFrame

    Returns:
        包含分析结果的字典
    """
    logger.info("开始Meta分析流水线...")

    # 1. 模式识别
    pattern_functions = [identify_custom_pattern]
    df = identify_patterns(df, pattern_functions)
    logger.info("模式识别完成")

    # 2. 特征构建
    feature_functions = [construct_custom_feature]
    df = build_features(df, feature_functions)
    logger.info("特征构建完成")

    # 3. 指标计算
    metric_functions = [calculate_custom_metric]
    metrics = calculate_metrics(df, metric_functions)
    logger.info("指标计算完成")

    logger.info("Meta分析流水线完成")

    return {"processed_data": df, "metrics": metrics}


if __name__ == "__main__":
    logger.info("Meta模块使用示例")
    logger.info("请使用 example_meta_analysis_pipeline() 函数运行完整分析流水线")
