"""
新模块化架构使用示例 - Example Usage of New Modular Architecture

展示如何使用重新设计的pattern、feature、metrics三个模块。
"""

from typing import Any, Dict

from loguru import logger
import pandas as pd

# 导入新的模块化接口
from src.analysis import (
    # 特征构建模块
    build_features,
    # 评估指标模块
    calculate_metrics,
    # 订单流分析
    calculate_order_book_delta_numpy,
    calculate_pattern_effectiveness,
    calculate_performance_metrics,
    calculate_prediction_accuracy,
    construct_order_flow_features,
    construct_statistical_features,
    construct_technical_features,
    identify_large_order_imbalance,
    identify_order_flow_patterns,
    # 模式识别模块
    identify_patterns,
)


def example_pattern_recognition(df: pd.DataFrame) -> pd.DataFrame:
    """
    模式识别示例。

    Args:
        df: 订单簿数据

    Returns:
        包含模式识别结果的DataFrame
    """
    logger.info("开始模式识别...")

    # 使用模式识别模块
    pattern_functions = [identify_large_order_imbalance, identify_order_flow_patterns]

    result_df = identify_patterns(df, pattern_functions)
    logger.info("模式识别完成")

    return result_df


def example_feature_construction(df: pd.DataFrame) -> pd.DataFrame:
    """
    特征构建示例。

    Args:
        df: 订单簿数据

    Returns:
        包含特征的DataFrame
    """
    logger.info("开始特征构建...")

    # 使用特征构建模块
    feature_functions = [
        construct_order_flow_features,
        construct_technical_features,
        construct_statistical_features,
    ]

    result_df = build_features(df, feature_functions)
    logger.info("特征构建完成")

    return result_df


def example_metrics_calculation(df: pd.DataFrame) -> Dict[str, float]:
    """
    指标计算示例。

    Args:
        df: 包含数据和结果的DataFrame

    Returns:
        指标字典
    """
    logger.info("开始指标计算...")

    # 使用评估指标模块
    metric_functions = [
        calculate_prediction_accuracy,
        calculate_performance_metrics,
        calculate_pattern_effectiveness,
    ]

    metrics = calculate_metrics(df, metric_functions)
    logger.info("指标计算完成")

    return metrics


def example_complete_pipeline(df: pd.DataFrame) -> Dict[str, Any]:
    """
    完整流水线示例。

    Args:
        df: 原始订单簿数据

    Returns:
        包含处理结果和指标的字典
    """
    logger.info("开始完整分析流水线...")

    # 1. 计算订单流数据
    df = calculate_order_book_delta_numpy(df)

    # 2. 模式识别
    df = example_pattern_recognition(df)

    # 3. 特征构建
    df = example_feature_construction(df)

    # 4. 指标计算
    metrics = example_metrics_calculation(df)

    logger.info("完整分析流水线完成")

    return {"processed_data": df, "metrics": metrics}


if __name__ == "__main__":
    # 这里可以添加测试代码
    logger.info("新模块化架构使用示例")
    logger.info("请使用 example_complete_pipeline() 函数运行完整分析流水线")
