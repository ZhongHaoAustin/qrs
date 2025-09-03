"""
模式效果指标模块 - Pattern Effectiveness Metrics Module

提供模式识别效果相关的评估指标。
"""

from typing import Dict, List

from loguru import logger
import numpy as np
import pandas as pd

from src.analysis.metrics.evaluation_metrics import register_metric


@register_metric("pattern_effectiveness")
def calculate_pattern_effectiveness(df: pd.DataFrame) -> Dict[str, float]:
    """
    计算模式识别效果指标。

    Args:
        df: 包含模式识别结果的DataFrame

    Returns:
        包含模式效果指标的字典
    """
    if "large_order_imbalance" in df.columns and "last_price" in df.columns:
        # 计算价格变化
        df["price_change"] = df["last_price"].diff()
        df["price_change_pct"] = df["last_price"].pct_change()

        # 模式发生时的价格变化
        imbalance_events = df[df["large_order_imbalance"] == True]

        if len(imbalance_events) > 0:
            # 模式发生后的价格变化
            future_price_changes = []
            for idx in imbalance_events.index:
                # 查看模式发生后1-5个时间点的价格变化
                future_changes = []
                for i in range(1, 6):
                    if idx + i < len(df):
                        future_changes.append(df["price_change_pct"].iloc[idx + i])
                if future_changes:
                    future_price_changes.extend(future_changes)

            if future_price_changes:
                future_changes_array = np.array(future_price_changes)

                # 模式效果指标
                avg_future_change = np.mean(future_changes_array)
                std_future_change = np.std(future_price_changes)

                # 正收益概率
                positive_prob = np.mean(future_changes_array > 0)

                # 显著收益概率（超过1个标准差）
                significant_prob = np.mean(
                    np.abs(future_changes_array) > std_future_change
                )

                return {
                    "pattern_frequency": len(imbalance_events) / len(df),
                    "avg_future_price_change": avg_future_change,
                    "std_future_price_change": std_future_change,
                    "positive_return_probability": positive_prob,
                    "significant_return_probability": significant_prob,
                    "pattern_effectiveness_score": positive_prob * avg_future_change,
                }

    return {
        "pattern_frequency": 0.0,
        "avg_future_price_change": 0.0,
        "std_future_price_change": 0.0,
        "positive_return_probability": 0.0,
        "significant_return_probability": 0.0,
        "pattern_effectiveness_score": 0.0,
    }


@register_metric("pattern_timing_accuracy")
def calculate_pattern_timing_accuracy(df: pd.DataFrame) -> Dict[str, float]:
    """
    计算模式时机准确性指标。

    Args:
        df: 包含模式识别结果的DataFrame

    Returns:
        包含模式时机准确性指标的字典
    """
    if "large_order_imbalance" in df.columns and "last_price" in df.columns:
        # 计算价格变化
        df["price_change_pct"] = df["last_price"].pct_change()

        # 识别价格趋势变化点
        df["price_trend"] = np.where(
            df["price_change_pct"] > 0, 1, np.where(df["price_change_pct"] < 0, -1, 0)
        )
        df["trend_change"] = df["price_trend"].diff().abs() == 2

        # 模式发生与趋势变化的时间关系
        imbalance_events = df[df["large_order_imbalance"] == True]
        trend_changes = df[df["trend_change"] == True]

        if len(imbalance_events) > 0 and len(trend_changes) > 0:
            # 计算模式发生到趋势变化的时间间隔
            timing_accuracy = []
            for idx in imbalance_events.index:
                # 查找后续的趋势变化
                future_trend_changes = trend_changes[trend_changes.index > idx]
                if len(future_trend_changes) > 0:
                    time_to_trend_change = future_trend_changes.index[0] - idx
                    timing_accuracy.append(time_to_trend_change)

            if timing_accuracy:
                return {
                    "avg_time_to_trend_change": np.mean(timing_accuracy),
                    "min_time_to_trend_change": np.min(timing_accuracy),
                    "max_time_to_trend_change": np.max(timing_accuracy),
                    "timing_accuracy_std": np.std(timing_accuracy),
                }

    return {
        "avg_time_to_trend_change": 0.0,
        "min_time_to_trend_change": 0.0,
        "max_time_to_trend_change": 0.0,
        "timing_accuracy_std": 0.0,
    }


@register_metric("pattern_stability")
def calculate_pattern_stability(df: pd.DataFrame) -> Dict[str, float]:
    """
    计算模式稳定性指标。

    Args:
        df: 包含模式识别结果的DataFrame

    Returns:
        包含模式稳定性指标的字典
    """
    if "large_order_imbalance" in df.columns:
        # 计算模式发生的连续性
        df["imbalance_shift"] = df["large_order_imbalance"].shift(1)
        df["imbalance_continuation"] = (
            df["large_order_imbalance"] & df["imbalance_shift"]
        )

        # 模式持续时间
        imbalance_periods = []
        current_period = 0

        for is_imbalance in df["large_order_imbalance"]:
            if is_imbalance:
                current_period += 1
            else:
                if current_period > 0:
                    imbalance_periods.append(current_period)
                current_period = 0

        if current_period > 0:
            imbalance_periods.append(current_period)

        if imbalance_periods:
            return {
                "avg_pattern_duration": np.mean(imbalance_periods),
                "max_pattern_duration": np.max(imbalance_periods),
                "pattern_duration_std": np.std(imbalance_periods),
                "continuation_rate": df["imbalance_continuation"].sum()
                / df["large_order_imbalance"].sum()
                if df["large_order_imbalance"].sum() > 0
                else 0,
            }

    return {
        "avg_pattern_duration": 0.0,
        "max_pattern_duration": 0.0,
        "pattern_duration_std": 0.0,
        "continuation_rate": 0.0,
    }


@register_metric("pattern_correlation")
def calculate_pattern_correlation(df: pd.DataFrame) -> Dict[str, float]:
    """
    计算模式与其他指标的相关性。

    Args:
        df: 包含模式识别结果和其他指标的DataFrame

    Returns:
        包含模式相关性指标的字典
    """
    correlations = {}

    if "large_order_imbalance" in df.columns:
        # 与价格变化的相关性
        if "price_change_pct" in df.columns:
            corr = df["large_order_imbalance"].astype(int).corr(df["price_change_pct"])
            correlations["price_change_correlation"] = (
                corr if not pd.isna(corr) else 0.0
            )

        # 与成交量的相关性
        volume_cols = [
            col for col in df.columns if "volume" in col and "delta" not in col
        ]
        if volume_cols:
            df["total_volume"] = df[volume_cols].sum(axis=1)
            corr = df["large_order_imbalance"].astype(int).corr(df["total_volume"])
            correlations["volume_correlation"] = corr if not pd.isna(corr) else 0.0

        # 与订单流的相关性
        if "net_order_flow" in df.columns:
            corr = df["large_order_imbalance"].astype(int).corr(df["net_order_flow"])
            correlations["order_flow_correlation"] = corr if not pd.isna(corr) else 0.0

    return correlations
