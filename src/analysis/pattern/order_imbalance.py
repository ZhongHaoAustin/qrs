"""
订单不平衡模式识别模块 - Order Imbalance Pattern Recognition

识别订单簿中的不平衡模式，包括大单不平衡等。
"""

from typing import Any

import numpy as np
import pandas as pd

from src.analysis.pattern.pattern_recognition import register_pattern


@register_pattern("large_order_imbalance")
def identify_large_order_imbalance(df: pd.DataFrame) -> pd.DataFrame:
    """
    识别大单不平衡模式。

    当买卖双方订单量差异超过阈值时，认为存在大单不平衡。

    Args:
        df: 包含订单簿数据的DataFrame

    Returns:
        添加了大单不平衡标识的DataFrame
    """
    df = df.copy()

    # 计算总买卖量
    bid_volume_cols = [col for col in df.columns if col.startswith("bid_volume")]
    ask_volume_cols = [col for col in df.columns if col.startswith("ask_volume")]

    if bid_volume_cols and ask_volume_cols:
        df["total_bid_volume"] = df[bid_volume_cols].sum(axis=1)
        df["total_ask_volume"] = df[ask_volume_cols].sum(axis=1)
        df["order_imbalance"] = (df["total_bid_volume"] - df["total_ask_volume"]) / (
            df["total_bid_volume"] + df["total_ask_volume"]
        )

        # 识别大单不平衡（绝对值大于0.5）
        df["large_order_imbalance"] = df["order_imbalance"].abs() > 0.5

        # 添加不平衡强度指标
        df["imbalance_strength"] = df["order_imbalance"].abs()

        # 添加不平衡方向
        df["imbalance_direction"] = np.where(
            df["order_imbalance"] > 0, 1, np.where(df["order_imbalance"] < 0, -1, 0)
        )

    return df


@register_pattern("extreme_imbalance")
def identify_extreme_imbalance(
    df: pd.DataFrame, threshold: float = 0.8
) -> pd.DataFrame:
    """
    识别极端不平衡模式。

    Args:
        df: 包含订单簿数据的DataFrame
        threshold: 极端不平衡阈值，默认0.8

    Returns:
        添加了极端不平衡标识的DataFrame
    """
    df = df.copy()

    if "order_imbalance" in df.columns:
        df["extreme_imbalance"] = df["order_imbalance"].abs() > threshold

        # 计算连续极端不平衡的持续时间
        df["extreme_imbalance_duration"] = (
            df["extreme_imbalance"]
            .groupby((~df["extreme_imbalance"]).cumsum())
            .cumsum()
        )

    return df


@register_pattern("imbalance_reversal")
def identify_imbalance_reversal(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    识别不平衡反转模式。

    当订单不平衡方向在短时间内发生反转时，可能存在反转信号。

    Args:
        df: 包含订单簿数据的DataFrame
        window: 观察窗口大小

    Returns:
        添加了不平衡反转标识的DataFrame
    """
    df = df.copy()

    if "imbalance_direction" in df.columns:
        # 计算方向变化
        df["direction_change"] = df["imbalance_direction"].diff()

        # 识别反转点（方向从正变负或从负变正）
        df["imbalance_reversal"] = df["direction_change"].abs() == 2

        # 计算反转强度（基于不平衡幅度的变化）
        if "imbalance_strength" in df.columns:
            df["strength_change"] = df["imbalance_strength"].diff()
            df["strong_reversal"] = df["imbalance_reversal"] & (
                df["strength_change"].abs() > df["strength_change"].std()
            )

    return df
