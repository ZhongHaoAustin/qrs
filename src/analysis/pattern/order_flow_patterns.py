"""
订单流模式识别模块 - Order Flow Pattern Recognition

识别订单流中的各种模式，包括订单流变化、累积订单流等。
"""

import numpy as np
import pandas as pd

from src.meta import register_pattern


@register_pattern("order_flow_patterns")
def identify_order_flow_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    识别订单流模式。

    基于订单流数据识别各种模式，包括：
    - 订单流加速/减速
    - 订单流方向变化
    - 异常订单流

    Args:
        df: 包含订单流数据的DataFrame

    Returns:
        添加了订单流模式标识的DataFrame
    """
    df = df.copy()

    # 检查是否有订单流数据
    if "bid_delta1" in df.columns and "ask_delta1" in df.columns:
        # 计算净订单流
        df["net_order_flow"] = df["bid_delta1"] + df["ask_delta1"]

        # 计算订单流变化率
        df["order_flow_change"] = df["net_order_flow"].diff()

        # 识别订单流加速（变化率增大）
        df["order_flow_acceleration"] = (
            df["order_flow_change"] > df["order_flow_change"].rolling(10).mean()
        )

        # 识别订单流减速（变化率减小）
        df["order_flow_deceleration"] = (
            df["order_flow_change"] < df["order_flow_change"].rolling(10).mean()
        )

        # 识别异常订单流（超过2个标准差）
        flow_std = df["net_order_flow"].std()
        df["abnormal_order_flow"] = df["net_order_flow"].abs() > 2 * flow_std

        # 计算订单流强度
        df["order_flow_strength"] = df["net_order_flow"].abs()

        # 识别订单流方向变化
        df["order_flow_direction"] = np.where(
            df["net_order_flow"] > 0, 1, np.where(df["net_order_flow"] < 0, -1, 0)
        )
        df["order_flow_direction_change"] = df["order_flow_direction"].diff().abs() == 2

    return df


@register_pattern("volume_spike")
def identify_volume_spike(
    df: pd.DataFrame, threshold_multiplier: float = 3.0
) -> pd.DataFrame:
    """
    识别成交量突增模式。

    Args:
        df: 包含成交量数据的DataFrame
        threshold_multiplier: 突增阈值倍数，默认3.0

    Returns:
        添加了成交量突增标识的DataFrame
    """
    df = df.copy()

    # 计算总成交量
    volume_cols = [col for col in df.columns if "volume" in col and "delta" not in col]
    if volume_cols:
        df["total_volume"] = df[volume_cols].sum(axis=1)

        # 计算成交量移动平均
        df["volume_ma"] = df["total_volume"].rolling(window=20).mean()
        df["volume_std"] = df["total_volume"].rolling(window=20).std()

        # 识别成交量突增
        threshold = df["volume_ma"] + threshold_multiplier * df["volume_std"]
        df["volume_spike"] = df["total_volume"] > threshold

        # 计算突增强度
        df["spike_intensity"] = df["total_volume"] / df["volume_ma"]

    return df


@register_pattern("price_momentum")
def identify_price_momentum(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    识别价格动量模式。

    Args:
        df: 包含价格数据的DataFrame
        window: 动量计算窗口

    Returns:
        添加了价格动量标识的DataFrame
    """
    df = df.copy()

    if "last_price" in df.columns:
        # 计算价格变化
        df["price_change"] = df["last_price"].diff()
        df["price_change_pct"] = df["last_price"].pct_change()

        # 计算价格动量
        df["price_momentum"] = df["price_change"].rolling(window=window).sum()
        df["price_momentum_pct"] = df["price_change_pct"].rolling(window=window).sum()

        # 识别强动量
        momentum_threshold = df["price_momentum_pct"].std()
        df["strong_momentum"] = df["price_momentum_pct"].abs() > momentum_threshold

        # 识别动量方向
        df["momentum_direction"] = np.where(
            df["price_momentum"] > 0, 1, np.where(df["price_momentum"] < 0, -1, 0)
        )

        # 识别动量反转
        df["momentum_reversal"] = df["momentum_direction"].diff().abs() == 2

    return df
