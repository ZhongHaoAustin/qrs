"""
订单流特征构建模块 - Order Flow Feature Construction

基于订单流数据构建各种特征，包括订单流变化、累积订单流等。
"""

from typing import Dict, List

from loguru import logger
import numpy as np
import pandas as pd

from src.analysis.feature.feature_construction import register_feature
from src.analysis.order_flow_np import calculate_order_book_delta_numpy


@register_feature("order_flow_features")
def construct_order_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    构建基于订单流数据的特征。

    包括：
    - 净订单流
    - 累积订单流
    - 订单流波动率
    - 订单流强度

    Args:
        df: 包含订单簿数据的DataFrame

    Returns:
        添加了订单流特征的DataFrame
    """
    df = df.copy()

    # 计算订单簿变化量（如果还没有）
    if "bid_delta1" not in df.columns:
        df = calculate_order_book_delta_numpy(df)

    # 计算净订单流
    if "bid_delta1" in df.columns and "ask_delta1" in df.columns:
        df["net_order_flow"] = df["bid_delta1"] + df["ask_delta1"]
        df["cumulative_order_flow"] = df["net_order_flow"].cumsum()

        # 计算订单流波动率
        df["order_flow_volatility"] = df["net_order_flow"].rolling(window=20).std()

        # 计算订单流强度
        df["order_flow_strength"] = df["net_order_flow"].abs()

        # 计算订单流变化率
        df["order_flow_change_rate"] = df["net_order_flow"].pct_change()

        # 计算订单流移动平均
        df["order_flow_ma_5"] = df["net_order_flow"].rolling(window=5).mean()
        df["order_flow_ma_20"] = df["net_order_flow"].rolling(window=20).mean()

        # 计算订单流相对强度
        df["order_flow_relative_strength"] = (
            df["net_order_flow"] / df["order_flow_ma_20"]
        )

    return df


@register_feature("order_imbalance_features")
def construct_order_imbalance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    构建基于订单不平衡的特征。

    Args:
        df: 包含订单簿数据的DataFrame

    Returns:
        添加了订单不平衡特征的DataFrame
    """
    df = df.copy()

    # 计算总买卖量
    bid_volume_cols = [col for col in df.columns if col.startswith("bid_volume")]
    ask_volume_cols = [col for col in df.columns if col.startswith("ask_volume")]

    if bid_volume_cols and ask_volume_cols:
        df["total_bid_volume"] = df[bid_volume_cols].sum(axis=1)
        df["total_ask_volume"] = df[ask_volume_cols].sum(axis=1)

        # 计算订单不平衡
        df["order_imbalance"] = (df["total_bid_volume"] - df["total_ask_volume"]) / (
            df["total_bid_volume"] + df["total_ask_volume"]
        )

        # 计算不平衡强度
        df["imbalance_strength"] = df["order_imbalance"].abs()

        # 计算不平衡方向
        df["imbalance_direction"] = np.where(
            df["order_imbalance"] > 0, 1, np.where(df["order_imbalance"] < 0, -1, 0)
        )

        # 计算不平衡变化率
        df["imbalance_change_rate"] = df["order_imbalance"].pct_change()

        # 计算不平衡移动平均
        df["imbalance_ma_10"] = df["order_imbalance"].rolling(window=10).mean()
        df["imbalance_ma_30"] = df["order_imbalance"].rolling(window=30).mean()

    return df


@register_feature("volume_features")
def construct_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    构建基于成交量的特征。

    Args:
        df: 包含成交量数据的DataFrame

    Returns:
        添加了成交量特征的DataFrame
    """
    df = df.copy()

    # 计算总成交量
    volume_cols = [col for col in df.columns if "volume" in col and "delta" not in col]
    if volume_cols:
        df["total_volume"] = df[volume_cols].sum(axis=1)

        # 计算成交量变化率
        df["volume_change_rate"] = df["total_volume"].pct_change()

        # 计算成交量移动平均
        df["volume_ma_5"] = df["total_volume"].rolling(window=5).mean()
        df["volume_ma_20"] = df["total_volume"].rolling(window=20).mean()
        df["volume_ma_60"] = df["total_volume"].rolling(window=60).mean()

        # 计算成交量相对强度
        df["volume_relative_strength"] = df["total_volume"] / df["volume_ma_20"]

        # 计算成交量波动率
        df["volume_volatility"] = df["total_volume"].rolling(window=20).std()

        # 计算成交量异常指标
        df["volume_anomaly"] = (df["total_volume"] - df["volume_ma_20"]) / df[
            "volume_volatility"
        ]

    return df
