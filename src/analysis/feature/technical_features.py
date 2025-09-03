"""
技术指标特征构建模块 - Technical Indicator Feature Construction

基于价格和成交量数据构建技术指标特征。
"""

from typing import Dict, List

from loguru import logger
import numpy as np
import pandas as pd

from src.analysis.feature.feature_construction import register_feature


@register_feature("technical_indicators")
def construct_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    构建技术指标特征。

    包括：
    - 移动平均线
    - 相对强弱指标
    - 布林带
    - MACD指标

    Args:
        df: 包含价格和成交量数据的DataFrame

    Returns:
        添加了技术指标特征的DataFrame
    """
    df = df.copy()

    if "last_price" in df.columns:
        # 移动平均线
        df["ma_5"] = df["last_price"].rolling(window=5).mean()
        df["ma_10"] = df["last_price"].rolling(window=10).mean()
        df["ma_20"] = df["last_price"].rolling(window=20).mean()
        df["ma_60"] = df["last_price"].rolling(window=60).mean()

        # 价格相对于移动平均线的位置
        df["price_vs_ma5"] = df["last_price"] / df["ma_5"] - 1
        df["price_vs_ma20"] = df["last_price"] / df["ma_20"] - 1

        # 移动平均线交叉信号
        df["ma_cross_5_20"] = np.where(df["ma_5"] > df["ma_20"], 1, -1)
        df["ma_cross_10_20"] = np.where(df["ma_10"] > df["ma_20"], 1, -1)

        # 布林带
        df["bb_middle"] = df["last_price"].rolling(window=20).mean()
        bb_std = df["last_price"].rolling(window=20).std()
        df["bb_upper"] = df["bb_middle"] + 2 * bb_std
        df["bb_lower"] = df["bb_middle"] - 2 * bb_std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        df["bb_position"] = (df["last_price"] - df["bb_lower"]) / (
            df["bb_upper"] - df["bb_lower"]
        )

        # RSI指标
        df["rsi"] = calculate_rsi(df["last_price"], window=14)

        # MACD指标
        macd_data = calculate_macd(df["last_price"])
        df["macd"] = macd_data["macd"]
        df["macd_signal"] = macd_data["signal"]
        df["macd_histogram"] = macd_data["histogram"]

        # 价格动量
        df["price_momentum_5"] = df["last_price"] / df["last_price"].shift(5) - 1
        df["price_momentum_10"] = df["last_price"] / df["last_price"].shift(10) - 1
        df["price_momentum_20"] = df["last_price"] / df["last_price"].shift(20) - 1

    return df


def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    计算相对强弱指标(RSI)。

    Args:
        prices: 价格序列
        window: 计算窗口

    Returns:
        RSI值序列
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(
    prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Dict[str, pd.Series]:
    """
    计算MACD指标。

    Args:
        prices: 价格序列
        fast: 快速EMA周期
        slow: 慢速EMA周期
        signal: 信号线EMA周期

    Returns:
        包含MACD、信号线和柱状图的字典
    """
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    histogram = macd - signal_line

    return {"macd": macd, "signal": signal_line, "histogram": histogram}


@register_feature("volatility_features")
def construct_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    构建波动率特征。

    Args:
        df: 包含价格数据的DataFrame

    Returns:
        添加了波动率特征的DataFrame
    """
    df = df.copy()

    if "last_price" in df.columns:
        # 价格变化
        df["price_change"] = df["last_price"].diff()
        df["price_change_pct"] = df["last_price"].pct_change()

        # 滚动波动率
        df["volatility_5"] = df["price_change_pct"].rolling(window=5).std()
        df["volatility_10"] = df["price_change_pct"].rolling(window=10).std()
        df["volatility_20"] = df["price_change_pct"].rolling(window=20).std()

        # 年化波动率
        df["annualized_volatility"] = df["volatility_20"] * np.sqrt(252)

        # 波动率变化率
        df["volatility_change_rate"] = df["volatility_20"].pct_change()

        # 价格范围
        df["price_range"] = (
            df["last_price"].rolling(window=20).max()
            - df["last_price"].rolling(window=20).min()
        )
        df["price_range_pct"] = df["price_range"] / df["last_price"]

    return df
