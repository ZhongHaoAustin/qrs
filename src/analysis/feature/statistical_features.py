"""
统计特征构建模块 - Statistical Feature Construction

基于数据统计特性构建特征，包括分布特征、相关性特征等。
"""

import numpy as np
import pandas as pd

from src.meta import register_feature


@register_feature("statistical_features")
def construct_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    构建统计特征。

    包括：
    - 分布特征
    - 相关性特征
    - 分位数特征
    - 偏度和峰度

    Args:
        df: 包含数据的DataFrame

    Returns:
        添加了统计特征的DataFrame
    """
    df = df.copy()

    # 价格相关统计特征
    if "last_price" in df.columns:
        # 分位数特征
        df["price_percentile_25"] = df["last_price"].rolling(window=20).quantile(0.25)
        df["price_percentile_75"] = df["last_price"].rolling(window=20).quantile(0.75)
        df["price_percentile_90"] = df["last_price"].rolling(window=20).quantile(0.90)

        # 价格位置（当前价格在分位数中的位置）
        df["price_position"] = (df["last_price"] - df["price_percentile_25"]) / (
            df["price_percentile_75"] - df["price_percentile_25"]
        )

        # 偏度和峰度
        df["price_skewness"] = df["last_price"].rolling(window=20).skew()
        df["price_kurtosis"] = df["last_price"].rolling(window=20).kurt()

        # 变异系数
        df["price_cv"] = (
            df["last_price"].rolling(window=20).std()
            / df["last_price"].rolling(window=20).mean()
        )

    # 成交量相关统计特征
    volume_cols = [col for col in df.columns if "volume" in col and "delta" not in col]
    if volume_cols:
        df["total_volume"] = df[volume_cols].sum(axis=1)

        # 成交量分位数
        df["volume_percentile_25"] = (
            df["total_volume"].rolling(window=20).quantile(0.25)
        )
        df["volume_percentile_75"] = (
            df["total_volume"].rolling(window=20).quantile(0.75)
        )

        # 成交量位置
        df["volume_position"] = (df["total_volume"] - df["volume_percentile_25"]) / (
            df["volume_percentile_75"] - df["volume_percentile_25"]
        )

        # 成交量偏度和峰度
        df["volume_skewness"] = df["total_volume"].rolling(window=20).skew()
        df["volume_kurtosis"] = df["total_volume"].rolling(window=20).kurt()

    return df


@register_feature("correlation_features")
def construct_correlation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    构建相关性特征。

    Args:
        df: 包含数据的DataFrame

    Returns:
        添加了相关性特征的DataFrame
    """
    df = df.copy()

    # 价格与成交量的相关性
    if "last_price" in df.columns:
        volume_cols = [
            col for col in df.columns if "volume" in col and "delta" not in col
        ]
        if volume_cols:
            df["total_volume"] = df[volume_cols].sum(axis=1)

            # 滚动相关性
            df["price_volume_corr"] = (
                df["last_price"].rolling(window=20).corr(df["total_volume"])
            )

            # 价格变化与成交量变化的相关性
            df["price_change"] = df["last_price"].diff()
            df["volume_change"] = df["total_volume"].diff()

            df["price_volume_change_corr"] = (
                df["price_change"].rolling(window=20).corr(df["volume_change"])
            )

    # 买卖不平衡与价格变化的相关性
    if "order_imbalance" in df.columns and "last_price" in df.columns:
        df["price_change_pct"] = df["last_price"].pct_change()

        df["imbalance_price_corr"] = (
            df["order_imbalance"].rolling(window=20).corr(df["price_change_pct"])
        )

    return df


@register_feature("distribution_features")
def construct_distribution_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    构建分布特征。

    Args:
        df: 包含数据的DataFrame

    Returns:
        添加了分布特征的DataFrame
    """
    df = df.copy()

    if "last_price" in df.columns:
        # 价格分布特征
        window = 20

        # 计算滚动窗口内的分布特征
        rolling_prices = df["last_price"].rolling(window=window)

        # 中位数
        df["price_median"] = rolling_prices.median()

        # 四分位距
        df["price_iqr"] = rolling_prices.quantile(0.75) - rolling_prices.quantile(0.25)

        # 极差
        df["price_range"] = rolling_prices.max() - rolling_prices.min()

        # 相对位置（当前价格在分布中的位置）
        df["price_relative_position"] = (df["last_price"] - rolling_prices.min()) / (
            rolling_prices.max() - rolling_prices.min()
        )

        # 标准化价格（Z-score）
        df["price_zscore"] = (
            df["last_price"] - rolling_prices.mean()
        ) / rolling_prices.std()

    return df


@register_feature("time_series_features")
def construct_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    构建时间序列特征。

    Args:
        df: 包含时间序列数据的DataFrame

    Returns:
        添加了时间序列特征的DataFrame
    """
    df = df.copy()

    if "datetime" in df.columns:
        # 时间特征
        df["hour"] = df["datetime"].dt.hour
        df["minute"] = df["datetime"].dt.minute
        df["day_of_week"] = df["datetime"].dt.dayofweek
        df["is_market_open"] = (df["hour"] >= 9) & (df["hour"] < 15)

        # 周期性特征
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # 滞后特征
    if "last_price" in df.columns:
        for lag in [1, 2, 3, 5, 10]:
            df[f"price_lag_{lag}"] = df["last_price"].shift(lag)
            df[f"price_change_lag_{lag}"] = df["last_price"].pct_change().shift(lag)

    return df
