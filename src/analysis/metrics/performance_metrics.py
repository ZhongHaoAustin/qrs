"""
性能指标模块 - Performance Metrics Module

提供策略和模型性能相关的评估指标。
"""

from typing import Dict, List

from loguru import logger
import numpy as np
import pandas as pd

from src.meta import register_metric


@register_metric("sharpe_ratio")
def calculate_sharpe_ratio(df: pd.DataFrame) -> float:
    """
    计算夏普比率。

    Args:
        df: 包含收益数据的DataFrame

    Returns:
        夏普比率
    """
    if "returns" in df.columns:
        returns = df["returns"].dropna()
        if len(returns) > 0:
            mean_return = returns.mean()
            std_return = returns.std()
            if std_return > 0:
                # 假设无风险利率为0
                sharpe_ratio = mean_return / std_return
                return sharpe_ratio
    return 0.0


@register_metric("max_drawdown")
def calculate_max_drawdown(df: pd.DataFrame) -> Dict[str, float]:
    """
    计算最大回撤。

    Args:
        df: 包含价格或净值数据的DataFrame

    Returns:
        包含最大回撤相关指标的字典
    """
    if "last_price" in df.columns:
        # 计算累积收益
        df["cumulative_return"] = (1 + df["last_price"].pct_change()).cumprod()

        # 计算滚动最大值
        df["rolling_max"] = df["cumulative_return"].expanding().max()

        # 计算回撤
        df["drawdown"] = (df["cumulative_return"] - df["rolling_max"]) / df[
            "rolling_max"
        ]

        # 最大回撤
        max_drawdown = df["drawdown"].min()

        # 回撤持续时间
        drawdown_periods = df["drawdown"] < 0
        drawdown_duration = drawdown_periods.sum()

        return {
            "max_drawdown": max_drawdown,
            "drawdown_duration": drawdown_duration,
            "avg_drawdown": df["drawdown"].mean(),
        }

    return {"max_drawdown": 0.0, "drawdown_duration": 0, "avg_drawdown": 0.0}


@register_metric("volatility_metrics")
def calculate_volatility_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    计算波动率相关指标。

    Args:
        df: 包含价格数据的DataFrame

    Returns:
        包含波动率指标的字典
    """
    if "last_price" in df.columns:
        # 计算收益率
        df["returns"] = df["last_price"].pct_change()
        returns = df["returns"].dropna()

        if len(returns) > 0:
            # 日波动率
            daily_volatility = returns.std()

            # 年化波动率
            annualized_volatility = daily_volatility * np.sqrt(252)

            # 滚动波动率
            rolling_vol = returns.rolling(window=20).std()

            # 波动率的波动率
            vol_of_vol = rolling_vol.std()

            return {
                "daily_volatility": daily_volatility,
                "annualized_volatility": annualized_volatility,
                "volatility_of_volatility": vol_of_vol,
                "avg_rolling_volatility": rolling_vol.mean(),
            }

    return {
        "daily_volatility": 0.0,
        "annualized_volatility": 0.0,
        "volatility_of_volatility": 0.0,
        "avg_rolling_volatility": 0.0,
    }


@register_metric("return_metrics")
def calculate_return_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    计算收益相关指标。

    Args:
        df: 包含价格数据的DataFrame

    Returns:
        包含收益指标的字典
    """
    if "last_price" in df.columns:
        # 计算收益率
        df["returns"] = df["last_price"].pct_change()
        returns = df["returns"].dropna()

        if len(returns) > 0:
            # 总收益
            total_return = (df["last_price"].iloc[-1] / df["last_price"].iloc[0]) - 1

            # 年化收益
            days = len(df)
            annualized_return = (1 + total_return) ** (252 / days) - 1

            # 正收益和负收益
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]

            # 胜率
            win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0

            # 平均正收益和负收益
            avg_positive_return = (
                positive_returns.mean() if len(positive_returns) > 0 else 0
            )
            avg_negative_return = (
                negative_returns.mean() if len(negative_returns) > 0 else 0
            )

            # 收益偏度
            return_skewness = returns.skew()

            # 收益峰度
            return_kurtosis = returns.kurtosis()

            return {
                "total_return": total_return,
                "annualized_return": annualized_return,
                "win_rate": win_rate,
                "avg_positive_return": avg_positive_return,
                "avg_negative_return": avg_negative_return,
                "return_skewness": return_skewness,
                "return_kurtosis": return_kurtosis,
            }

    return {
        "total_return": 0.0,
        "annualized_return": 0.0,
        "win_rate": 0.0,
        "avg_positive_return": 0.0,
        "avg_negative_return": 0.0,
        "return_skewness": 0.0,
        "return_kurtosis": 0.0,
    }


@register_metric("risk_adjusted_returns")
def calculate_risk_adjusted_returns(df: pd.DataFrame) -> Dict[str, float]:
    """
    计算风险调整收益指标。

    Args:
        df: 包含价格数据的DataFrame

    Returns:
        包含风险调整收益指标的字典
    """
    if "last_price" in df.columns:
        # 计算收益率
        df["returns"] = df["last_price"].pct_change()
        returns = df["returns"].dropna()

        if len(returns) > 0:
            mean_return = returns.mean()
            std_return = returns.std()

            if std_return > 0:
                # 夏普比率
                sharpe_ratio = mean_return / std_return

                # 索提诺比率（只考虑下行风险）
                downside_returns = returns[returns < 0]
                downside_std = (
                    downside_returns.std() if len(downside_returns) > 0 else 0
                )
                sortino_ratio = mean_return / downside_std if downside_std > 0 else 0

                # 卡尔玛比率（收益/最大回撤）
                df["cumulative_return"] = (1 + returns).cumprod()
                df["rolling_max"] = df["cumulative_return"].expanding().max()
                df["drawdown"] = (df["cumulative_return"] - df["rolling_max"]) / df[
                    "rolling_max"
                ]
                max_drawdown = abs(df["drawdown"].min())
                calmar_ratio = mean_return / max_drawdown if max_drawdown > 0 else 0

                return {
                    "sharpe_ratio": sharpe_ratio,
                    "sortino_ratio": sortino_ratio,
                    "calmar_ratio": calmar_ratio,
                }

    return {"sharpe_ratio": 0.0, "sortino_ratio": 0.0, "calmar_ratio": 0.0}


@register_metric("performance_metrics")
def calculate_performance_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    计算综合性能指标。

    Args:
        df: 包含价格数据的DataFrame

    Returns:
        包含所有性能指标的字典
    """
    metrics = {}

    # 计算各种性能指标
    sharpe_ratio = calculate_sharpe_ratio(df)
    max_drawdown_data = calculate_max_drawdown(df)
    volatility_data = calculate_volatility_metrics(df)
    return_data = calculate_return_metrics(df)
    risk_adjusted_data = calculate_risk_adjusted_returns(df)

    # 合并所有指标
    metrics.update({"sharpe_ratio": sharpe_ratio})
    metrics.update(max_drawdown_data)
    metrics.update(volatility_data)
    metrics.update(return_data)
    metrics.update(risk_adjusted_data)

    return metrics
