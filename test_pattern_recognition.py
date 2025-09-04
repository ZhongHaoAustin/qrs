#!/usr/bin/env python3
"""
测试模式识别研究脚本 - 使用模拟数据运行完整的分析流程
"""

from pathlib import Path
import sys

from loguru import logger
import numpy as np
import pandas as pd

# 添加src到路径
sys.path.append(str(Path(__file__).parent))

from src.analysis.order_flow_np import calculate_order_book_delta_numpy
from src.data.data_processing import prepare_data
from src.meta import build_features, calculate_metrics, identify_patterns

# 设置日志
logger.remove()
logger.add(
    sys.stdout,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)


# Pattern recognition functions
@register_pattern("large_order_imbalance")
def identify_large_order_imbalance(df: pd.DataFrame) -> pd.DataFrame:
    """Identify periods with large order book imbalance."""
    df = df.copy()

    # Calculate total bid and ask volumes
    bid_volume_cols = [col for col in df.columns if col.startswith("bid_volume")]
    ask_volume_cols = [col for col in df.columns if col.startswith("ask_volume")]

    if bid_volume_cols and ask_volume_cols:
        df["total_bid_volume"] = df[bid_volume_cols].sum(axis=1)
        df["total_ask_volume"] = df[ask_volume_cols].sum(axis=1)
        df["order_imbalance"] = (df["total_bid_volume"] - df["total_ask_volume"]) / (
            df["total_bid_volume"] + df["total_ask_volume"]
        )

        # Identify large order imbalance (greater than 0.5)
        df["large_order_imbalance"] = df["order_imbalance"].abs() > 0.5

        logger.info(f"发现 {df['large_order_imbalance'].sum()} 个大订单不平衡时期")

    return df


# Feature construction functions
@register_feature("order_flow_features")
def construct_order_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construct features based on order flow data."""
    df = df.copy()

    # Calculate order book delta if not already present
    if "bid_delta1" not in df.columns:
        logger.info("计算订单簿变化量...")
        df = calculate_order_book_delta_numpy(df)

    # Calculate cumulative order flow
    if "bid_delta1" in df.columns and "ask_delta1" in df.columns:
        df["net_order_flow"] = df["bid_delta1"] + df["ask_delta1"]
        df["cumulative_order_flow"] = df["net_order_flow"].cumsum()

        # Calculate order flow volatility
        df["order_flow_volatility"] = df["net_order_flow"].rolling(window=20).std()

        logger.info("订单流特征构建完成")

    return df


# Evaluation metric functions
@register_metric("prediction_accuracy")
def calculate_prediction_accuracy(df: pd.DataFrame) -> float:
    """Calculate the accuracy of pattern predictions."""
    if "large_order_imbalance" in df.columns and "last_price" in df.columns:
        # For this example, we'll assume price movement is the direction of next tick
        df["price_movement"] = (
            df["last_price"].diff().shift(-1).apply(lambda x: 1 if x > 0 else 0)
        )
        df["prediction"] = df["large_order_imbalance"].apply(lambda x: 1 if x else 0)

        # 计算准确率
        valid_predictions = df["prediction"].notna() & df["price_movement"].notna()
        if valid_predictions.sum() > 0:
            accuracy = (
                df.loc[valid_predictions, "prediction"]
                == df.loc[valid_predictions, "price_movement"]
            ).mean()
            logger.info(f"预测准确率: {accuracy:.4f}")
            return accuracy
    return 0.0


@register_metric("order_imbalance_stats")
def calculate_order_imbalance_stats(df: pd.DataFrame) -> dict:
    """Calculate statistics about order imbalance."""
    if "order_imbalance" in df.columns:
        stats = {
            "mean_imbalance": df["order_imbalance"].mean(),
            "std_imbalance": df["order_imbalance"].std(),
            "max_imbalance": df["order_imbalance"].max(),
            "min_imbalance": df["order_imbalance"].min(),
            "large_imbalance_ratio": (df["order_imbalance"].abs() > 0.5).mean(),
        }
        logger.info(f"订单不平衡统计: {stats}")
        return stats
    return {}


def load_test_data(filename="test_orderbook_data.csv"):
    """加载测试数据"""
    try:
        df = pd.read_csv(filename)
        df["datetime"] = pd.to_datetime(df["datetime"])
        logger.info(f"成功加载测试数据: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"测试数据文件 {filename} 未找到")
        return None


def main():
    """主函数 - 运行完整的模式识别研究流程"""
    logger.info("开始模式识别研究测试")

    # 加载测试数据
    df = load_test_data()
    if df is None:
        logger.error("无法加载测试数据，退出")
        return

    # 准备数据
    logger.info("准备数据...")
    df = prepare_data(df)
    logger.info(f"数据准备完成，形状: {df.shape}")

    # 显示数据基本信息
    logger.info(f"数据时间范围: {df['datetime'].min()} 到 {df['datetime'].max()}")
    logger.info(f"数据列: {list(df.columns)}")

    # 识别模式
    logger.info("识别模式...")
    pattern_functions = [identify_large_order_imbalance]
    df = identify_patterns(df, pattern_functions)
    logger.info("模式识别完成")

    # 构建特征
    logger.info("构建特征...")
    feature_functions = [construct_order_flow_features]
    df = build_features(df, feature_functions)
    logger.info("特征构建完成")

    # 计算指标
    logger.info("计算评估指标...")
    metric_functions = [calculate_prediction_accuracy, calculate_order_imbalance_stats]
    metrics = calculate_metrics(df, metric_functions)
    logger.info("指标计算完成")

    # 显示结果
    logger.info("=== 分析结果 ===")
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, dict):
            logger.info(f"{metric_name}:")
            for key, value in metric_value.items():
                logger.info(f"  {key}: {value}")
        else:
            logger.info(f"{metric_name}: {metric_value}")

    # 保存结果
    output_file = "pattern_recognition_results.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"结果已保存到 {output_file}")

    # 显示一些关键统计
    if "large_order_imbalance" in df.columns:
        imbalance_count = df["large_order_imbalance"].sum()
        total_count = len(df)
        logger.info(
            f"大订单不平衡时期: {imbalance_count}/{total_count} ({imbalance_count / total_count * 100:.2f}%)"
        )

    if "order_flow_volatility" in df.columns:
        avg_volatility = df["order_flow_volatility"].mean()
        logger.info(f"平均订单流波动率: {avg_volatility:.4f}")

    logger.success("模式识别研究测试完成！")

    return {"data": df, "metrics": metrics}


if __name__ == "__main__":
    result = main()
