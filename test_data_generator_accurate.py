#!/usr/bin/env python3
"""
准确的测试数据生成器 - 根据orderbook.md和data_fetching.py规范生成模拟订单簿数据
"""

from datetime import datetime, timedelta
import random

import numpy as np
import pandas as pd


def generate_accurate_orderbook_data(num_rows=1000, start_time=None):
    """
    根据orderbook.md规范生成准确的模拟订单簿tick数据

    Args:
        num_rows: 生成的数据行数
        start_time: 开始时间，默认为当前时间

    Returns:
        pd.DataFrame: 包含订单簿数据的DataFrame，符合data_fetching.py的字段规范
    """
    if start_time is None:
        start_time = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)

    # 生成时间序列（每500ms一个tick，符合高频交易特点）
    timestamps = [start_time + timedelta(milliseconds=500 * i) for i in range(num_rows)]

    # 基础价格（模拟期权价格，符合SSE.10009482的特点）
    base_price = 0.05
    price_trend = np.cumsum(np.random.normal(0, 0.001, num_rows))
    last_prices = base_price + price_trend

    data = []

    for i, (timestamp, last_price) in enumerate(zip(timestamps, last_prices)):
        # 生成符合orderbook.md规范的字段
        row = {
            # 时间相关字段
            "datetime": timestamp,
            "localtime": timestamp,
            # 价格相关字段
            "last_price": last_price,
            "highest": last_price * (1 + random.uniform(0, 0.02)),
            "lowest": last_price * (1 - random.uniform(0, 0.02)),
            "average": last_price * (1 + random.uniform(-0.01, 0.01)),
            # 成交量相关字段
            "volume": random.randint(100, 10000),
            "amount": last_price * random.randint(100, 10000),
            "open_interest": random.randint(100000, 1000000),
            # 合约信息
            "instrument_id": "SSE.10009482",
        }

        # 生成5档买卖盘价格和数量（符合orderbook.md的5档深度规范）
        spread = random.uniform(0.0001, 0.001)
        for level in range(1, 6):
            # 买盘价格（递减，bid_price1 > bid_price2 > ...）
            bid_price = last_price - spread / 2 - (level - 1) * 0.0001
            # 卖盘价格（递增，ask_price1 < ask_price2 < ...）
            ask_price = last_price + spread / 2 + (level - 1) * 0.0001

            # 生成数量（模拟订单簿不平衡）
            base_volume = random.randint(1000, 10000)

            # 在某些时间点创建订单簿不平衡（模拟真实市场情况）
            if i % 100 < 20:  # 20%的时间有订单簿不平衡
                imbalance_factor = random.uniform(0.3, 0.8)
                if random.random() > 0.5:
                    # 买盘偏多
                    bid_volume = int(base_volume * (1 + imbalance_factor))
                    ask_volume = int(base_volume * (1 - imbalance_factor))
                else:
                    # 卖盘偏多
                    bid_volume = int(base_volume * (1 - imbalance_factor))
                    ask_volume = int(base_volume * (1 + imbalance_factor))
            else:
                # 正常情况
                bid_volume = base_volume + random.randint(-1000, 1000)
                ask_volume = base_volume + random.randint(-1000, 1000)

            # 确保价格和数量的合理性
            row[f"bid_price{level}"] = max(0, bid_price)
            row[f"bid_volume{level}"] = max(0, bid_volume)
            row[f"ask_price{level}"] = max(0, ask_price)
            row[f"ask_volume{level}"] = max(0, ask_volume)

        # 添加vwap字段（根据data_fetching.py中的计算方式）
        if row["volume"] != 0:
            row["vwap"] = row["amount"] / row["volume"] / 10000  # multipier=10000
        else:
            row["vwap"] = 0

        data.append(row)

    df = pd.DataFrame(data)

    # 确保数据类型正确
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["localtime"] = pd.to_datetime(df["localtime"])

    return df


def save_test_data(df, filename="test_orderbook_data_accurate.csv"):
    """保存测试数据到CSV文件"""
    df.to_csv(filename, index=False)
    print(f"测试数据已保存到 {filename}")
    print(f"数据形状: {df.shape}")
    print(f"时间范围: {df['datetime'].min()} 到 {df['datetime'].max()}")

    # 验证字段完整性
    expected_columns = [
        "datetime",
        "last_price",
        "highest",
        "lowest",
        "average",
        "volume",
        "amount",
        "open_interest",
        "bid_price1",
        "bid_volume1",
        "ask_price1",
        "ask_volume1",
        "bid_price2",
        "bid_volume2",
        "ask_price2",
        "ask_volume2",
        "bid_price3",
        "bid_volume3",
        "ask_price3",
        "ask_volume3",
        "bid_price4",
        "bid_volume4",
        "ask_price4",
        "ask_volume4",
        "bid_price5",
        "bid_volume5",
        "ask_price5",
        "ask_volume5",
        "instrument_id",
        "localtime",
        "vwap",
    ]

    missing_columns = set(expected_columns) - set(df.columns)
    if missing_columns:
        print(f"警告: 缺少字段 {missing_columns}")
    else:
        print("✓ 所有必需字段都存在")

    return filename


def validate_data_structure(df):
    """验证数据结构的正确性"""
    print("\n=== 数据结构验证 ===")

    # 检查时间序列
    if df["datetime"].is_monotonic_increasing:
        print("✓ 时间序列单调递增")
    else:
        print("✗ 时间序列不是单调递增")

    # 检查价格合理性
    price_cols = ["last_price", "highest", "lowest", "average"]
    for col in price_cols:
        if col in df.columns and (df[col] > 0).all():
            print(f"✓ {col} 价格均为正数")
        else:
            print(f"✗ {col} 存在非正数价格")

    # 检查买卖盘价格关系
    for level in range(1, 6):
        bid_col = f"bid_price{level}"
        ask_col = f"ask_price{level}"
        if bid_col in df.columns and ask_col in df.columns:
            if (df[ask_col] > df[bid_col]).all():
                print(f"✓ 第{level}档卖价 > 买价")
            else:
                print(f"✗ 第{level}档卖价 <= 买价")

    # 检查订单簿不平衡
    if "bid_volume1" in df.columns and "ask_volume1" in df.columns:
        total_bid = df[[col for col in df.columns if col.startswith("bid_volume")]].sum(
            axis=1
        )
        total_ask = df[[col for col in df.columns if col.startswith("ask_volume")]].sum(
            axis=1
        )
        imbalance = (total_bid - total_ask) / (total_bid + total_ask)
        large_imbalance_ratio = (imbalance.abs() > 0.5).mean()
        print(f"✓ 大订单不平衡比例: {large_imbalance_ratio:.2%}")


if __name__ == "__main__":
    # 生成测试数据
    print("正在生成符合规范的模拟订单簿数据...")
    test_df = generate_accurate_orderbook_data(num_rows=2000)

    # 保存数据
    filename = save_test_data(test_df)

    # 验证数据结构
    validate_data_structure(test_df)

    # 显示数据样本
    print("\n=== 数据样本 ===")
    print(test_df.head())

    print("\n=== 数据统计 ===")
    print(test_df.describe())

    print("\n=== 订单簿不平衡分析 ===")
    bid_volume_cols = [col for col in test_df.columns if col.startswith("bid_volume")]
    ask_volume_cols = [col for col in test_df.columns if col.startswith("ask_volume")]

    if bid_volume_cols and ask_volume_cols:
        test_df["total_bid_volume"] = test_df[bid_volume_cols].sum(axis=1)
        test_df["total_ask_volume"] = test_df[ask_volume_cols].sum(axis=1)
        test_df["order_imbalance"] = (
            test_df["total_bid_volume"] - test_df["total_ask_volume"]
        ) / (test_df["total_bid_volume"] + test_df["total_ask_volume"])

        print(f"平均订单不平衡: {test_df['order_imbalance'].mean():.4f}")
        print(f"订单不平衡标准差: {test_df['order_imbalance'].std():.4f}")
        print(
            f"大订单不平衡时期: {(test_df['order_imbalance'].abs() > 0.5).sum()}/{len(test_df)}"
        )
