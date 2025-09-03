#!/usr/bin/env python3
"""
测试数据生成器 - 为模式识别研究创建模拟的订单簿数据
"""

from datetime import datetime, timedelta
import random

import numpy as np
import pandas as pd


def generate_mock_orderbook_data(num_rows=1000, start_time=None):
    """
    生成模拟的订单簿tick数据

    Args:
        num_rows: 生成的数据行数
        start_time: 开始时间，默认为当前时间

    Returns:
        pd.DataFrame: 包含订单簿数据的DataFrame
    """
    if start_time is None:
        start_time = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)

    # 生成时间序列（每500ms一个tick）
    timestamps = [start_time + timedelta(milliseconds=500 * i) for i in range(num_rows)]

    # 基础价格（模拟期权价格）
    base_price = 0.05
    price_trend = np.cumsum(np.random.normal(0, 0.001, num_rows))
    last_prices = base_price + price_trend

    data = []

    for i, (timestamp, last_price) in enumerate(zip(timestamps, last_prices)):
        # 生成5档买卖盘数据
        row = {
            "datetime": timestamp,
            "last_price": last_price,
            "highest": last_price * (1 + random.uniform(0, 0.02)),
            "lowest": last_price * (1 - random.uniform(0, 0.02)),
            "average": last_price * (1 + random.uniform(-0.01, 0.01)),
            "volume": random.randint(100, 10000),
            "amount": last_price * random.randint(100, 10000),
            "open_interest": random.randint(100000, 1000000),
            "instrument_id": "SSE.10009482",
            "localtime": timestamp,
        }

        # 生成5档买卖盘价格和数量
        spread = random.uniform(0.0001, 0.001)
        for level in range(1, 6):
            # 买盘价格（递减）
            bid_price = last_price - spread / 2 - (level - 1) * 0.0001
            # 卖盘价格（递增）
            ask_price = last_price + spread / 2 + (level - 1) * 0.0001

            # 生成数量（模拟订单簿不平衡）
            base_volume = random.randint(1000, 10000)

            # 在某些时间点创建订单簿不平衡
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

            row[f"bid_price{level}"] = bid_price
            row[f"bid_volume{level}"] = max(0, bid_volume)
            row[f"ask_price{level}"] = ask_price
            row[f"ask_volume{level}"] = max(0, ask_volume)

        data.append(row)

    df = pd.DataFrame(data)
    return df


def save_test_data(df, filename="test_orderbook_data.csv"):
    """保存测试数据到CSV文件"""
    df.to_csv(filename, index=False)
    print(f"测试数据已保存到 {filename}")
    print(f"数据形状: {df.shape}")
    print(f"时间范围: {df['datetime'].min()} 到 {df['datetime'].max()}")
    return filename


if __name__ == "__main__":
    # 生成测试数据
    print("正在生成模拟订单簿数据...")
    test_df = generate_mock_orderbook_data(num_rows=2000)

    # 保存数据
    filename = save_test_data(test_df)

    # 显示数据样本
    print("\n数据样本:")
    print(test_df.head())

    print("\n数据统计:")
    print(test_df.describe())
