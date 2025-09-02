from typing import Dict, List

import numpy as np
import pandas as pd


def calculate_order_book_delta_numpy(
    df: pd.DataFrame, max_level: int = 1
) -> pd.DataFrame:
    """
    使用numpy优化计算订单簿变化量/delta：在同一日期内，如果bid/ask price_n与上一行相同，
    则使用当前bid/ask volume_n减去上一行的值

    正数表示新增订单，负数表示撤单

    Args:
        df: 包含期权tick数据的DataFrame，需要包含以下列:
            - datetime: 时间戳
            - bid_price1到bid_price5: 买方价格
            - ask_price1到ask_price5: 卖方价格
            - bid_volume1到bid_volume5: 买方量
            - ask_volume1到ask_volume5: 卖方量
        max_level: 最大计算的价格档位，默认为1（只计算bid_delta1和ask_delta1）

    Returns:
        pd.DataFrame: 包含订单簿变化量计算结果的DataFrame
    """
    # 确保数据按时间排序
    df = df.sort_index().copy()

    # 添加日期列用于分组
    df["date"] = df["datetime"].dt.date

    # 初始化订单簿变化量列
    delta_cols = {}

    # 对每个价格档位计算订单簿变化量（按顺序处理，模拟原始函数的行为）
    for i in range(1, max_level + 1):
        bid_price_col = f"bid_price{i}"
        ask_price_col = f"ask_price{i}"
        bid_volume_col = f"bid_volume{i}"
        ask_volume_col = f"ask_volume{i}"
        bid_delta_col = f"bid_delta{i}"
        ask_delta_col = f"ask_delta{i}"

        if (
            bid_price_col in df.columns
            and bid_volume_col in df.columns
            and ask_price_col in df.columns
            and ask_volume_col in df.columns
        ):
            # 按日期分组计算前一行数据（每次只计算当前列的prev值）
            df[f"prev_{bid_price_col}"] = df.groupby("date")[bid_price_col].shift(1)
            df[f"prev_{bid_volume_col}"] = df.groupby("date")[bid_volume_col].shift(1)
            df[f"prev_{ask_price_col}"] = df.groupby("date")[ask_price_col].shift(1)
            df[f"prev_{ask_volume_col}"] = df.groupby("date")[ask_volume_col].shift(1)

            # 计算相同价格情况下的delta
            bid_delta_same_price = df[bid_volume_col] - df[f"prev_{bid_volume_col}"]
            ask_delta_same_price = df[ask_volume_col] - df[f"prev_{ask_volume_col}"]

            # 初始化delta数组
            bid_delta = np.zeros(len(df))
            ask_delta = np.zeros(len(df))

            # 处理相同价格的情况
            same_date = df["date"] == df["date"].shift(1)
            # 注意：这里应该比较当前价格与前一行的prev价格
            same_bid_price = (
                df[bid_price_col] == df[f"prev_{bid_price_col}"]
            ) & same_date
            same_ask_price = (
                df[ask_price_col] == df[f"prev_{ask_price_col}"]
            ) & same_date

            bid_delta[same_bid_price.fillna(False)] = bid_delta_same_price[
                same_bid_price.fillna(False)
            ]
            ask_delta[same_ask_price.fillna(False)] = ask_delta_same_price[
                same_ask_price.fillna(False)
            ]

            # 处理价格不同的情况 - 需要在前一行中查找相同价格
            diff_bid_price = (~same_bid_price.fillna(True)) & same_date
            diff_ask_price = (~same_ask_price.fillna(True)) & same_date

            # 处理NaN值 - 如果当前行或前一行的价格是NaN，则不计算delta
            valid_current_bid = ~df[bid_price_col].isna()
            valid_prev_bid = ~df[f"prev_{bid_price_col}"].isna()
            valid_bid = valid_current_bid & valid_prev_bid

            valid_current_ask = ~df[ask_price_col].isna()
            valid_prev_ask = ~df[f"prev_{ask_price_col}"].isna()
            valid_ask = valid_current_ask & valid_prev_ask

            diff_bid_price = diff_bid_price & valid_bid
            diff_ask_price = diff_ask_price & valid_ask

            if diff_bid_price.any():
                # 对于买方价格不同的情况，在前一行的所有价格档位中查找相同价格
                for idx in range(len(df)):
                    # 第一行没有前一行数据，跳过
                    if idx == 0:
                        continue

                    if diff_bid_price.iloc[idx]:
                        # 查找当前价格在前一行的哪个档位
                        current_price = df[bid_price_col].iloc[idx]
                        prev_volume = 0

                        # 检查当前价格是否为NaN
                        if pd.isna(current_price):
                            continue

                        # 在前一行的所有价格档位中查找相同价格
                        for j in range(1, max_level + 1):
                            prev_bid_price_col = f"bid_price{j}"
                            prev_bid_volume_col = f"bid_volume{j}"
                            if (
                                prev_bid_price_col in df.columns
                                and prev_bid_volume_col in df.columns
                                and not pd.isna(df[prev_bid_price_col].iloc[idx - 1])
                                and df[prev_bid_price_col].iloc[idx - 1]
                                == current_price
                            ):
                                prev_volume = df[prev_bid_volume_col].iloc[idx - 1]
                                break

                        # 计算差值
                        bid_delta[idx] = df[bid_volume_col].iloc[idx] - prev_volume

            if diff_ask_price.any():
                # 对于卖方价格不同的情况，在前一行的所有价格档位中查找相同价格
                for idx in range(len(df)):
                    # 第一行没有前一行数据，跳过
                    if idx == 0:
                        continue

                    if diff_ask_price.iloc[idx]:
                        # 查找当前价格在前一行的哪个档位
                        current_price = df[ask_price_col].iloc[idx]
                        prev_volume = 0

                        # 检查当前价格是否为NaN
                        if pd.isna(current_price):
                            continue

                        # 在前一行的所有价格档位中查找相同价格
                        for j in range(1, max_level + 1):
                            prev_ask_price_col = f"ask_price{j}"
                            prev_ask_volume_col = f"ask_volume{j}"
                            if (
                                prev_ask_price_col in df.columns
                                and prev_ask_volume_col in df.columns
                                and not pd.isna(df[prev_ask_price_col].iloc[idx - 1])
                                and df[prev_ask_price_col].iloc[idx - 1]
                                == current_price
                            ):
                                prev_volume = df[prev_ask_volume_col].iloc[idx - 1]
                                break

                        # 计算差值
                        ask_delta[idx] = df[ask_volume_col].iloc[idx] - prev_volume

            delta_cols[bid_delta_col] = bid_delta
            delta_cols[ask_delta_col] = ask_delta

    # 创建结果DataFrame
    result_df = pd.DataFrame(delta_cols, index=df.index)

    # 删除临时列
    temp_cols = [col for col in df.columns if col.startswith("prev_")]
    df = df.drop(columns=temp_cols + ["date"])

    # 合并结果
    result_df = pd.concat([df, result_df], axis=1)

    return result_df


def calculate_order_book_delta_numpy_v2(
    df: pd.DataFrame, max_level: int = 1
) -> pd.DataFrame:
    """
    使用numpy优化的另一个版本，采用向量化操作进一步提高性能

    Args:
        df: 包含期权tick数据的DataFrame
        max_level: 最大计算的价格档位，默认为1（只计算bid_delta1和ask_delta1）

    Returns:
        pd.DataFrame: 包含订单簿变化量计算结果的DataFrame
    """
    # 确保数据按时间排序
    df = df.sort_index().copy()

    # 添加日期列用于分组
    df["date"] = df["datetime"].dt.date

    # 初始化订单簿变化量列
    delta_data = {}

    # 对每个价格档位计算订单簿变化量（按顺序处理，模拟原始函数的行为）
    for i in range(1, max_level + 1):
        bid_price_col = f"bid_price{i}"
        ask_price_col = f"ask_price{i}"
        bid_volume_col = f"bid_volume{i}"
        ask_volume_col = f"ask_volume{i}"
        bid_delta_col = f"bid_delta{i}"
        ask_delta_col = f"ask_delta{i}"

        if (
            bid_price_col in df.columns
            and bid_volume_col in df.columns
            and ask_price_col in df.columns
            and ask_volume_col in df.columns
        ):
            # 按日期分组计算前一行数据（每次只计算当前列的prev值）
            df[f"prev_{bid_price_col}"] = df.groupby("date")[bid_price_col].shift(1)
            df[f"prev_{bid_volume_col}"] = df.groupby("date")[bid_volume_col].shift(1)
            df[f"prev_{ask_price_col}"] = df.groupby("date")[ask_price_col].shift(1)
            df[f"prev_{ask_volume_col}"] = df.groupby("date")[ask_volume_col].shift(1)

            # 计算相同价格情况下的delta
            bid_delta_same_price = df[bid_volume_col] - df[f"prev_{bid_volume_col}"]
            ask_delta_same_price = df[ask_volume_col] - df[f"prev_{ask_volume_col}"]

            # 初始化delta数组
            bid_delta = np.zeros(len(df))
            ask_delta = np.zeros(len(df))

            # 处理相同价格的情况
            same_date = df["date"] == df["date"].shift(1)
            # 注意：这里应该比较当前价格与前一行的prev价格
            same_bid_price = (
                df[bid_price_col] == df[f"prev_{bid_price_col}"]
            ) & same_date
            same_ask_price = (
                df[ask_price_col] == df[f"prev_{ask_price_col}"]
            ) & same_date

            bid_delta[same_bid_price.fillna(False)] = bid_delta_same_price[
                same_bid_price.fillna(False)
            ]
            ask_delta[same_ask_price.fillna(False)] = ask_delta_same_price[
                same_ask_price.fillna(False)
            ]

            # 处理价格不同的情况 - 需要在前一行中查找相同价格
            diff_bid_price = (~same_bid_price.fillna(True)) & same_date
            diff_ask_price = (~same_ask_price.fillna(True)) & same_date

            # 处理NaN值 - 如果当前行或前一行的价格是NaN，则不计算delta
            valid_current_bid = ~df[bid_price_col].isna()
            valid_prev_bid = ~df[f"prev_{bid_price_col}"].isna()
            valid_bid = valid_current_bid & valid_prev_bid

            valid_current_ask = ~df[ask_price_col].isna()
            valid_prev_ask = ~df[f"prev_{ask_price_col}"].isna()
            valid_ask = valid_current_ask & valid_prev_ask

            diff_bid_price = diff_bid_price & valid_bid
            diff_ask_price = diff_ask_price & valid_ask

            if diff_bid_price.any():
                # 对于买方价格不同的情况，在前一行的所有价格档位中查找相同价格
                for idx in range(len(df)):
                    # 第一行没有前一行数据，跳过
                    if idx == 0:
                        continue

                    if diff_bid_price.iloc[idx]:
                        # 查找当前价格在前一行的哪个档位
                        current_price = df[bid_price_col].iloc[idx]
                        prev_volume = 0

                        # 检查当前价格是否为NaN
                        if pd.isna(current_price):
                            continue

                        # 在前一行的所有价格档位中查找相同价格
                        for j in range(1, max_level + 1):
                            prev_bid_price_col = f"bid_price{j}"
                            prev_bid_volume_col = f"bid_volume{j}"
                            if (
                                prev_bid_price_col in df.columns
                                and prev_bid_volume_col in df.columns
                                and not pd.isna(df[prev_bid_price_col].iloc[idx - 1])
                                and df[prev_bid_price_col].iloc[idx - 1]
                                == current_price
                            ):
                                prev_volume = df[prev_bid_volume_col].iloc[idx - 1]
                                break

                        # 计算差值
                        bid_delta[idx] = df[bid_volume_col].iloc[idx] - prev_volume

            if diff_ask_price.any():
                # 对于卖方价格不同的情况，在前一行的所有价格档位中查找相同价格
                for idx in range(len(df)):
                    # 第一行没有前一行数据，跳过
                    if idx == 0:
                        continue

                    if diff_ask_price.iloc[idx]:
                        # 查找当前价格在前一行的哪个档位
                        current_price = df[ask_price_col].iloc[idx]
                        prev_volume = 0

                        # 检查当前价格是否为NaN
                        if pd.isna(current_price):
                            continue

                        # 在前一行的所有价格档位中查找相同价格
                        for j in range(1, max_level + 1):
                            prev_ask_price_col = f"ask_price{j}"
                            prev_ask_volume_col = f"ask_volume{j}"
                            if (
                                prev_ask_price_col in df.columns
                                and prev_ask_volume_col in df.columns
                                and not pd.isna(df[prev_ask_price_col].iloc[idx - 1])
                                and df[prev_ask_price_col].iloc[idx - 1]
                                == current_price
                            ):
                                prev_volume = df[prev_ask_volume_col].iloc[idx - 1]
                                break

                        # 计算差值
                        ask_delta[idx] = df[ask_volume_col].iloc[idx] - prev_volume

            delta_data[bid_delta_col] = bid_delta
            delta_data[ask_delta_col] = ask_delta

    # 创建结果DataFrame
    result_df = pd.DataFrame(delta_data, index=df.index)

    # 删除临时列
    temp_cols = [col for col in df.columns if col.startswith("prev_")]
    df = df.drop(columns=temp_cols + ["date"])

    # 合并结果
    result_df = pd.concat([df, result_df], axis=1)

    return result_df
