from datetime import time
from typing import Optional, Union

from bokeh.models import ColumnDataSource
from loguru import logger
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare and clean the data for visualization."""
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def filter_daily_trading_hours(group_data) -> DataFrame:
    """Filter trading hours for daily data."""
    return filter_trading_hours(group_data, dt_col_name="datetime_opt")


def create_source(df: pd.DataFrame):
    """Create a ColumnDataSource from the dataframe."""
    return ColumnDataSource(df)


def display_sample_data(df: pd.DataFrame) -> pd.DataFrame:
    """Display sample data for verification."""
    return (
        df.set_index("datetime")
        .loc["2025-08-06 14:23:25":][
            [
                "last_price",
                "volume",
                "bid_price1",
                "ask_price1",
                "bid_volume1",
                "ask_volume1",
            ]
        ]
        .head(30)
    )


def add_calculated_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add calculated columns to the dataframe."""
    df["mid_price"] = (df["ask_price1"] + df["bid_price1"]) / 2
    df["vwmid_price"] = (
        df["ask_price1"] * df["ask_volume1"] + df["bid_price1"] * df["bid_volume1"]
    ) / (df["ask_volume1"] + df["bid_volume1"])
    return df


def filter_trading_hours(df: pd.DataFrame, dt_col_name="datetime") -> pd.DataFrame:
    """Filter data to keep only trading hours: 9:30-11:30 and 13:00-15:00."""
    if df.empty:
        return df

    df = df.copy()

    # First filter out NaT values to avoid NaTType attribute access issues
    valid_datetime_mask = df[dt_col_name].notna()
    df_filtered = df.loc[valid_datetime_mask].copy()

    if df_filtered.empty:
        return df_filtered

    # Extract time as datetime.time objects
    df_filtered["time"] = df_filtered[dt_col_name].dt.time

    # Create time objects for comparison using datetime.time
    morning_start = time(9, 30, 0)  # 09:30:00
    morning_end = time(11, 30, 0)  # 11:30:00
    afternoon_start = time(13, 0, 0)  # 13:00:00
    afternoon_end = time(15, 0, 0)  # 15:00:00

    # Filter for trading hours (morning session and afternoon session)
    trading_hours_filter = (
        (df_filtered["time"] >= morning_start) & (df_filtered["time"] <= morning_end)
    ) | (
        (df_filtered["time"] >= afternoon_start)
        & (df_filtered["time"] <= afternoon_end)
    )

    result_df = df_filtered.loc[trading_hours_filter].copy()
    result_df = result_df.drop("time", axis=1)
    return result_df


def calculate_spread(df: pd.DataFrame, tick_size: float = 0.0001) -> pd.DataFrame:
    """
    Calculate the spread as the difference between ask_price1 and bid_price1.
    Also calculate spread in terms of number of ticks.

    Args:
        df (pd.DataFrame): DataFrame with bid and ask price columns
        tick_size (float): The tick size for the instrument

    Returns:
        pd.DataFrame: DataFrame with added 'spread' and 'spread_tick_count' columns
    """
    if "ask_price1" in df.columns and "bid_price1" in df.columns:
        df["spread"] = df["ask_price1"] - df["bid_price1"]
        # Calculate spread in terms of number of ticks, rounded to nearest integer
        df["spread_tick_count"] = np.round(df["spread"] / tick_size)
    else:
        raise ValueError(
            "Required columns 'ask_price1' and 'bid_price1' not found in data"
        )
    return df


def filter_open_close_time(
    df: pd.DataFrame, open_minutes: int = 10, close_minutes: int = 10
) -> pd.DataFrame:
    """
    Filter out data within the first and last N minutes of each trading day.

    Args:
        df (pd.DataFrame): DataFrame with datetime column
        open_minutes (int): Number of minutes to exclude from the beginning of trading
        close_minutes (int): Number of minutes to exclude from the end of trading

    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    if df.empty:
        return df

    df = df.copy()

    # Filter out NaT values first
    valid_datetime_mask = df["datetime"].notna()
    df_filtered = df.loc[valid_datetime_mask].copy()

    if df_filtered.empty:
        return df_filtered

    df_filtered["date"] = df_filtered["datetime"].dt.date
    df_filtered["time"] = df_filtered["datetime"].dt.time

    # Calculate the time limits for filtering using datetime.time
    # Market opens at 9:30, closes at 15:00
    open_hour = 9
    open_minute = 30 + open_minutes
    if open_minute >= 60:
        open_hour += open_minute // 60
        open_minute: int = open_minute % 60

    close_hour = 15
    close_minute = 0 - close_minutes
    if close_minute < 0:
        close_hour += close_minute // 60  # This will subtract 1 from hour
        close_minute: int = 60 + (close_minute % 60)

    open_limit: time = time(hour=open_hour, minute=open_minute, second=0)
    close_limit: time = time(hour=close_hour, minute=close_minute, second=0)

    # Filter out data within the first and last N minutes
    filtered_df = df_filtered.loc[
        (df_filtered["time"] >= open_limit) & (df_filtered["time"] <= close_limit)
    ].copy()

    # Drop the helper columns
    filtered_df = filtered_df.drop(["date", "time"], axis=1)

    return filtered_df


def round_datetime_to_interval(
    df: pd.DataFrame, interval_ms: int = 500
) -> pd.DataFrame:
    """Round datetime to nearest specified interval in milliseconds."""
    df = df.copy()
    df["datetime_floor"] = df["datetime"].dt.round(f"{interval_ms}ms")
    return df


def log_duplicates(df: pd.DataFrame, data_type: str) -> None:
    """Log information about duplicate entries in the data."""
    duplicates = df[df.duplicated(subset=["datetime_floor"], keep="first")]
    if not duplicates.empty:
        logger.info(
            f"Found {len(duplicates)} duplicates in {data_type} data, dropping..., details: {duplicates}"
        )


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate entries based on datetime_floor column."""
    result = df.drop_duplicates(subset=["datetime_floor"], keep="first")
    # Ensure we return a DataFrame, not None
    if result is None:
        return df.iloc[:0].copy()  # Return empty DataFrame with same structure
    return result


def merge_data_asof(
    opt_data: pd.DataFrame, udly_data: pd.DataFrame, tolerance_ms: int = 500
) -> pd.DataFrame:
    """Merge option and underlying data using asof merge."""
    return pd.merge_asof(
        left=opt_data.sort_values("datetime_floor").assign(
            date=lambda x: x["datetime_floor"].dt.date
        ),
        right=udly_data.sort_values("datetime_floor").assign(
            date=lambda x: x["datetime_floor"].dt.date
        ),
        on="datetime_floor",
        suffixes=("_opt", "_udly"),
        direction="backward",
        tolerance=pd.Timedelta(milliseconds=tolerance_ms),
        by="date",
    )


def merge_data_standard(
    opt_data: pd.DataFrame, udly_data: pd.DataFrame, how: str = "inner"
) -> pd.DataFrame:
    """Merge option and underlying data using standard merge."""
    # Validate and convert how parameter
    valid_how = ["left", "right", "inner", "outer", "cross"]
    if how not in valid_how:
        how = "inner"  # Default fallback

    return pd.merge(
        left=opt_data,
        right=udly_data,
        on="datetime_floor",
        suffixes=("_opt", "_udly"),
        how=how,  # type: ignore[arg-type]
    )


def merge_datasets(
    opt_data: pd.DataFrame, udly_data: pd.DataFrame, merge_type: str = "asof"
) -> pd.DataFrame:
    """Merge option and underlying datasets based on the specified merge type."""
    if merge_type == "asof":
        return merge_data_asof(opt_data, udly_data)
    elif merge_type in ["inner", "outer", "left", "right"]:
        return merge_data_standard(opt_data, udly_data, how=merge_type)
    else:
        # Default to asof merge if invalid merge type specified
        logger.warning(
            f"Invalid merge type '{merge_type}' specified. Using 'asof' merge as default."
        )
        return merge_data_asof(opt_data, udly_data)


def calculate_mid_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate mid prices for option and underlying."""
    df = df.copy()
    df["opt_mid_price"] = (df["ask_price1_opt"] + df["bid_price1_opt"]) / 2
    df["udly_mid_price"] = (df["ask_price1_udly"] + df["bid_price1_udly"]) / 2
    return df


def calculate_price_changes(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate price changes for option and underlying."""
    df = df.copy()
    df["opt_mid_price_change"] = df["opt_mid_price"].diff()
    df["udly_mid_price_change"] = df["udly_mid_price"].diff()
    return df


def calculate_option_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate option delta using mid prices."""
    df = df.copy()
    # Delta = (Option Price Change) / (Underlying Price Change)
    # Avoid division by zero by using np.where to handle cases where underlying price change is 0
    df["option_delta"] = np.where(
        df["udly_mid_price_change"] != 0,
        df["opt_mid_price_change"] / df["udly_mid_price_change"],
        np.nan,
    )

    # Replace 0 values with NaN and forward-fill
    df["option_delta"] = df["option_delta"].replace(0, np.nan).ffill()
    return df


def add_symbol_info(
    df: pd.DataFrame, opt_symbol: str, udly_symbol: str
) -> pd.DataFrame:
    """Add symbol information to the dataframe."""
    df = df.copy()
    df["opt_symbol"] = opt_symbol
    df["udly_symbol"] = udly_symbol
    return df


def merge_and_analyze_option_underlying_data(
    opt_data: pd.DataFrame,
    udly_data: pd.DataFrame,
    opt_symbol: str = "90005540",
    udly_symbol: str = "159915",
):
    """Process tick data by rounding datetime to nearest 500ms, removing duplicates, merging and calculating option delta."""
    # Import load_config function
    from src.data.data_fetching import load_config

    # Load configuration
    config = load_config()
    merge_type = config.get("merge_type", "asof")

    # Data preprocessing - round datetime
    opt_data = round_datetime_to_interval(opt_data)
    udly_data = round_datetime_to_interval(udly_data)

    # Log duplicates before removing them
    log_duplicates(opt_data, "option")
    log_duplicates(udly_data, "underlying")

    # Remove duplicates
    opt_data = remove_duplicates(opt_data)
    udly_data = remove_duplicates(udly_data)

    # Merge the two datasets based on configuration
    merged_data = merge_datasets(opt_data, udly_data, merge_type)

    # Filter data to keep only trading hours: 9:30-11:30 and 13:00-15:00
    merged_data = filter_trading_hours(merged_data, dt_col_name="datetime_opt")

    # Calculate mid prices
    merged_data = calculate_mid_prices(merged_data)

    # Calculate price changes
    merged_data = calculate_price_changes(merged_data)

    # Calculate option delta
    merged_data = calculate_option_delta(merged_data)

    # Add symbol names to the merged data
    merged_data = add_symbol_info(merged_data, opt_symbol, udly_symbol)

    return merged_data


def calculate_order_imbalance(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate order book imbalance.

    Order book imbalance measures the imbalance between bid and ask volumes.

    Args:
        df: DataFrame with order book data including bid/ask prices and volumes

    Returns:
        DataFrame with added order_imbalance column
    """
    df = df.copy()

    # Calculate bid/ask volumes (sum of all levels)
    bid_volume_cols = [col for col in df.columns if col.startswith("bid_volume")]
    ask_volume_cols = [col for col in df.columns if col.startswith("ask_volume")]

    if bid_volume_cols:
        df["total_bid_volume"] = df[bid_volume_cols].sum(axis=1)
    if ask_volume_cols:
        df["total_ask_volume"] = df[ask_volume_cols].sum(axis=1)

    # Calculate order book imbalance
    if "total_bid_volume" in df.columns and "total_ask_volume" in df.columns:
        df["order_imbalance"] = (df["total_bid_volume"] - df["total_ask_volume"]) / (
            df["total_bid_volume"] + df["total_ask_volume"]
        )

    return df


def calculate_volume_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate volume ratio (volume relative to average).

    Volume ratio compares current volume to average volume to identify
    unusual trading activity.

    Args:
        df: DataFrame with order book data including volume column

    Returns:
        DataFrame with added volume_ratio column
    """
    df = df.copy()

    # Calculate volume ratio (volume relative to average)
    if "volume" in df.columns:
        df["volume_ratio"] = (
            df["volume"] / df["volume"].rolling(window=20, min_periods=1).mean()
        )

    return df


def calculate_depth_decay(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate order book depth decay.

    Depth decay measures the decrease in depth from level 1 to subsequent levels
    to identify momentum exhaustion.

    Args:
        df: DataFrame with order book data including bid/ask prices and volumes

    Returns:
        DataFrame with added spread, mid_price, depth_level_x and depth_decay columns
    """
    df = df.copy()

    # Calculate price levels for depth analysis
    if "bid_price1" in df.columns and "ask_price1" in df.columns:
        df["spread"] = df["ask_price1"] - df["bid_price1"]
        df["mid_price"] = (df["ask_price1"] + df["bid_price1"]) / 2

    # Calculate depth decay indicators
    price_level_cols = []
    for i in range(1, 6):
        bid_price_col = f"bid_price{i}"
        ask_price_col = f"ask_price{i}"
        bid_vol_col = f"bid_volume{i}"
        ask_vol_col = f"ask_volume{i}"

        if (
            bid_price_col in df.columns
            and ask_price_col in df.columns
            and bid_vol_col in df.columns
            and ask_vol_col in df.columns
        ):
            # Calculate depth at price levels
            df[f"depth_level_{i}"] = df[bid_vol_col] + df[ask_vol_col]
            price_level_cols.append(f"depth_level_{i}")

    # Calculate depth decay (decrease in depth from level 1 to subsequent levels)
    if len(price_level_cols) >= 2:
        df["depth_decay"] = (
            df[price_level_cols[0]] - df[price_level_cols[1:]].mean(axis=1)
        ) / df[price_level_cols[0]]
        df["depth_decay"].fillna(0, inplace=True)

    return df


def calculate_orderbook_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate order book metrics for parameter estimation.

    Calculates:
    1. Cancel ratio - to identify false breakouts
    2. Order book depth decay - to identify momentum exhaustion
    3. Volume ratio - to confirm insufficient follow-through

    Args:
        df: DataFrame with order book data including bid/ask prices and volumes

    Returns:
        DataFrame with added metrics columns
    """
    return (
        df.pipe(calculate_order_imbalance)
        .pipe(calculate_volume_ratio)
        .pipe(calculate_depth_decay)
    )


def estimate_scalping_parameters(df: pd.DataFrame) -> dict:
    """Estimate parameters for scalping strategy based on order book metrics.

    Parameters to estimate:
    1. Cancel ratio threshold (>70%) - to filter out false breakouts
    2. Order book depth decay threshold (>50%) - to identify momentum exhaustion
    3. Volume ratio threshold (<0.8) - to confirm insufficient follow-through

    Args:
        df: DataFrame with order book data and calculated metrics

    Returns:
        Dictionary with estimated parameter thresholds
    """
    df = calculate_orderbook_metrics(df)

    parameters = {}

    # Estimate cancel ratio threshold
    # In real implementation, this would be based on actual cancel data
    # For now, we'll use a default value based on the requirement (>70%)
    parameters["cancel_ratio_threshold"] = 0.7

    # Estimate order book depth decay threshold
    # This identifies momentum exhaustion when depth decays significantly
    if "depth_decay" in df.columns:
        # Calculate the percentage of time depth decay exceeds 50%
        depth_decay_threshold = (df["depth_decay"] > 0.5).mean()
        parameters["depth_decay_threshold"] = max(0.5, depth_decay_threshold)
    else:
        parameters["depth_decay_threshold"] = 0.5

    # Estimate volume ratio threshold
    # This confirms insufficient follow-through when volume is below average
    if "volume_ratio" in df.columns:
        # Calculate the percentage of time volume ratio is below 0.8
        volume_ratio_threshold = (df["volume_ratio"] < 0.8).mean()
        parameters["volume_ratio_threshold"] = min(0.8, volume_ratio_threshold)
    else:
        parameters["volume_ratio_threshold"] = 0.8

    return parameters


def filter_based_on_parameters(
    df: pd.DataFrame, parameters: Optional[dict] = None
) -> pd.DataFrame:
    """Filter data based on estimated parameters to identify trading opportunities.

    Args:
        df: DataFrame with order book data
        parameters: Dictionary with parameter thresholds, if None will estimate them

    Returns:
        DataFrame with added filter columns
    """
    df = df.copy()

    if parameters is None:
        parameters = estimate_scalping_parameters(df)

    # Calculate metrics
    df = calculate_orderbook_metrics(df)

    # Apply filters based on parameters
    # Filter for false breakouts (high cancel ratio)
    if "cancel_ratio" in df.columns:
        df["false_breakout"] = df["cancel_ratio"] > parameters["cancel_ratio_threshold"]
    else:
        df["false_breakout"] = False

    # Filter for momentum exhaustion (high depth decay)
    if "depth_decay" in df.columns:
        df["momentum_exhaustion"] = (
            df["depth_decay"] > parameters["depth_decay_threshold"]
        )
    else:
        df["momentum_exhaustion"] = False

    # Filter for insufficient follow-through (low volume ratio)
    if "volume_ratio" in df.columns:
        df["insufficient_follow_through"] = (
            df["volume_ratio"] < parameters["volume_ratio_threshold"]
        )
    else:
        df["insufficient_follow_through"] = True

    return df
