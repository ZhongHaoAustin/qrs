"""
Data utilities for the orderbook analysis project.
"""

from loguru import logger
import pandas as pd


def display_dataframe(df: pd.DataFrame, title: str = "", max_rows: int = None):
    """
    Display a DataFrame with optional title and row limit.

    Args:
        df (pd.DataFrame): DataFrame to display
        title (str): Optional title to display before the DataFrame
        max_rows (int): Maximum number of rows to display
    """
    if title:
        print(f"\n{title}")

    if max_rows and len(df) > max_rows:
        print(f"\n{df.head(max_rows).to_string()}")
        print(f"\n... ({len(df) - max_rows} more rows not shown)")
    else:
        print(f"\n{df.to_string()}")


def validate_orderbook_data(df: pd.DataFrame) -> bool:
    """
    Validate that the DataFrame has the expected orderbook structure.

    Args:
        df (pd.DataFrame): DataFrame to validate

    Returns:
        bool: True if valid, False otherwise
    """
    required_columns = [
        "datetime",
        "last_price",
        "volume",
        "amount",
        "bid_price1",
        "bid_volume1",
        "ask_price1",
        "ask_volume1",
        "instrument_id",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False

    # Check that bid prices are in descending order
    bid_price_cols = [f"bid_price{i}" for i in range(1, 6)]
    ask_price_cols = [f"ask_price{i}" for i in range(1, 6)]

    # Check a sample of rows for order consistency
    sample_df = df.head(10)

    for idx, row in sample_df.iterrows():
        # Check bid prices (should be descending)
        bid_prices = [row[col] for col in bid_price_cols if pd.notna(row[col])]
        if bid_prices != sorted(bid_prices, reverse=True):
            logger.warning(f"Row {idx}: Bid prices not in descending order")

        # Check ask prices (should be ascending)
        ask_prices = [row[col] for col in ask_price_cols if pd.notna(row[col])]
        if ask_prices != sorted(ask_prices):
            logger.warning(f"Row {idx}: Ask prices not in ascending order")

    logger.success("Data validation completed")
    return True
