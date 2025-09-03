from pathlib import Path
import re
from typing import Any

from ck_utils import ck_str_keep_trading_hours
import numpy as np
import pandas as pd
import yaml

from mrk_data.ck.connect import gen_client
from mrk_data.ck.template import generate_table_name
from mrk_data.utils.from_vnpy import Exchange, Interval


def load_config(config_name="config") -> Any:
    """Load configuration from YAML file.

    Args:
        config_name (str): Name of the config file (without .yaml extension).
                          If not provided, defaults to "config".
    """
    # Create config directory path
    config_dir = Path("config")

    # Make sure config directory exists
    config_dir.mkdir(exist_ok=True)

    # Construct config file path
    config_path = config_dir / f"{config_name}.yaml"

    # If specific config doesn't exist, try the default one in the root directory
    if not config_path.exists():
        config_path = (
            Path("/home/zhonghao/ht-daily-work/options_scalping_strategy")
            / f"{config_name}.yaml"
        )

    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def fetch_tick_data(instrument_id: str, exchange: Exchange, date: int) -> pd.DataFrame:
    """Fetch tick data for a specific instrument and date.

    >>> df.columns
    Index(['datetime', 'last_price', 'highest', 'lowest', 'average', 'volume',
        'amount', 'open_interest', 'bid_price1', 'bid_volume1', 'ask_price1',
        'ask_volume1', 'bid_price2', 'bid_volume2', 'ask_price2', 'ask_volume2',
        'bid_price3', 'bid_volume3', 'ask_price3', 'ask_volume3', 'bid_price4',
        'bid_volume4', 'ask_price4', 'ask_volume4', 'bid_price5', 'bid_volume5',
        'ask_price5', 'ask_volume5', 'instrument_id', 'localtime'],
        dtype='object')
    """
    with gen_client() as client:
        df = client.query_df(
            f"""
            SELECT
                *
            FROM
                {generate_table_name(symbol=instrument_id, exchange=exchange, interval=Interval.TICK)}
            WHERE
                toYYYYMMDD(datetime) == {date}
            """
        )
    assert not df.empty, "No data found for the specified date."
    df["date"] = df["datetime"].dt.date
    return df


def fetch_tick_data_range(
    instrument_id: str,
    exchange: Exchange,
    start_date: int,
    end_date: int,
    multipier=10000,
) -> pd.DataFrame:
    """Fetch tick data for a specific instrument within a date range.

    Args:
        instrument_id (str): The instrument ID to fetch data for
        exchange (Exchange): The exchange where the instrument is traded
        start_date (int): Start date in YYYYMMDD format
        end_date (int): End date in YYYYMMDD format

    Returns:
        pd.DataFrame: DataFrame containing tick data for the specified date range

    >>> df.columns
    Index(['datetime', 'last_price', 'highest', 'lowest', 'average', 'volume',
        'amount', 'open_interest', 'bid_price1', 'bid_volume1', 'ask_price1',
        'ask_volume1', 'bid_price2', 'bid_volume2', 'ask_price2', 'ask_volume2',
        'bid_price3', 'bid_volume3', 'ask_price3', 'ask_volume3', 'bid_price4',
        'bid_volume4', 'ask_price4', 'ask_volume4', 'bid_price5', 'bid_volume5',
        'ask_price5', 'ask_volume5', 'instrument_id', 'localtime'],
        dtype='object')
    """
    # 正则匹配所有数字，只保留数字
    instrument_id = re.sub(r"\D", "", instrument_id)
    with gen_client() as client:
        df: Any = client.query_df(
            query=f"""
            SELECT
                *
            FROM
                {generate_table_name(symbol=instrument_id, exchange=exchange, interval=Interval.TICK)}
            WHERE
                toYYYYMMDD(datetime) >= {start_date} AND toYYYYMMDD(datetime) <= {end_date}
                AND {ck_str_keep_trading_hours}
            ORDER BY
                datetime
            """
        )
    assert not df.empty, "No data found for the specified date range."

    df["vwap"] = np.where(df["volume"] != 0, df["amount"] / df["volume"] / multipier, 0)
    return df
