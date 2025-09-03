"""
Main module for orderbook pattern recognition research.
"""

from typing import Any, Dict

from loguru import logger
import pandas as pd

from src.analysis.evaluation_metrics import calculate_metrics, register_metric
from src.analysis.feature_construction import build_features, register_feature
from src.analysis.order_flow_np import calculate_order_book_delta_numpy
from src.analysis.pattern_recognition import identify_patterns, register_pattern
from src.data.data_fetching import fetch_tick_data_range
from src.data.data_processing import prepare_data
from src.utils.config_manager import load_config


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

    return df


# Feature construction functions
@register_feature("order_flow_features")
def construct_order_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construct features based on order flow data."""
    df = df.copy()

    # Calculate order book delta if not already present
    if "bid_delta1" not in df.columns:
        df = calculate_order_book_delta_numpy(df)

    # Calculate cumulative order flow
    if "bid_delta1" in df.columns and "ask_delta1" in df.columns:
        df["net_order_flow"] = df["bid_delta1"] + df["ask_delta1"]
        df["cumulative_order_flow"] = df["net_order_flow"].cumsum()

        # Calculate order flow volatility
        df["order_flow_volatility"] = df["net_order_flow"].rolling(window=20).std()

    return df


# Evaluation metric functions
@register_metric("prediction_accuracy")
def calculate_prediction_accuracy(df: pd.DataFrame) -> float:
    """Calculate the accuracy of pattern predictions."""
    if "large_order_imbalance" in df.columns and "price_movement" in df.columns:
        # For this example, we'll assume price movement is the direction of next tick
        df["price_movement"] = (
            df["last_price"].diff().shift(-1).apply(lambda x: 1 if x > 0 else 0)
        )
        df["prediction"] = df["large_order_imbalance"].apply(lambda x: 1 if x else 0)

        accuracy = (df["prediction"] == df["price_movement"]).mean()
        return accuracy
    return 0.0


def main(config_path: str = "config/orderbook_config.yaml") -> Dict[str, Any]:
    """Main function to run the pattern recognition research pipeline."""
    # Load configuration
    config = load_config(config_path)
    logger.info("Configuration loaded successfully")

    # Fetch data
    data_config = config["data"]
    df = fetch_tick_data_range(
        instrument_id=data_config["instrument_id"],
        exchange=data_config["exchange"],
        start_date=data_config["start_date"],
        end_date=data_config["end_date"],
    )
    logger.info(f"Data fetched successfully, shape: {df.shape}")

    # Prepare data
    df = prepare_data(df)
    logger.info("Data prepared successfully")

    # Identify patterns
    pattern_functions = [identify_large_order_imbalance]
    df = identify_patterns(df, pattern_functions)
    logger.info("Patterns identified successfully")

    # Build features
    feature_functions = [construct_order_flow_features]
    df = build_features(df, feature_functions)
    logger.info("Features constructed successfully")

    # Calculate metrics
    metric_functions = [calculate_prediction_accuracy]
    metrics = calculate_metrics(df, metric_functions)
    logger.info("Metrics calculated successfully")

    # Log results
    logger.info(f"Research completed with metrics: {metrics}")

    return {"data": df, "metrics": metrics}


if __name__ == "__main__":
    result = main()
