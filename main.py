#!/usr/bin/env python3
"""
Main execution script for the orderbook analysis project.
"""

import argparse
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from src.analysis.order_flow_np import calculate_order_book_delta_numpy
from src.data.data_fetching import Exchange, fetch_tick_data_range
from src.utils.config_manager import load_config
from src.utils.data_utils import display_dataframe, validate_orderbook_data
from src.utils.logging_config import setup_logger

# Set up logger
logger = setup_logger("main")


def main(config_path: str):
    """
    Main execution function.

    Args:
        config_path (str): Path to the configuration file
    """
    logger.info("Starting orderbook analysis")

    try:
        # Load configuration
        config = load_config(config_path)

        # Fetch data
        logger.info("Fetching tick data")
        df = fetch_tick_data_range(
            instrument_id=config["data"]["instrument_id"],
            exchange=Exchange.SSE,  # Using mock Exchange class
            start_date=config["data"]["start_date"],
            end_date=config["data"]["end_date"],
        )

        if df is None:
            logger.error("Failed to fetch data")
            return

        # Validate data
        if not validate_orderbook_data(df):
            logger.error("Data validation failed")
            return

        # Perform analysis
        if config["analysis"]["enable_order_flow"]:
            logger.info("Calculating order flow indicators")
            df = calculate_order_book_delta_numpy(df, config["analysis"]["window_size"])

            if df is None:
                logger.error("Order flow calculation failed")
                return

        # Display results
        display_dataframe(df, "Analysis Results", max_rows=20)

        logger.success("Analysis completed successfully")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orderbook Analysis Tool")
    parser.add_argument(
        "--config",
        type=str,
        default="config/orderbook_config.yaml",
        help="Path to the configuration file",
    )

    args = parser.parse_args()
    main(args.config)
