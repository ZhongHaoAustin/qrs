import importlib
import sys

from src.data.data_fetching import (
    fetch_tick_data_range,
    load_config,
)
from src.data.data_processing import (
    add_calculated_columns,
    filter_daily_trading_hours,
    merge_and_analyze_option_underlying_data,
)
from src.visualization.plotting import (
    add_pattern_markers,
    create_bid_ask_plot,
    create_cumulative_volume_plot,
    create_layout,
    create_source,
    create_volume_and_delta_plot,
    save_plot,
)
from vnpy.trader.constant import Exchange


def load_symbol_info(udly_symbol):
    """Load symbol information."""
    query_function_name = f"query_new_{udly_symbol}_option_symbols_info"
    try:
        # Import the query function dynamically
        sym_info_module = importlib.import_module("mrk_data.tq.option_sym_info")
        query_function = getattr(sym_info_module, query_function_name)
    except (ImportError, AttributeError):
        raise ImportError(f"Could not find query function: {query_function_name}")

    sym_info = query_function()
    return sym_info


def extract_option_details(sym_info, opt_symbol):
    """Extract option details from symbol information."""
    option_details = sym_info[sym_info["instrument_id"].str.contains(opt_symbol)]

    if not option_details.empty:
        # Extract option details
        relevant_option = option_details.iloc[0]  # Take the first match if multiple
        expire_datetime = relevant_option["expire_datetime"]
        strike_price = relevant_option["strike_price"]
        option_class = relevant_option["option_class"]

        expire_date_str = expire_datetime.strftime("%Y%m%d")
    else:
        # Default values if option details not found
        expire_date_str = "unknown"
        strike_price = "unknown"
        option_class = "unknown"

    return expire_date_str, strike_price, option_class


def process_daily_data(merged_data):
    """Process data by date."""
    merged_data["date"] = merged_data["datetime_opt"].dt.date
    return merged_data.groupby("date")


def create_daily_data_subsets(group_data):
    """Create subsets for option and underlying data."""
    opt_columns = [col for col in group_data.columns if col.endswith("_opt")]
    udly_columns = [col for col in group_data.columns if col.endswith("_udly")]

    # Create option data subset
    opt_data_date = group_data[opt_columns].copy()
    opt_data_date.columns = [col.replace("_opt", "") for col in opt_data_date.columns]
    opt_data_date["datetime"] = group_data["datetime_opt"]

    # Create underlying data subset
    udly_data_date = group_data[udly_columns].copy()
    udly_data_date.columns = [
        col.replace("_udly", "") for col in udly_data_date.columns
    ]
    udly_data_date["datetime"] = group_data["datetime_udly"]

    return opt_data_date, udly_data_date


def create_data_sources(opt_data_date, udly_data_date):
    """Create data sources for plotting."""
    # 在函数内部导入，避免循环导入
    from src.analysis.order_flow_np import (
        calculate_order_book_delta_numpy,
    )

    # Calculate order book delta for option data
    opt_data_with_delta = calculate_order_book_delta_numpy(opt_data_date, max_level=2)
    opt_source = create_source(opt_data_with_delta)
    udly_source = create_source(udly_data_date.dropna())
    return opt_source, udly_source


def generate_plots(opt_source, opt_data_date, opt_symbol, udly_symbol, date):
    """Generate plots for the given data."""
    # Option plots
    p1 = create_bid_ask_plot(
        opt_source,
        opt_symbol=opt_symbol,
        udly_symbol=udly_symbol,
        plot_type="option",
        date=date,
    )

    # Load pattern data and add pattern markers to the bid-ask plot
    try:
        # Load pattern data from CSV file
        pattern_file = f"output/data/patterns/merged/merged_scalping_patterns_time_delta_80_20250801_20250826.csv"
        import pandas as pd

        pattern_data = pd.read_csv(pattern_file)
        # Filter pattern data for the current date
        pattern_data["bid_timestamp"] = pd.to_datetime(pattern_data["bid_timestamp"])
        pattern_data["ask_timestamp"] = pd.to_datetime(pattern_data["ask_timestamp"])
        current_date = pd.to_datetime(date)
        pattern_data_for_date = pattern_data[
            (pattern_data["bid_timestamp"].dt.date == current_date.date())
            | (pattern_data["ask_timestamp"].dt.date == current_date.date())
        ]
        # Add pattern markers to the plot
        p1 = add_pattern_markers(p1, pattern_data_for_date)
    except Exception as e:
        print(f"Warning: Could not add pattern markers: {e}")

    # Cumulative volume plot
    p3 = create_cumulative_volume_plot(
        opt_source,
        opt_symbol=opt_symbol,
        udly_symbol=udly_symbol,
        date=date,
    )

    # Combined volume and delta plot (replaces p2 and p4)
    p2_combined = create_volume_and_delta_plot(
        opt_source,
        p1.x_range,
        opt_symbol=opt_symbol,
        udly_symbol=udly_symbol,
        plot_type="option",
        date=date,
        sum_of_vol=opt_data_date.volume.sum(),
    )

    # We only need 3 plots now (p1, p2_combined, p3), removing p4 (bid-ask volume delta plot)
    return p1, p2_combined, p3


def save_daily_plot(
    p1,
    p2_combined,
    p3,
    opt_symbol,
    udly_symbol,
    strike_price,
    option_class,
    expire_date_str,
    date,
):
    """Save the plot for the given date."""
    # Create layout with all plots - now only 3 plots
    layout = create_layout(p1, p2_combined, p3)

    # Save plot with symbol names, option details and date in filename
    output_filename = (
        f"options_scalping_strategy_{opt_symbol}_{udly_symbol}_"
        f"{strike_price}_{option_class}_{expire_date_str}_{date.strftime('%Y%m%d')}.html"
    )
    save_plot(layout, output_filename)


def main(config_name="config"):
    """Main function to orchestrate the data fetching and plotting."""
    # Load and parse configuration
    config = load_config(config_name)

    opt_symbol = config["opt_symbol"]
    opt_exchange_str = config["opt_exchange"]
    udly_symbol = config["udly_symbol"]
    udly_exchange_str = config["udly_exchange"]
    start_date = config["start_date"]
    end_date = config["end_date"]

    # Convert exchange strings to Exchange objects
    opt_exchange = getattr(Exchange, opt_exchange_str)
    udly_exchange = getattr(Exchange, udly_exchange_str)

    # Fetch data using date range from config
    opt_data = fetch_tick_data_range(
        opt_symbol,
        opt_exchange,
        start_date,
        end_date,
    )
    udly_data = fetch_tick_data_range(
        udly_symbol,
        udly_exchange,
        start_date,
        end_date,
    )

    # Add calculated columns to both datasets
    opt_data = add_calculated_columns(opt_data)
    udly_data = add_calculated_columns(udly_data)

    # Process data - round datetime to nearest 500ms, remove duplicates, merge and calculate delta
    merged_data = merge_and_analyze_option_underlying_data(
        opt_data, udly_data, opt_symbol, udly_symbol
    )

    # Get symbol information based on udly_symbol
    sym_info = load_symbol_info(udly_symbol)

    # Find the specific option in sym_info
    expire_date_str, strike_price, option_class = extract_option_details(
        sym_info, opt_symbol
    )

    # Group data by date and generate separate plots for each date
    grouped = process_daily_data(merged_data)

    for date, group_data in grouped:
        # Filter out non-trading hours for this specific date
        group_data = filter_daily_trading_hours(group_data)

        # Separate data for option and underlying for this date
        opt_data_date, udly_data_date = create_daily_data_subsets(group_data)

        # Create sources
        opt_source, udly_source = create_data_sources(opt_data_date, udly_data_date)

        # Create plots with symbol names
        p1, p2, p3 = generate_plots(
            opt_source,
            opt_data_date,
            opt_symbol,
            udly_symbol,
            date,
        )

        # Save plot
        save_daily_plot(
            p1,
            p2,
            p3,
            opt_symbol,
            udly_symbol,
            strike_price,
            option_class,
            expire_date_str,
            date,
        )

        print(f"Generated plot for {date}")


if __name__ == "__main__":
    config_name = sys.argv[1] if len(sys.argv) > 1 else "config"
    main(config_name)
