from bokeh.layouts import column
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, CrosshairTool, HoverTool, Scatter
from bokeh.plotting import figure, output_file, save
from bokeh.palettes import Category10
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any


def _create_plot_title(
    base_title, opt_symbol=None, udly_symbol=None, plot_type="option", date=None
):
    """Helper function to create consistent plot titles."""
    title = base_title
    if opt_symbol and udly_symbol:
        if plot_type == "option":
            title = f"{base_title} ({opt_symbol})"
        else:
            title = f"{base_title} ({udly_symbol})"
    elif opt_symbol:
        title = f"{base_title} ({opt_symbol})"
    elif udly_symbol:
        title = f"{base_title} ({udly_symbol})"

    if date:
        title += f" - {date}"

    return title


def create_source(df: pd.DataFrame):
    """Create a ColumnDataSource from the dataframe."""
    return ColumnDataSource(df)


def create_bid_ask_plot(
    source: ColumnDataSource,
    x_range=None,
    opt_symbol=None,
    udly_symbol=None,
    plot_type="option",
    date=None,
) -> Any:
    """Create the bid and ask prices plot."""
    # Create title with symbol names if provided
    title = _create_plot_title(
        "Bid and Ask Prices", opt_symbol, udly_symbol, plot_type, date
    )

    # Filter out x_range from kwargs if it's None
    figure_kwargs = {
        "title": title,
        "x_axis_type": "datetime",
        "sizing_mode": "stretch_width",
        "height": 300,
        "tools": "pan,wheel_zoom,box_zoom,reset,save",
    }

    # Only add x_range if it's not None
    if x_range is not None:
        figure_kwargs["x_range"] = x_range

    p1 = figure(**figure_kwargs)

    # Add crosshair tool
    crosshair1 = CrosshairTool(
        line_color="gray",
        line_alpha=0.5,
        line_width=1,
    )
    p1.add_tools(crosshair1)

    # Plot bid price (using bid_price1)
    p1.line(
        x="datetime",
        y="bid_price1",
        source=source,
        color="#1f77b4",  # Blue
        legend_label="Bid Price",
        line_width=2,
    )

    # Plot ask price (using ask_price1)
    p1.line(
        x="datetime",
        y="ask_price1",
        source=source,
        color="#e377c2",  # Pink color
        legend_label="Ask Price",
        line_width=2,
    )

    # Plot mid price
    p1.line(
        x="datetime",
        y="mid_price",
        source=source,
        color="#2ca02c",  # Green
        legend_label="Mid Price",
        line_width=2,
    )

    p1.line(
        x="datetime",
        y="vwmid_price",
        source=source,
        color="#d62728",  # Red
        legend_label="VWMid Price",
        line_width=2,
    )

    # Add hover tool with millisecond precision
    hover1 = HoverTool(
        tooltips=[
            ("Time", "@datetime{%H:%M:%S.%3N}"),
            ("Ask Price2", "@ask_price2{0.0000}"),
            ("Ask Price1", "@ask_price1{0.0000}"),
            ("Bid Price1", "@bid_price1{0.0000}"),
            ("Bid Price2", "@bid_price2{0.0000}"),
            ("Ask Vol2", "@ask_volume2{0,0}"),
            ("Ask Vol1", "@ask_volume1{0,0}"),
            ("Bid Vol1", "@bid_volume1{0,0}"),
            ("Bid Vol2", "@bid_volume2{0,0}"),
            ("Volume", "@volume{0,0}"),
            ("Mid Price", "@mid_price{0.0000}"),
            ("VWMid Price", "@vwmid_price{0.0000}"),
            ("VWAP", "@vwap{0.000000}"),
            ("Amount", "@amount{0.000000}"),
        ],
        formatters={"@datetime": "datetime"},
    )
    p1.add_tools(hover1)
    p1.legend.location = "top_left"

    return p1


def create_volume_plot(
    source: ColumnDataSource,
    x_range=None,
    opt_symbol=None,
    udly_symbol=None,
    plot_type="option",
    date=None,
    sum_of_vol=None,
) -> Any:
    """Create the volume plot."""
    # Create title with symbol names if provided
    title = _create_plot_title("Volume", opt_symbol, udly_symbol, plot_type, date)
    title += f" - Sum of Vol: {sum_of_vol if sum_of_vol else 'N/A'}"

    p2 = figure(
        title=title,
        x_axis_type="datetime",
        x_range=x_range,
        sizing_mode="stretch_width",
        height=200,
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )

    # Add crosshair tool
    crosshair2 = CrosshairTool(
        line_color="gray",
        line_alpha=0.5,
        line_width=1,
    )
    p2.add_tools(crosshair2)

    # Plot volume as bars
    p2.vbar(
        x="datetime",
        top="volume",
        width=pd.Timedelta(milliseconds=500),
        source=source,
        color="#9467bd",  # Purple color for bars
        alpha=0.7,
    )

    # Add hover tool with millisecond precision
    hover2 = HoverTool(
        tooltips=[("Time", "@datetime{%H:%M:%S.%3N}"), ("Volume", "@volume{0,0}")],
        formatters={"@datetime": "datetime"},
    )
    p2.add_tools(hover2)

    return p2


def create_cumulative_volume_plot(
    source: ColumnDataSource,
    opt_symbol=None,
    udly_symbol=None,
    date=None,
) -> Any:
    """Create the cumulative volume plot."""
    # Create title with symbol names if provided
    title = _create_plot_title(
        "Cumulative Volume",
        opt_symbol,
        udly_symbol,
        plot_type="option",
        date=date,
    )

    p3 = figure(
        title=title,
        x_axis_type="datetime",
        sizing_mode="stretch_width",
        height=200,
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )

    # Add crosshair tool
    crosshair3 = CrosshairTool(
        line_color="gray",
        line_alpha=0.5,
        line_width=1,
    )
    p3.add_tools(crosshair3)

    # Calculate cumulative volume
    volume_data = source.data.get("volume", [])
    datetime_data = source.data.get("datetime", [])

    if len(volume_data) > 0 and len(datetime_data) > 0:
        # Convert to numpy arrays for calculation
        volume_array = np.array(volume_data)
        datetime_array = np.array(datetime_data)

        # Calculate cumulative volume
        cumulative_volume = np.cumsum(volume_array)

        # Create new source for cumulative data
        cumulative_source = ColumnDataSource(
            {"datetime": datetime_array, "cumulative_volume": cumulative_volume}
        )

        # Plot cumulative volume as line
        p3.line(
            x="datetime",
            y="cumulative_volume",
            source=cumulative_source,
            color="#ff7f0e",  # Orange color
            legend_label="Cumulative Volume",
            line_width=2,
        )

        # Add hover tool
        hover3 = HoverTool(
            tooltips=[
                ("Time", "@datetime{%H:%M:%S.%3N}"),
                ("Cumulative Volume", "@cumulative_volume{0,0}"),
            ],
            formatters={"@datetime": "datetime"},
        )
        p3.add_tools(hover3)

        p3.yaxis.axis_label = "Cumulative Volume"
        p3.xaxis.axis_label = "Time"
        p3.legend.location = "top_left"
    else:
        # Handle case with no data
        p3.text(
            x=[0],
            y=[0],
            text=["No volume data available"],
            text_align="center",
            text_baseline="middle",
        )

    return p3


def create_bid_ask_volume_delta_plot(
    source: ColumnDataSource,
    x_range=None,
    opt_symbol=None,
    udly_symbol=None,
    plot_type="option",
    date=None,
) -> Any:
    """Create the bid-ask volume delta plot over time."""
    # Create title with symbol names if provided
    title = _create_plot_title(
        "Bid and Ask Deltas", opt_symbol, udly_symbol, plot_type, date
    )

    p4 = figure(
        title=title,
        x_axis_type="datetime",
        x_range=x_range,
        sizing_mode="stretch_width",
        height=200,
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )

    # Add crosshair tool
    crosshair4 = CrosshairTool(
        line_color="gray",
        line_alpha=0.5,
        line_width=1,
    )
    p4.add_tools(crosshair4)

    # Plot bid_delta1 and ask_delta1 as lines with attractive colors
    p4.line(
        x="datetime",
        y="bid_delta1",
        source=source,
        color="#1f77b4",  # Blue color
        legend_label="Bid Delta",
        line_width=2,
    )

    p4.line(
        x="datetime",
        y="ask_delta1",
        source=source,
        color="#e377c2",  # Pink color
        legend_label="Ask Delta",
        line_width=2,
    )

    # Add horizontal line at y=0 for reference
    p4.line(
        x="datetime",
        y=0,
        source=source,
        color="gray",
        line_width=1,
        line_dash="dashed",
        alpha=0.5,
    )

    # Add hover tool with millisecond precision
    hover4 = HoverTool(
        tooltips=[
            ("Time", "@datetime{%H:%M:%S.%3N}"),
            ("Bid Delta", "@bid_delta1{0,0}"),
            ("Ask Delta", "@ask_delta1{0,0}"),
        ],
        formatters={"@datetime": "datetime"},
    )
    p4.add_tools(hover4)

    # Position the legend
    p4.legend.location = "top_left"

    # Set y-axis label
    p4.yaxis.axis_label = "Volume Delta"

    return p4


def create_volume_and_delta_plot(
    source: ColumnDataSource,
    x_range=None,
    opt_symbol=None,
    udly_symbol=None,
    plot_type="option",
    date=None,
    sum_of_vol=None,
) -> Any:
    """Create a combined plot with volume bars and bid-ask delta lines."""
    # Create title with symbol names if provided
    title = _create_plot_title(
        "Volume and Bid-Ask Deltas", opt_symbol, udly_symbol, plot_type, date
    )
    title += f" - Sum of Vol: {sum_of_vol if sum_of_vol else 'N/A'}"

    # Create figure with shared y-axis
    p = figure(
        title=title,
        x_axis_type="datetime",
        x_range=x_range,
        sizing_mode="stretch_width",
        height=400,
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )

    # Add crosshair tool
    crosshair = CrosshairTool(
        line_color="gray",
        line_alpha=0.5,
        line_width=1,
    )
    p.add_tools(crosshair)

    # Plot volume as bars (as in create_volume_plot)
    volume_renderer = p.vbar(
        x="datetime",
        top="volume",
        width=pd.Timedelta(milliseconds=500),
        source=source,
        color="#9467bd",  # Purple color for bars
        alpha=0.7,
        legend_label="Volume",
    )

    # Create second y-axis for bid-ask delta lines
    p.extra_y_ranges = {"delta": p.y_range}
    p.extra_y_ranges["volume1"] = p.y_range

    # Plot bid_delta1 and ask_delta1 as solid lines with attractive colors (no longer dashed)
    bid_delta_renderer = p.line(
        x="datetime",
        y="bid_delta1",
        source=source,
        color="#1f77b4",  # Blue color
        legend_label="Bid Delta",
        line_width=2,
        y_range_name="delta",
    )

    ask_delta_renderer = p.line(
        x="datetime",
        y="ask_delta1",
        source=source,
        color="#e377c2",  # Pink color
        legend_label="Ask Delta",
        line_width=2,
        y_range_name="delta",
    )

    # Plot bid_volume1 and ask_volume1 as solid lines with darker colors on the same axis as volume
    bid_volume_renderer = p.line(
        x="datetime",
        y="bid_volume1",
        source=source,
        color="#0b3d91",  # Darker blue color
        legend_label="Bid Volume1",
        line_width=2,
        y_range_name="volume1",
    )

    ask_volume_renderer = p.line(
        x="datetime",
        y="ask_volume1",
        source=source,
        color="#c71585",  # Darker pink color
        legend_label="Ask Volume1",
        line_width=2,
        y_range_name="volume1",
    )

    # Add horizontal line at y=0 for reference
    p.line(
        x="datetime",
        y=0,
        source=source,
        color="gray",
        line_width=1,
        line_dash="dashed",
        alpha=0.5,
        y_range_name="delta",
    )

    # Add unified hover tool with millisecond precision showing all values
    unified_hover = HoverTool(
        tooltips=[
            ("Time", "@datetime{%H:%M:%S.%3N}"),
            ("Volume", "@volume{0,0}"),
            ("Bid Volume1", "@bid_volume1{0,0}"),
            ("Ask Volume1", "@ask_volume1{0,0}"),
            ("Bid Delta", "@bid_delta1{0,0}"),
            ("Ask Delta", "@ask_delta1{0,0}"),
        ],
        formatters={"@datetime": "datetime"},
    )

    p.add_tools(unified_hover)

    # Position the legend
    p.legend.location = "top_left"

    # Enable legend click policy to hide/show glyphs
    p.legend.click_policy = "hide"

    # Set y-axis labels
    p.yaxis.axis_label = "Volume/Volume Delta"

    return p


def create_layout(p1: Any, p2: Any, p3: Any = None, p4: Any = None) -> Any:
    """Create the combined layout."""
    if p4 is not None:
        return column(p1, p2, p3, p4, spacing=20, sizing_mode="stretch_width")
    elif p3 is not None:
        return column(p1, p2, p3, spacing=20, sizing_mode="stretch_width")
    else:
        return column(p1, p2, spacing=20, sizing_mode="stretch_width")


def save_plot(layout: Any, filename: str) -> None:
    """Save the plot to an HTML file."""
    # Extract symbol names from filename for plot title
    title = "Bid Ask Volume Plot"
    if "_" in filename and "." in filename:
        symbol_part = filename.split("_")[2].split(".")[0]  # Extract symbols part
        if symbol_part:
            title = f"Bid Ask Volume Plot ({symbol_part})"

    output_file(filename=filename, title=title)
    save(layout)


def add_pattern_markers(
    plot_figure: Any,
    pattern_data: pd.DataFrame,
    marker_config: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Add scatter markers to indicate identified scalping patterns on price plots.

    Args:
        plot_figure: The bokeh figure to add markers to
        pattern_data: DataFrame containing pattern detection results
        marker_config: Configuration for marker appearance

    Returns:
        The updated figure with pattern markers
    """
    if pattern_data.empty:
        return plot_figure

    # Default marker configuration
    default_config = {
        "price_before_bid_move": {
            "marker": "triangle",
            "size": 12,
            "color": "#ff4444",  # Red for bid moves (down)
            "alpha": 0.8,
            "legend_label": "Price Before Bid Move",
        },
        "price_after_bid_move": {
            "marker": "inverted_triangle",
            "size": 12,
            "color": "#ff8888",  # Light red for after bid moves
            "alpha": 0.8,
            "legend_label": "Price After Bid Move",
        },
        "price_before_ask_move": {
            "marker": "triangle",
            "size": 12,
            "color": "#44ff44",  # Green for ask moves (up)
            "alpha": 0.8,
            "legend_label": "Price Before Ask Move",
        },
        "price_after_ask_move": {
            "marker": "inverted_triangle",
            "size": 12,
            "color": "#88ff88",  # Light green for after ask moves
            "alpha": 0.8,
            "legend_label": "Price After Ask Move",
        },
        "complete_scalping": {
            "marker": "diamond",
            "size": 20,
            "color": "#ffaa00",  # Orange for complete cycles
            "alpha": 0.9,
            "legend_label": "Complete Scalping",
        },
    }

    if marker_config:
        # Update default config with user provided config
        for key, value in marker_config.items():
            if key in default_config:
                default_config[key].update(value)

    # Add price before bid move markers
    if (
        "bid_timestamp" in pattern_data.columns
        and "price_before_bid_move" in pattern_data.columns
    ):
        bid_source = ColumnDataSource(
            data=dict(
                timestamp=pd.to_datetime(pattern_data["bid_timestamp"]).tolist(),
                price=pattern_data["price_before_bid_move"].tolist(),
                excess_volume=[pattern_data.get("excess_ask_volume", 0)]
                * len(pattern_data),
                symbol=pattern_data.get(
                    "bid_symbol", pd.Series([""] * len(pattern_data))
                )
                .astype(str)
                .tolist(),
            )
        )

        config = default_config["price_before_bid_move"]
        plot_figure.scatter(
            x="timestamp",
            y="price",
            source=bid_source,
            marker=config["marker"],
            size=config["size"],
            color=config["color"],
            alpha=config["alpha"],
            legend_label=config["legend_label"],
        )

        # Add hover for price before bid moves
        bid_hover = HoverTool(
            tooltips=[
                ("Time", "@timestamp{%H:%M:%S.%3N}"),
                ("Price", "@price{0.0000}"),
                ("Excess Volume", "@excess_volume{0,0}"),
                ("Symbol", "@symbol"),
            ],
            formatters={"@timestamp": "datetime"},
            renderers=[plot_figure.renderers[-1]],  # Only for the last added renderer
        )
        plot_figure.add_tools(bid_hover)

    # Add price after bid move markers
    if (
        "bid_timestamp" in pattern_data.columns
        and "price_after_bid_move" in pattern_data.columns
    ):
        bid_after_source = ColumnDataSource(
            data=dict(
                timestamp=pd.to_datetime(pattern_data["bid_timestamp"]).tolist(),
                price=pattern_data["price_after_bid_move"].tolist(),
                excess_volume=[pattern_data.get("excess_ask_volume", 0)]
                * len(pattern_data),
                symbol=pattern_data.get(
                    "bid_symbol", pd.Series([""] * len(pattern_data))
                )
                .astype(str)
                .tolist(),
            )
        )

        config = default_config["price_after_bid_move"]
        plot_figure.scatter(
            x="timestamp",
            y="price",
            source=bid_after_source,
            marker=config["marker"],
            size=config["size"],
            color=config["color"],
            alpha=config["alpha"],
            legend_label=config["legend_label"],
        )

        # Add hover for price after bid moves
        bid_after_hover = HoverTool(
            tooltips=[
                ("Time", "@timestamp{%H:%M:%S.%3N}"),
                ("Price", "@price{0.0000}"),
                ("Excess Volume", "@excess_volume{0,0}"),
                ("Symbol", "@symbol"),
            ],
            formatters={"@timestamp": "datetime"},
            renderers=[plot_figure.renderers[-1]],  # Only for the last added renderer
        )
        plot_figure.add_tools(bid_after_hover)

    # Add price before ask move markers
    if (
        "ask_timestamp" in pattern_data.columns
        and "price_before_ask_move" in pattern_data.columns
    ):
        ask_source = ColumnDataSource(
            data=dict(
                timestamp=pd.to_datetime(pattern_data["ask_timestamp"]).tolist(),
                price=pattern_data["price_before_ask_move"].tolist(),
                consumed_volume=[pattern_data.get("volume_consumed", 0)]
                * len(pattern_data),
                symbol=pattern_data.get(
                    "ask_symbol", pd.Series([""] * len(pattern_data))
                )
                .astype(str)
                .tolist(),
            )
        )

        config = default_config["price_before_ask_move"]
        plot_figure.scatter(
            x="timestamp",
            y="price",
            source=ask_source,
            marker=config["marker"],
            size=config["size"],
            color=config["color"],
            alpha=config["alpha"],
            legend_label=config["legend_label"],
        )

        # Add hover for price before ask moves
        ask_hover = HoverTool(
            tooltips=[
                ("Time", "@timestamp{%H:%M:%S.%3N}"),
                ("Price", "@price{0.0000}"),
                ("Consumed Volume", "@consumed_volume{0,0}"),
                ("Symbol", "@symbol"),
            ],
            formatters={"@timestamp": "datetime"},
            renderers=[plot_figure.renderers[-1]],  # Only for the last added renderer
        )
        plot_figure.add_tools(ask_hover)

    # Add price after ask move markers
    if (
        "ask_timestamp" in pattern_data.columns
        and "price_after_ask_move" in pattern_data.columns
    ):
        ask_after_source = ColumnDataSource(
            data=dict(
                timestamp=pd.to_datetime(pattern_data["ask_timestamp"]).tolist(),
                price=pattern_data["price_after_ask_move"].tolist(),
                consumed_volume=[pattern_data.get("volume_consumed", 0)]
                * len(pattern_data),
                symbol=pattern_data.get(
                    "ask_symbol", pd.Series([""] * len(pattern_data))
                )
                .astype(str)
                .tolist(),
            )
        )

        config = default_config["price_after_ask_move"]
        plot_figure.scatter(
            x="timestamp",
            y="price",
            source=ask_after_source,
            marker=config["marker"],
            size=config["size"],
            color=config["color"],
            alpha=config["alpha"],
            legend_label=config["legend_label"],
        )

        # Add hover for price after ask moves
        ask_after_hover = HoverTool(
            tooltips=[
                ("Time", "@timestamp{%H:%M:%S.%3N}"),
                ("Price", "@price{0.0000}"),
                ("Consumed Volume", "@consumed_volume{0,0}"),
                ("Symbol", "@symbol"),
            ],
            formatters={"@timestamp": "datetime"},
            renderers=[plot_figure.renderers[-1]],  # Only for the last added renderer
        )
        plot_figure.add_tools(ask_after_hover)

    # Add complete scalping cycle markers (highlight cycles that completed successfully)
    if "is_complete_scalping" in pattern_data.columns:
        complete_patterns = pattern_data[pattern_data["is_complete_scalping"] == True]

        if not complete_patterns.empty and "bid_timestamp" in complete_patterns.columns:
            complete_source = ColumnDataSource(
                data=dict(
                    timestamp=pd.to_datetime(
                        complete_patterns["bid_timestamp"]
                    ).tolist(),
                    price=complete_patterns["price_before_bid_move"].tolist(),
                    time_gap=[complete_patterns.get("time_gap_2_move", 0)]
                    * len(complete_patterns),
                    total_volume=[complete_patterns.get("total_scalping_volume", 0)]
                    * len(complete_patterns),
                    symbol=complete_patterns.get(
                        "bid_symbol", pd.Series([""] * len(complete_patterns))
                    )
                    .astype(str)
                    .tolist(),
                )
            )

            config = default_config["complete_scalping"]
            plot_figure.scatter(
                x="timestamp",
                y="price",
                source=complete_source,
                marker=config["marker"],
                size=config["size"],
                color=config["color"],
                alpha=config["alpha"],
                legend_label=config["legend_label"],
            )

            # Add hover for complete scalping cycles
            complete_hover = HoverTool(
                tooltips=[
                    ("Time", "@timestamp{%H:%M:%S.%3N}"),
                    ("Price", "@price{0.0000}"),
                    ("Time Gap", "@time_gap{0.0}s"),
                    ("Total Volume", "@total_volume{0,0}"),
                    ("Symbol", "@symbol"),
                ],
                formatters={"@timestamp": "datetime"},
                renderers=[
                    plot_figure.renderers[-1]
                ],  # Only for the last added renderer
            )
            plot_figure.add_tools(complete_hover)

    return plot_figure


def create_scalping_pattern_markers_source(
    pattern_data: pd.DataFrame, pattern_type: str = "merged"
) -> Dict[str, ColumnDataSource]:
    """
    Create ColumnDataSource objects for different types of scalping pattern markers.

    Args:
        pattern_data: DataFrame with pattern data
        pattern_type: Type of patterns ("bid", "ask", "merged")

    Returns:
        Dictionary of ColumnDataSource objects for each marker type
    """
    sources = {}

    if pattern_type == "bid" and not pattern_data.empty:
        # Create source for bid patterns
        sources["bid_patterns"] = ColumnDataSource(
            {
                "timestamp": pd.to_datetime(pattern_data["timestamp"]),
                "price": pattern_data["price_before_bid_move"],
                "delta": pattern_data["bid_delta1"],
                "excess_volume": pattern_data.get("excess_ask_volume", 0),
                "symbol": pattern_data["symbol"],
            }
        )

    elif pattern_type == "ask" and not pattern_data.empty:
        # Create source for ask patterns
        sources["ask_patterns"] = ColumnDataSource(
            {
                "timestamp": pd.to_datetime(pattern_data["timestamp"]),
                "price": pattern_data["price_before_ask_move"],
                "delta": pattern_data["ask_delta1"],
                "consumed_volume": pattern_data.get("volume_consumed", 0),
                "symbol": pattern_data["symbol"],
            }
        )

    elif pattern_type == "merged" and not pattern_data.empty:
        # Create sources for merged patterns
        if "bid_timestamp" in pattern_data.columns:
            sources["bid_moves"] = ColumnDataSource(
                {
                    "timestamp": pd.to_datetime(pattern_data["bid_timestamp"]),
                    "price": pattern_data["price_before_bid_move"],
                    "excess_volume": pattern_data.get("excess_ask_volume", 0),
                    "symbol": pattern_data.get("bid_symbol", ""),
                }
            )

        if "ask_timestamp" in pattern_data.columns:
            sources["ask_moves"] = ColumnDataSource(
                {
                    "timestamp": pd.to_datetime(pattern_data["ask_timestamp"]),
                    "price": pattern_data["price_after_ask_move"],
                    "consumed_volume": pattern_data.get("volume_consumed", 0),
                    "symbol": pattern_data.get("ask_symbol", ""),
                }
            )

        # Complete scalping cycles
        if "is_complete_scalping" in pattern_data.columns:
            complete_patterns = pattern_data[
                pattern_data["is_complete_scalping"] == True
            ]
            if not complete_patterns.empty:
                sources["complete_scalping"] = ColumnDataSource(
                    {
                        "timestamp": pd.to_datetime(complete_patterns["bid_timestamp"]),
                        "price": complete_patterns["price_before_bid_move"],
                        "time_gap": complete_patterns.get("time_gap_2_move", 0),
                        "total_volume": complete_patterns.get(
                            "total_scalping_volume", 0
                        ),
                        "symbol": complete_patterns.get("bid_symbol", ""),
                    }
                )

    return sources
