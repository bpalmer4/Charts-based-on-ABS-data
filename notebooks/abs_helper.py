"""A collection of functions to make working with ABS data just a littel bit easier."""

import pandas as pd
import readabs as ra
from plotting import set_chart_dir, clear_chart_dir


def get_data(cat: str) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, str, str]:
    """Get ABS data, create plot directories."""

    # get data
    abs_dict_, meta_ = ra.read_abs_cat(cat)
    source_ = f"ABS: {cat}"
    chart_dir = f"./CHARTS/{cat} - {ra.catalogue_map().loc[cat, "Topic"]}/"
    recent_ = "2019-12-01"

    # create plot plot directories
    set_chart_dir(chart_dir)
    clear_chart_dir(chart_dir)

    return abs_dict_, meta_, source_, recent_


# Useful constants for plotting
ANNUAL_CPI_TARGET_RANGE = {
    "ymin": 2,
    "ymax": 3,
    "color": "#dddddd",
    "label": "2-3% annual inflation target range",
    "zorder": -1,
}

ANNUAL_CPI_TARGET = {
    "y": 2.5,
    "color": "gray",
    "linestyle": "--",
    "linewidth": 0.75,
    "label": "2.5% annual inflation target",
    "zorder": -1,
}

MONTHLY_CPI_TARGET = {
    "y": (pow(1.025, 1.0/12.0) - 1) * 100,
    "color": "darkred",
    "linewidth": 0.75,
    "linestyle": "--",
    "label": "Monthly growth consistent with a 2.5% annual inflation target",
    "zorder": -1,
}



