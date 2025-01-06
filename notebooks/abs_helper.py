"""A collection of functions to make working with ABS data just a litte bit easier."""

# === imports
from typing import Any
import pandas as pd
import readabs as ra
from plotting import set_chart_dir, clear_chart_dir


# === frequently used data sources


def get_gdp(gdp_type="CP", seasonal="SA") -> tuple[pd.Series, str]:
    """Return the ABS GDP Series (from the key aggregates table).
    Arguments:
    - gdp_type: str - The type of the series to return.
      gdp_types is one of: 'CP' or 'CVM, which stand for
      'Current price' and 'Chain volume measures'.
    - seasonal: str - The seasonal adjustment type.
      seasonal is one of 'SA', 'T', 'O', which stand for
      'Seasonally Adjusted', 'Trend' and 'Original'.
    Returns:
    a tuple comprising the selected GDP series and the units
    of that series."""

    # validate the arguments
    did_cvm = "Gross domestic product: Chain volume measures ;"
    did_cp = "Gross domestic product: Current prices ;"
    gdp_types = {
        # gdp-type: data-item-description in the ABS data
        "CP": did_cp,
        "Current price": did_cp,
        "Current prices": did_cp,
        "CVM": did_cvm,
        "Volumetric": did_cvm,
    }
    seasonals = {
        "SA": "Seasonally Adjusted",
        "S": "Seasonally Adjusted",
        "T": "Trend",
        "O": "Original",
    }
    assert gdp_type in gdp_types, f"Invalid GDP type: {gdp_type}"
    assert seasonal in seasonals, f"Invalid seasonal adjustment type: {seasonal}"

    # get the series
    cat = "5206.0"
    seo = "5206001_Key_Aggregates"
    (
        gdp_data,
        gdp_meta,
    ) = ra.read_abs_cat(cat, single_excel_only=seo, verbose=False)
    selector = {
        gdp_types[gdp_type]: ra.metacol.did,
        seasonals[seasonal]: ra.metacol.stype,
    }
    table, series_id, units = ra.find_abs_id(gdp_meta, selector, verbose=False)
    gdp = gdp_data[table][series_id]

    return gdp, units


# === data retrieval and initialisation
def get_abs_data(
    cat: str,
    **kwargs: Any,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, str, str]:
    """Get ABS data for a specific catalogue number and create plot directories.
    My standard set-up for notebooks that use ABS data.

    Argument: an ABS catalogue number (as a string, eg. "6401.0")
    Keyword arguments: any additional arguments to pass to read_abs_cat

    Returns: the data in a dictionary, metadata, source and a recent date
    to plot from."""

    # get data -
    abs_dict_, meta_ = ra.read_abs_cat(cat, **kwargs)
    source_ = f"ABS: {cat}"
    recent_ = "2020-12-01"

    # create plot plot directories
    chart_dir = f"./CHARTS/{cat} - {ra.abs_catalogue().loc[cat, "Topic"]}/"
    set_chart_dir(chart_dir)
    clear_chart_dir(chart_dir)

    return abs_dict_, meta_, source_, recent_


# === Useful constants for plotting
# CPI targets
ANNUAL_CPI_TARGET = {
    "y": 2.5,
    "color": "gray",
    "linestyle": "--",
    "linewidth": 0.75,
    "label": "2.5% annual inflation target",
    "zorder": -1,
}

ANNUAL_CPI_TARGET_RANGE = {
    "ymin": 2,
    "ymax": 3,
    "color": "#dddddd",
    "label": "2-3% annual inflation target range",
    "zorder": -1,
}

QUARTERLY_CPI_TARGET = {
    "y": (pow(1.025, 0.25) - 1) * 100,
    "linestyle": "dashed",
    "linewidth": 0.75,
    "color": "darkred",
    "label": "Quarterly growth consistent with 2.5% annual inflation",
}

QUARTERLY_CPI_RANGE = {
    "ymin": (pow(1.02, 0.25) - 1) * 100,
    "ymax": (pow(1.03, 0.25) - 1) * 100,
    "color": "#ffdddd",
    "label": "Quarterly growth consistent with 2-3% annual inflation target",
    "zorder": -1,
}

MONTHLY_CPI_TARGET = {
    "y": (pow(1.025, 1.0 / 12.0) - 1) * 100,
    "color": "darkred",
    "linewidth": 0.75,
    "linestyle": "--",
    "label": "Monthly growth consistent with a 2.5% annual inflation target",
    "zorder": -1,
}
