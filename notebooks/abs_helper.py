"""Helpers for fetching ABS data and standard notebook set-up.

Provides get_abs_data (the standard fetch plus chart-directory set-up),
collate_summary_data (build a summary table for mgplot.summary_plot), and the
CPI target constants used across the inflation notebooks.
"""

# === imports
from typing import Any

import readabs as ra
from mgplot import clear_chart_dir, set_chart_dir
from pandas import DataFrame
from readabs import metacol as mc


# === data retrieval and initialisation
def get_abs_data(
    cat: str,
    chart_dir_suffix: str = "",
    **kwargs: Any,
) -> tuple[dict[str, DataFrame], DataFrame, str, str]:
    """Fetch ABS data for a catalogue number and set up its chart directory.

    The standard set-up for notebooks that use ABS data: fetch the catalogue,
    derive the source label and a recent start date, and point the chart
    directory at CHARTS/<topic>/ (clearing it first).

    Args:
        cat: an ABS catalogue number (as a string, e.g. "6401.0").
        chart_dir_suffix: optional suffix appended to the chart directory name
            (useful for splitting notebooks, e.g. " - Productivity").
        **kwargs: any additional arguments passed to read_abs_cat.

    Returns:
        A tuple of the data dictionary, metadata, source string and a recent
        date to plot from.

    """
    # get data -
    abs_dict_, meta_ = ra.read_abs_cat(cat, **kwargs)
    source_ = f"ABS: {cat}"
    recent_ = "2020-12-01"

    # create plot plot directories
    chart_dir = f"./CHARTS/{cat} - {ra.abs_catalogue().loc[cat, "Topic"]}{chart_dir_suffix}/"
    set_chart_dir(chart_dir)
    clear_chart_dir()

    return abs_dict_, meta_, source_, recent_


# --- data collation
def collate_summary_data(
    to_get: dict[str, tuple[str, int]],
    abs_data: dict[str, DataFrame],
    md: DataFrame,
    *,
    verbose: bool = False,
) -> DataFrame:
    """Build a summary DataFrame of key ABS series for mgplot.summary_plot.

    Fetch each requested data item; where a growth period is given, replace the
    level with its percentage change over that period.

    Args:
        to_get: mapping of label -> (series_id, n_periods_growth); a zero period
            keeps the level, a non-zero period takes the percentage change.
        abs_data: the ABS data dictionary from readabs.
        md: the ABS metadata table from readabs.
        verbose: if True, print each selected series' table and description.

    Returns:
        A DataFrame with one column per label.

    """
    data = DataFrame()
    for label, [code, period] in to_get.items():
        selected = md[md[mc.id] == code].iloc[0]
        table_desc = selected[mc.tdesc]
        table = selected[mc.table]
        did = selected[mc.did]
        stype = selected[mc.stype]
        if verbose:
            print(code, table, table_desc, did, stype)
        series = abs_data[table][code]
        if period:
            series = series.pct_change(periods=period, fill_method=None) * 100
        data[label] = series
    return data


# === Useful constants for plotting
# CPI targets
ANNUAL_CPI_TARGET: dict[str, float | str | int] = {
    "y": 2.5,
    "color": "gray",
    "linestyle": "--",
    "linewidth": 0.75,
    "label": "2.5% annual inflation target",
    "zorder": -1,
}

ANNUAL_CPI_TARGET_RANGE: dict[str, float |str | int] = {
    "ymin": 2,
    "ymax": 3,
    "color": "#dddddd",
    "label": "2-3% annual inflation target range",
    "zorder": -1,
}

QUARTERLY_CPI_TARGET: dict[str, float | str | int] = {
    "y": (pow(1.025, 0.25) - 1) * 100,
    "linestyle": "dashed",
    "linewidth": 0.75,
    "color": "darkred",
    "label": "Quarterly growth consistent with 2.5% annual inflation",
}

QUARTERLY_CPI_RANGE: dict[str, float | str | int] = {
    "ymin": (pow(1.02, 0.25) - 1) * 100,
    "ymax": (pow(1.03, 0.25) - 1) * 100,
    "color": "#ffdddd",
    "label": "Quarterly growth consistent with 2-3% annual inflation target",
    "zorder": -1,
}

MONTHLY_CPI_TARGET: dict[str, float | str | int] = {
    "y": (pow(1.025, 1.0 / 12.0) - 1) * 100,
    "color": "darkred",
    "linewidth": 0.75,
    "linestyle": "--",
    "label": "Monthly growth consistent with a 2.5% annual inflation target",
    "zorder": -1,
}
