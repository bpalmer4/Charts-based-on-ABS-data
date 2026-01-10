"""
abs_helper.py
A collection of functions to make working with ABS data 
just a litte bit easier.
"""

# === imports
from typing import Any
from pandas import DataFrame, Series
import readabs as ra
from readabs import metacol as mc

from mgplot import set_chart_dir, clear_chart_dir


# === frequently used data sources


def get_gdp(gdp_type="CP", seasonal="SA") -> tuple[Series, str]:
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


def get_population(
        state="Australia",
        project=True,
        **kwargs,
    ) -> tuple[Series, str]:
    """Return the ABS population Series for a given state.

    Arguments:
    - state: str - The state to return the population for.
      Defaults to 'Australia' for the national population.
    - project: bool - Whether to project the population to the present.

    Returns a tuple comprising:
    - the selected population series and 
    - the units of that series.
    """

    # get the series
    cat = "3101.0"
    table = "310104"
    pop_data, pop_meta = ra.read_abs_cat(cat, single_excel_only=table, verbose=False)
    selector = {
        f";  {state} ;": mc.did,  # too many states have "Australia" in their name
        "Estimated Resident Population ;  Persons ;  ": mc.did
    }
    _table, series_id, units = ra.find_abs_id(pop_meta, selector, **kwargs)
    pop = pop_data[table][series_id]

    if project:
        # a bit rough - but should do for such a simple series
        # over such a short period (6 months)
        rate = pop.iloc[-1] / pop.iloc[-2]
        base_period = pop.index[-1]
        for i in range(1, 3):
            pop[base_period + i] = pop[base_period + i - 1] * rate

    return pop, units


# === data retrieval and initialisation
def get_abs_data(
    cat: str,
    chart_dir_suffix: str = "",
    **kwargs: Any,
) -> tuple[dict[str, DataFrame], DataFrame, str, str]:
    """Get ABS data for a specific catalogue number and create
    the associated plot directories. This is my standard set-up
    for notebooks that use ABS data.

    Arguments:
        cat: an ABS catalogue number (as a string, eg. "6401.0")
        chart_dir_suffix: optional suffix to append to chart directory name
                          (useful for splitting notebooks, e.g. " - Productivity")
        **kwargs: any additional arguments to pass to read_abs_cat

    Returns: the data in a dictionary, metadata, source and a recent date
    to plot from."""

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
    verbose: bool = False,
) -> DataFrame:
    """
    Construct a summary DataFrame of key ABS data.
    Get required data items. If period is specified,
    calculate the percentage change over that period.
    Return a DataFrame with the data. This DataFrame 
    is then passed to the mgplot.summary_plot() function.

    Args
    - to_get: in the form {label: (series_id: str, n_periods_growth: int), ...}
    - abs_data: duct of ABS data from readabs
    - md: ABS meta data table from readabs
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
