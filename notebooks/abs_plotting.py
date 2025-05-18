"""Plot ABS seasonally adjusted and trend data for a given series."""

# === Imports
from typing import Any, Callable, Final
from pandas import DataFrame, Series
from readabs import recalibrate, search_abs_meta
from readabs import metacol as mc
from mgplot import (
    seas_trend_plot,
    colorise_list,
    abbreviate_state,
    get_setting,
    line_plot_finalise,
    state_names,
)



# === Constants
SEAS_ADJ: Final[str] = "Seasonally Adjusted"
TREND: Final[str] = "Trend"
ORIG: Final[str] = "Original"


# === Functions
def iudts_from_row(row: Series) -> tuple[str, str, str, str, str]:
    """Return a tuple comrising series_id, units, data_description,
    table_number, series_type."""
    return (
        row[mc.id],
        row[mc.unit],
        row[mc.did],
        row[mc.table],
        row[mc.stype],
    )


def search_args_from_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return a dictionary of arguments that are not used by search_abs_meta()."""

    args = {}
    for arg in "exact_natch", "regex", "validate_unique", "verbose":
        if arg in kwargs:
            args[arg] = kwargs.pop(arg)
    return args


def plot_rows_seas_trend(
    abs_dict: dict[str, DataFrame],
    meta: DataFrame,
    selector: dict[str, str],
    **kwargs: Any,
) -> None:
    """Produce an seasonal/Trend chart for the rows selected from
    the metadata with selector.
    Agruments:
    - abs_dict - dict[str, DataFrame] - dictionary of ABS dataframes
    - selector - dict - used with search_abs_meta() to select rows from meta
      this needs to select both a Trend and Seasonally Adjusted row, and
      must exclude the "Series Type" column
    - **kwargs - arguments passed to plotting function."""

    # sanity checks - make sure seas/trend not in the selector
    if mc.stype in selector.values():
        print(f'Check: unexpected column "{mc.stype}" in the selector')
        return

    # identify the plot-sets using the selector ...
    args = search_args_from_kwargs(kwargs)
    st_data = {}
    for series_type in SEAS_ADJ, TREND:
        st_data[series_type] = search_abs_meta(
            meta,
            {**selector, series_type: mc.stype},
            **args,
        )

    # check plot-sets look reasonable
    if len(st_data[SEAS_ADJ]) != len(st_data[TREND]):
        print("The number of Trend and Seasonally Adjusted rows do not match")
        return
    if not st_data[SEAS_ADJ][mc.did].is_unique or not st_data[TREND][mc.did].is_unique:
        print("Data item descriptions are not unique")
        return

    # plot Seaspnal + Trend charts one-by-one
    for did in st_data[TREND][mc.did]:
        # get data series
        frame_data = {}
        for row_type in SEAS_ADJ, TREND:
            row = st_data[row_type][st_data[row_type][mc.did] == did].iloc[0]
            r_id, r_units, _, r_table, _ = iudts_from_row(row)
            frame_data[row_type] = abs_dict[r_table][r_id]

        # put the data into a frame and plot
        # Note - assume SA and Trend units are the same, this is not checked.
        frame, r_units = recalibrate(DataFrame(frame_data), r_units)
        seas_trend_plot(
            frame,  # cast(DataFrame, frame),
            title=did.replace(" ;  ", ": ").replace(" ;", ""),  # NEEDS THINKING
            ylabel=r_units,
            **kwargs,
        )


def plot_rows_individually(
    abs_dict: dict[str, DataFrame],
    meta: DataFrame,
    selector: dict[str, str],
    plot_function: Callable,
    **kwargs: Any,  # passed to plotting function
) -> None:
    """Produce an single chart for each row selected from
    the meta data with selector.
    Agruments:
    - abs_dict - dict[str, DataFrame] - dictionary of ABS dataframes
    - selector - dict - used with find_rows() to select rows from meta
    - plot_function - callable - for plotting a series of dara
    - regex - bool - used with selector in find_rows()
    - verbose - bool - used for feedback from find_rows()
    - **kwargs - arguments passed to plotting function."""

    args = search_args_from_kwargs(kwargs)
    rows = search_abs_meta(meta, selector, **args)

    for _, row in rows.iterrows():
        series_id, units, did, table, series_type = iudts_from_row(row)
        series, units = recalibrate(abs_dict[table][series_id], units)
        series.name = f"{series_type.capitalize()} series"

        plot_function(
            series,
            title=did.replace(" ;  ", ": ").replace(" ;", ""),
            ylabel=units,
            **kwargs,
        )


def _column_name_fix(r_frame: DataFrame) -> tuple[DataFrame, str, list[str]]:
    """Shorten column names."""
    columns = r_frame.columns.to_list()
    title = longest_common_prefex(columns)
    renamer = {x: abbreviate_state(x) for x in r_frame.columns}
    colours = colorise_list(renamer.values())
    r_frame = r_frame.rename(columns=renamer)
    return r_frame, title, colours


def longest_common_prefex(strings: list[str]) -> str:
    """Find the longest common string prefix."""
    num_strings: int = len(strings)

    # trivial cases
    if num_strings == 0:
        return ""
    if num_strings == 1:
        return strings[0]

    # harder cases
    broken = False
    for i in range(0, len(strings[0]) + 1):
        if i == len(strings[0]):
            break
        for j in range(1, num_strings):
            if i >= len(strings[j]) or strings[0][i] != strings[j][i]:
                broken = True
                break
        if broken:
            break

    return strings[0][:i]


def plot_rows_collectively(
    abs_dict: dict[str, DataFrame],
    meta: DataFrame,
    selector: dict[str, str],
    **kwargs: Any,  # passed to plotting function
) -> None:
    """Produce an collective/single chart covering each row
    selected from the meta data with selector.
    Agruments:
    - abs_dict - dict[str, DataFrame] - dictionary of ABS dataframes
    - selector - dict - used with find_rows() to select rows from meta
    - regex - bool - used with selector in find_rows()
    - verbose - bool - used for feedback from find_rows()
    - **kwargs - arguments passed to plotting function."""

    args = search_args_from_kwargs(kwargs)
    rows = search_abs_meta(meta, selector, **args)

    frame = DataFrame()

    for _, row in rows.iterrows():
        series_id, units, did, table, _ = iudts_from_row(row)
        frame[did.replace(" ;  ", ": ").replace(" ;", "")] = abs_dict[table][series_id]
    if len(frame) == 0:
        return

    frame, units = recalibrate(frame, units)
    frame, title, colours = _column_name_fix(frame)

    legend = {
        **get_setting("legend"),
        "ncols": 2,
        **(kwargs.pop("legend", {}))
    }
    line_plot_finalise(
        frame,
        title=title,
        ylabel=units,
        legend=legend,
        color=colours,
        **kwargs,
    )
    # end plot_rows_collectively()


# public
def fix_abs_title(title: str, lfooter: str) -> tuple[str, str]:
    """Simplify complex ABS series names."""

    check = [
        "Chain volume measures",  # National Accounts,
        "Chain Volume Measures",  # Business indicators,
        "Chain Volume Measure",  # Retail Trade
        "Current prices",  # National Accounts
        "Current Prices",
        "Current Price",
        "Total (State)",
        "Total (Industry)",
        # Business Indicators
        "CORP",
        "TOTAL (SCP_SCOPE)",
    ]
    for c in check:
        if c in title:
            title = title.replace(f"{c} ;", "")
            lfooter += f"{c}. "

    for s, abbr in state_names.items():
        title = title.replace(s, abbr)

    title = (
        title.replace(";", "")
        .replace(" - ", " ")
        .replace("    ", " ")
        .replace("   ", " ")
        .replace("  ", " ")
        .strip()
    )
    return title, lfooter
