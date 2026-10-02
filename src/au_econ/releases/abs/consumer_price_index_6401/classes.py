"""CPI expenditure classes from 6401.0: a growth chart per class, and summaries across them.

The group / sub-group / class structure comes from the ABS SDMX codelist (sources.abs).
"""

# --- dependencies
from typing import TYPE_CHECKING, Unpack

import mgplot as mg
import pandas as pd
import readabs as ra
from readabs import metacol as mc

from au_econ.charting.targets import ANNUAL_CPI_TARGET_RANGE, MONTHLY_CPI_TARGET, QUARTERLY_CPI_TARGET
from au_econ.charting.windows import (
    MONTHS_PER_YEAR,
    QUARTERS_PER_YEAR,
    monthly_plot_times,
    quarterly_plot_times,
)
from au_econ.sources.abs import cpi_names

if TYPE_CHECKING:
    from au_econ.sources.abs import AbsRelease

# --- constants
CLASSES_SUBDIR = "Expenditure classes"
SUMMARY_SUBDIR = "Expenditure class summary"
QUARTERLY_TABLE = "6401018"
MONTHLY_TABLE = "640103"
INDEX_DID, AUSTRALIA_DID = "Index Numbers", "Australia"
ALL_GROUPS = "All groups CPI"
RENTS, NEW_DWELLINGS = "Rents", "New dwelling purchase by owner-occupiers"
POST_COVID_BASE = "2019-12"
PER_CENT = "Per cent"
CLASS_ORIGINAL_NOTE = "Original series. CPI expenditure classes only. "
INDEX_100_LINE = {"y": 100, "color": "black", "linestyle": "--", "linewidth": 0.75}

# class growth charts, by cadence: (table, label, target line, plot_from, periods in a year)
CADENCES = (
    (MONTHLY_TABLE, "Monthly", MONTHLY_CPI_TARGET, monthly_plot_times[1], MONTHS_PER_YEAR),
    (QUARTERLY_TABLE, "Quarterly", QUARTERLY_CPI_TARGET, quarterly_plot_times[1], QUARTERS_PER_YEAR),
)

# group-index charts
GROUP_SPAN_QUARTERS = 40  # ten years
PRE_COVID = pd.Period("2019Q4", freq="Q")
GROUP_MARKERS = (" ", "o", "s", "d", "p", "*", "<", ">", "v", "^", "P", "X")

# growth bars: (level, how many to show at each end, or None for all)
BAR_LEVELS = (("group", None), ("sub-group", None), ("class", 30))
BAR_MODES = ("annual", "quarterly", "post-covid")
BAR_SOURCE_NOTES = {"annual": "Monthly CPI series. ", "quarterly": "Quarterly CPI series. "}
BAR_SOURCE_NOTES["post-covid"] = BAR_SOURCE_NOTES["quarterly"]
BAR_WIDTH_INCHES, BAR_ROW_INCHES, BAR_MIN_HEIGHT, BAR_SPLIT_HEIGHT = 9.0, 0.3, 4.5, 8.0
INFREQUENT_LOOKBACK, INFREQUENT_ZEROS, NEAR_ZERO = 12, 5, 1e-9  # quarters; zero changes; tolerance

# breadth: share of classes above the top of the target band
BREADTH_ANNUAL_PCT = 3.0

# tails and distributions
QUANTILES = (0.1, 0.5, 0.9)
HISTORY_YEARS = 11
RECENT_QUARTERS = 20
DISTRIBUTION_BOUNDS, DISTRIBUTION_BOUNDS_ANNUAL = 15, 25
DISTRIBUTION_PADDING = 0.025
BOX_LABEL_ROTATION = 30

QUARTERLY_LFOOTER = "Australia. Quarterly CPI series. "


# --- helpers
def _class_name(did: str) -> str:
    """Return the expenditure item name from an 'Index Numbers ;  <name> ;  Australia ;' description."""
    return did.replace("Index Numbers ;", "").replace(";  Australia ;", "").strip().rstrip(";").strip()


def _index_series(release: AbsRelease, table: str) -> dict[str, pd.Series]:
    """Return every Australian index series in a table, keyed by expenditure item name."""
    rows = ra.search_abs_meta(release.meta, {table: mc.table, INDEX_DID: mc.did, AUSTRALIA_DID: mc.did})
    found: dict[str, pd.Series] = {}
    for did, series_id in zip(rows[mc.did], rows[mc.id], strict=True):
        name = _class_name(did)
        if name and series_id in release.data[table].columns:
            found[name] = release.data[table][series_id]
    return found


def _quarterly_items(release: AbsRelease) -> pd.DataFrame:
    """Return every quarterly expenditure item index (groups, sub-groups, classes)."""
    return pd.DataFrame(_index_series(release, QUARTERLY_TABLE)).sort_index()


def _level_index(release: AbsRelease, level: str, table: str) -> pd.DataFrame:
    """Return the index for every item at one level of the hierarchy, from one table."""
    series = _index_series(release, table)
    columns = {name: series[name] for name in cpi_names(level) if name in series}
    return pd.DataFrame(columns) if columns else pd.DataFrame()


def _level_growth(release: AbsRelease, level: str, mode: str) -> pd.DataFrame | None:
    """Return growth for every item at one level, or None if the data does not reach back far enough.

    mode: "annual" (12-month, monthly table), "quarterly" (quarter-on-quarter),
    "annual-quarterly" (4-quarter) or "post-covid" (cumulative since December 2019).
    """
    if mode == "annual":
        data, lag = _level_index(release, level, MONTHLY_TABLE), MONTHS_PER_YEAR
    else:
        data = _level_index(release, level, QUARTERLY_TABLE)
        lag = {"quarterly": 1, "annual-quarterly": QUARTERS_PER_YEAR}.get(mode, 0)
    if data.empty:
        return None
    if mode == "post-covid":
        start = pd.Period(POST_COVID_BASE, freq="Q")
        if start not in data.index:
            return None
        position = data.index.get_loc(start)
        if not isinstance(position, int):
            raise TypeError(f"Expected one {start} row, got {position!r}")
        return (data / data.iloc[position] - 1) * 100
    if not lag or data.index[-1] - lag not in data.index:
        return None
    return (data / data.shift(lag) - 1) * 100


def _infrequent(data: pd.DataFrame) -> set[str]:
    """Return classes whose index barely moves within the year (many zero quarterly changes)."""
    recent = data.iloc[-INFREQUENT_LOOKBACK:]
    zero_counts = (recent.abs() < NEAR_ZERO).sum(axis=0)
    return set(zero_counts[zero_counts >= INFREQUENT_ZEROS].index)


def _growth_bars_at(release: AbsRelease, level: str, top_keep: int | None) -> None:
    """Draw latest-growth bar charts for one hierarchy level, for each growth measure."""
    n_items = len(cpi_names(level))
    for mode in BAR_MODES:
        data = _level_growth(release, level, mode)
        if data is None or data.empty:
            continue
        infrequent = _infrequent(data) if mode == "quarterly" and level == "class" else set()
        latest = data.iloc[-1].dropna().sort_values(ascending=True)
        if latest.empty:
            continue
        period = data.index[-1]
        if infrequent:
            latest.index = [f"* {n}" if n in infrequent else n for n in latest.index]
        start = {
            "annual": period - MONTHS_PER_YEAR,
            "quarterly": period - 1,
            "post-covid": pd.Period(POST_COVID_BASE, freq="Q"),
        }[mode]
        start_note = f"From {start}. {BAR_SOURCE_NOTES[mode]}"
        infrequent_note = "* Item updated less frequently than quarterly. " if infrequent else ""

        if top_keep is None:
            mg.bar_plot_finalise(
                latest,
                horizontal=True,
                annotate=True,
                rounding=1,
                title=f"{mode.capitalize()} Growth by\nCPI {level.capitalize()}",
                xlabel=PER_CENT,
                rfooter=release.source,
                lfooter=f"Australia. {period}. {start_note}{infrequent_note}Original series.",
                figsize=(BAR_WIDTH_INCHES, max(BAR_MIN_HEIGHT, len(latest) * BAR_ROW_INCHES)),
            )
            continue

        mid_start, mid_end = top_keep, len(latest) - top_keep
        subsets = [("bottom", latest[:top_keep]), ("top", latest[-top_keep:])]
        if mid_end > mid_start:
            subsets.insert(1, ("middle", latest[mid_start:mid_end]))
        for tag, subset in subsets:
            tag_label = f"{tag} ({len(subset)})" if tag == "middle" else f"{tag} {top_keep}"
            mg.bar_plot_finalise(
                subset,
                horizontal=True,
                annotate=True,
                rounding=1,
                title=f"{mode.capitalize()} Growth by CPI\nExpenditure {level.capitalize()}: {tag_label}",
                tag=tag,
                xlabel=PER_CENT,
                rfooter=release.source,
                lfooter=f"Australia. {period}. {start_note}{infrequent_note}"
                f"Note: there are {n_items} expenditure {level}s.",
                figsize=(BAR_WIDTH_INCHES, BAR_SPLIT_HEIGHT),
            )


def _period_index(data: pd.DataFrame) -> pd.PeriodIndex:
    """Return the frame's index, which must be a PeriodIndex."""
    if not isinstance(data.index, pd.PeriodIndex):
        raise TypeError(f"Expected a PeriodIndex, got {type(data.index).__name__}")
    return data.index


def _box_plot(data: pd.DataFrame, bounds: int, **kwargs: Unpack[mg.FinaliseKwargs]) -> None:
    """Draw box plots of the distribution across classes, one box per period.

    mgplot has no box plot, so pandas draws onto the axes and mgplot finalises it.
    """
    centre = data.median(axis=1).max()
    min_plot, max_plot = centre - bounds, centre + bounds
    min_data, max_data = data.min().min(), data.max().max()
    if min_data < min_plot or max_data > max_plot:
        min_plot, max_plot = max(min_plot, min_data), min(max_plot, max_data)
        kwargs["lfooter"] = f"{kwargs.get('lfooter', '')}Larger outliers not plotted. "
    pad = (max_plot - min_plot) * DISTRIBUTION_PADDING
    kwargs |= {"tag": "box", "ylabel": PER_CENT, "ylim": (min_plot - pad, max_plot + pad)}
    axes = data.T.boxplot(rot=BOX_LABEL_ROTATION)
    mg.finalise_plot(axes, **kwargs)


# --- charts
def class_growth(release: AbsRelease) -> None:
    """Monthly and quarterly growth for every CPI expenditure class; recent."""
    names = sorted(cpi_names("class"))
    with mg.chart_subdir(CLASSES_SUBDIR):
        for table, label, target, plot_from, periods_per_year in CADENCES:
            series_by_name = _index_series(release, table)
            for name in names:
                if name not in series_by_name:
                    continue
                series = series_by_name[name].dropna()
                if len(series) < periods_per_year + 1:
                    continue
                mg.growth_plot_finalise(
                    mg.calc_growth(series),
                    plot_from=plot_from,
                    title=f"Growth: {label} {name} (Orig)",
                    ylabel=PER_CENT,
                    lfooter=f"Australia. {label} data. Original series. ",
                    axhline=target,
                    axhspan=ANNUAL_CPI_TARGET_RANGE,
                    rfooter=release.source,
                    y0=True,
                )


def expenditure_groups(release: AbsRelease) -> None:
    """CPI index by expenditure group, rebased: the last ten years, and since 2019Q4."""
    items = _quarterly_items(release)
    groups = set(cpi_names("group"))
    data = items[[c for c in items.columns if c in groups]].sort_index(axis=1)
    if data.empty:
        return
    with mg.chart_subdir(SUMMARY_SUBDIR):
        for span in (data.index[-1] - GROUP_SPAN_QUARTERS, PRE_COVID):
            valid = data.loc[span:].dropna(axis=1, how="all")
            start = max(span, valid.dropna(how="any").index[0])
            rebased = valid.div(valid.loc[start]).mul(100)
            mg.line_plot_finalise(
                rebased[rebased.index >= start],
                title=f"CPI Inflation by Groups (Index: {start}=100)",
                ylabel="Index",
                marker=list(GROUP_MARKERS),
                markersize=3,
                legend={"loc": "upper left", "fontsize": "xx-small", "ncols": 2},
                axhline=INDEX_100_LINE,
                rfooter=release.source,
                lfooter="Australia. Quarterly CPI series. Original series.",
            )


def growth_bars(release: AbsRelease) -> None:
    """Latest annual, quarterly and post-COVID growth by group, sub-group and class."""
    with mg.chart_subdir(SUMMARY_SUBDIR):
        for level, top_keep in BAR_LEVELS:
            _growth_bars_at(release, level, top_keep)


def breadth(release: AbsRelease) -> None:
    """Share of expenditure classes growing faster than the top of the target band."""
    index = _level_index(release, "class", QUARTERLY_TABLE)
    quarterly_pct = (((BREADTH_ANNUAL_PCT / 100) + 1) ** (1 / QUARTERS_PER_YEAR) - 1) * 100
    measures = (
        ("annual", (index / index.shift(QUARTERS_PER_YEAR) - 1) * 100, BREADTH_ANNUAL_PCT, 1),
        ("quarterly", (index / index.shift(1) - 1) * 100, quarterly_pct, 3),
    )
    with mg.chart_subdir(SUMMARY_SUBDIR):
        for modality, data, threshold, decimals in measures:
            if data.empty:
                continue
            share = (data > threshold).sum(axis=1, skipna=True) / data.notna().sum(axis=1) * 100
            mg.line_plot_finalise(
                share,
                annotate=True,
                rounding=1,
                title=f"CPI exp. classes with {modality} price growth > {round(threshold, decimals)}%",
                ylabel=PER_CENT,
                rfooter=release.source,
                lfooter=f"{QUARTERLY_LFOOTER}CPI expenditure classes only. "
                f"Endpoint: {share.iloc[-1]:0.2f}% of CPI expenditure classes. ",
            )


def tails(release: AbsRelease) -> None:
    """10th, 50th and 90th percentiles of class growth, annual and quarterly."""
    recent_start = _quarterly_items(release).index[-1].year - HISTORY_YEARS
    with mg.chart_subdir(SUMMARY_SUBDIR):
        for mode, label in (("annual-quarterly", "Annual"), ("quarterly", "Quarterly")):
            data = _level_growth(release, "class", mode)
            if data is None or data.empty:
                continue
            base = data if mode == "annual-quarterly" else data[_period_index(data).year >= recent_start]
            mg.line_plot_finalise(
                base.T.quantile(q=list(QUANTILES), numeric_only=True).T,
                title=f"{label} Growth by CPI Expenditure Classes",
                tag="quantile",
                ylabel=PER_CENT,
                y0=True,
                rfooter=release.source,
                lfooter=f"{QUARTERLY_LFOOTER}{CLASS_ORIGINAL_NOTE}",
                legend={"title": "Quantiles", "loc": "best", "fontsize": "x-small"},
                annotate=True,
            )


def distributions(release: AbsRelease) -> None:
    """Box plots of class growth: recent quarters, and the latest quarter in past years."""
    lfooter = f"{QUARTERLY_LFOOTER}{CLASS_ORIGINAL_NOTE}"
    with mg.chart_subdir(SUMMARY_SUBDIR):
        quarterly = _level_growth(release, "class", "quarterly")
        if quarterly is not None and not quarterly.empty:
            _box_plot(
                quarterly.dropna(how="all").iloc[-RECENT_QUARTERS:],
                DISTRIBUTION_BOUNDS,
                title="Distribution of Quarterly Growth by CPI Exp. Class",
                rfooter=release.source,
                lfooter=lfooter,
                pre_tag="recent",
                axhline=QUARTERLY_CPI_TARGET,
                legend={"loc": "best", "fontsize": "x-small"},
            )
        for mode, label, bounds in (
            ("annual-quarterly", "Annual", DISTRIBUTION_BOUNDS_ANNUAL),
            ("quarterly", "Quarterly", DISTRIBUTION_BOUNDS),
        ):
            data = _level_growth(release, "class", mode)
            if data is None or data.empty:
                continue
            index = _period_index(data)
            latest = index[-1]
            same_quarter = data[(index.year >= latest.year - HISTORY_YEARS) & (index.quarter == latest.quarter)]
            _box_plot(
                same_quarter,
                bounds,
                title=f"Distribution of {label} Growth by CPI Exp. Class",
                rfooter=release.source,
                lfooter=lfooter,
                axhline=QUARTERLY_CPI_TARGET,
                pre_tag="by-year",
                y0=True,
            )


def residential(release: AbsRelease) -> None:
    """Annual growth in rents and new dwelling prices, and their gap to the All Groups CPI."""
    items = _quarterly_items(release)
    residential_index = items[[RENTS, NEW_DWELLINGS]].dropna(how="all")
    growth = (residential_index / residential_index.shift(QUARTERS_PER_YEAR) - 1) * 100
    all_groups = items[ALL_GROUPS].dropna()
    gap = growth.sub((all_groups / all_groups.shift(QUARTERS_PER_YEAR) - 1) * 100, axis=0)
    charts = (
        (growth, "CPI: Rents and New Dwelling Purchase by Owner-Occupiers", "Per cent per year"),
        (gap, "Residential Costs Above/Below All Groups CPI Inflation", "Annual percentage points"),
    )
    with mg.chart_subdir(SUMMARY_SUBDIR):
        for data, title, ylabel in charts:
            mg.line_plot_finalise(
                data,
                title=title,
                ylabel=ylabel,
                rfooter=release.source,
                lfooter="Australia. Quarterly CPI series. Original series. ",
                y0=True,
                annotate=True,
            )


# --- table of contents, in run order
CHARTS = (
    (expenditure_groups, ()),
    (growth_bars, ()),
    (breadth, ()),
    (tails, ()),
    (distributions, ()),
    (residential, ()),
    (class_growth, ()),
)
