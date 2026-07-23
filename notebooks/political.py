"""Generic helpers for charting data by government epoch.

Nothing here knows about a particular ABS series or a particular chart: each
function takes a series and the epoch table returned by the notebook's
get_governments(), and splits, rebases or summarises the series epoch by epoch.
The ABS fetchers and the plotting functions live in the notebook.
"""

# system imports
import textwrap

# analytic imports
import mgplot as mg
import pandas as pd
from pandas import DataFrame, Series

# periods per year, by data frequency: prices are quarterly, labour monthly
QUARTERS_PER_YEAR = 4
MONTHS_PER_YEAR = 12

# a growth rate needs two points: one observation gives no change to measure
MIN_GROWTH_OBSERVATIONS = 2

def segment_by_government(series: Series, govts: DataFrame) -> DataFrame:
    """Split a series into one column per government epoch.

    Columns are NaN outside their own epoch, so they share one calendar axis
    and each can take its party's colour. The election period falls in both the
    outgoing and the incoming epoch, so consecutive segments join rather than
    leave a gap in what should read as one line.

    Args:
        series: any series on a PeriodIndex.
        govts: government epochs, as returned by get_governments().

    Returns:
        One column per epoch, on the index of `series`.

    """
    index = series.index
    if not isinstance(index, pd.PeriodIndex):
        raise TypeError(f"series must have a PeriodIndex, got {type(index).__name__}")

    last = index[-1]
    segments: dict[str, Series] = {}
    for name, row in govts.iterrows():
        start = pd.Period(row["start"], freq=index.freqstr)
        end = min(pd.Period(row["end"], freq=index.freqstr), last)
        segments[str(name)] = series.where((index >= start) & (index <= end))
    return DataFrame(segments)


def mean_by_government(series: Series, govts: DataFrame) -> Series:
    """Average level of a series over each government epoch.

    The right statistic for a rate, where compounding has no meaning. The
    election period counts once on each side - immaterial against epochs of 12
    to 92 periods, and it keeps the epochs contiguous.

    Args:
        series: any series on a PeriodIndex.
        govts: government epochs, as returned by get_governments().

    Returns:
        The mean, indexed by epoch name, in the units of `series`.

    """
    return segment_by_government(series, govts).mean()


def change_by_government(series: Series, govts: DataFrame) -> Series:
    """Change in a series from the first to the last observation of each epoch.

    Measured between observations that exist, not between election dates, so an
    epoch whose data starts late is measured over what it has. Both endpoints
    are single periods, so a sharp move late in a long term dominates the
    result - read alongside the average, never alone.

    Args:
        series: any series on a PeriodIndex.
        govts: government epochs, as returned by get_governments().

    Returns:
        The change, indexed by epoch name, in the units of `series`.

    """
    segments = segment_by_government(series, govts)
    changes: dict[str, float] = {}
    for name in segments.columns:
        observed = segments[name].dropna()
        if observed.empty:
            raise ValueError(f"No data within the {name} epoch to measure change over")
        changes[str(name)] = float(observed.iloc[-1] - observed.iloc[0])
    return Series(changes)


def index_by_government(series: Series, govts: DataFrame, base: float = 100.0) -> DataFrame:
    """Rebase a level series to `base` at the start of each government epoch.

    Shows how far a level moved under each government rather than where it
    stood. The series must be seasonally adjusted: rebasing an original series
    to a single period would bake that period's seasonal factor into the base,
    and the elections fall in four different quarters.

    Args:
        series: a level series on a PeriodIndex.
        govts: government epochs, as returned by get_governments().
        base: the index value at each epoch's first observation.

    Returns:
        One column per epoch, indexed to `base` at its start.

    """
    segments = segment_by_government(series, govts)
    indexed: dict[str, Series] = {}
    for name in segments.columns:
        observed = segments[name].dropna()
        if observed.empty:
            raise ValueError(f"No data within the {name} epoch to index")
        indexed[str(name)] = segments[name] / observed.iloc[0] * base
    return DataFrame(indexed)

def cagr_by_government(
    series: Series,
    govts: DataFrame,
    periods_per_year: int = QUARTERS_PER_YEAR,
) -> Series:
    """Compound annual growth of a level series over each government epoch.

    Measured between the first and last observations inside the epoch, not
    between election dates, so an epoch whose data starts late is measured over
    what it has. Annualised rather than cumulative because the epochs run from
    3 years to 23, so a cumulative measure would mostly report longevity.

    Args:
        series: a level series (e.g. a price index) on a PeriodIndex.
        govts: government epochs, as returned by get_governments().
        periods_per_year: periods per year in `series`.

    Returns:
        The growth rate in per cent per year, indexed by epoch name.

    """
    segments = segment_by_government(series, govts)
    growth: dict[str, float] = {}
    for name in segments.columns:
        observed = segments[name].dropna()
        if len(observed) < MIN_GROWTH_OBSERVATIONS:
            raise ValueError(f"Fewer than two observations within the {name} epoch")
        elapsed = (observed.index[-1] - observed.index[0]).n
        growth[str(name)] = (
            (observed.iloc[-1] / observed.iloc[0]) ** (periods_per_year / elapsed) - 1
        ) * 100
    return Series(growth)

def cagr_path_by_government(
    series: Series,
    govts: DataFrame,
    periods_per_year: int = QUARTERS_PER_YEAR,
    min_periods: int = 2,
) -> DataFrame:
    """Return the running compound annual growth rate within each epoch.

    At each period, the annualised growth from that epoch's election to that
    period, so each column converges on the epoch's final rate - the number in
    the bar chart - while showing how it got there. The first `min_periods`
    periods are dropped: annualising a single quarter multiplies it by four,
    which says more about the arithmetic than about the government.

    Args:
        series: a level series (e.g. a price index) on a PeriodIndex.
        govts: government epochs, as returned by get_governments().
        periods_per_year: periods per year in `series`.
        min_periods: periods to skip at the start of each epoch.

    Returns:
        One column per epoch, in per cent per year.

    """
    index = series.index
    if not isinstance(index, pd.PeriodIndex):
        raise TypeError(f"series must have a PeriodIndex, got {type(index).__name__}")

    last = index[-1]
    paths: dict[str, Series] = {}
    for name, row in govts.iterrows():
        start = pd.Period(row["start"], freq=index.freqstr)
        end = min(pd.Period(row["end"], freq=index.freqstr), last)
        window = series[(index >= start) & (index <= end)]
        elapsed = Series(range(len(window)), index=window.index, dtype=float)
        wanted = elapsed >= min_periods
        path = Series(float("nan"), index=window.index)
        path[wanted] = (
            (window[wanted] / window.iloc[0]) ** (periods_per_year / elapsed[wanted]) - 1
        ) * 100
        paths[str(name)] = path
    return DataFrame(paths)

def epoch_vlines(
    govts: DataFrame,
    freq: str,
    loc: str = "auto right",
) -> list[dict[str, object]]:
    """Thin vertical lines at each election, carrying the epoch name.

    finalise_plot() treats "text"/"loc" in an axvline dict as a label for the
    line, anchoring x in data and y in axes coordinates and taking the line's
    colour - so colouring the line by party colours the label to match.

    `loc` takes an edge ("auto", "top", "bottom") and a side ("left", "right").
    "auto" sends each label to whichever end has more clearance near its own
    election, judged from a narrow band of x, so a long label can still run
    into data further along - hence the option to fix the edge for a chart.

    Args:
        govts: government epochs, as returned by get_governments().
        freq: the frequency of the chart's PeriodIndex, for the x ordinal.
        loc: label placement, passed through to finalise_plot.

    Returns:
        One axvline dict per epoch, for finalise_plot().

    """
    return [
        {
            "x": pd.Period(row["start"], freq=freq).ordinal,
            "text": str(name),
            "loc": loc,
            "color": mg.get_color(row["party"]),
            "linewidth": 0.5,
        }
        for name, row in govts.iterrows()
    ]

def year_ended_growth(series: Series, periods_per_year: int = QUARTERS_PER_YEAR) -> Series:
    """Year-ended growth of a level series, in per cent.

    Args:
        series: a level series (e.g. a price index) on a PeriodIndex.
        periods_per_year: periods per year in `series`.

    Returns:
        The growth rate in per cent per year.

    """
    return ((series / series.shift(periods_per_year) - 1) * 100).dropna()
