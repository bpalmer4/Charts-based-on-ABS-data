"""Charts comparing countries: Australia in world context, country groups, and recent ranges (OHLC)."""

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import pandas as pd
from mgplot import finalise_plot, line_plot, line_plot_finalise

from au_econ.sources.oecd import COUNTRIES

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes

AUSTRALIA = "Australia"
MEAN_MEDIAN = 0.80  # share of countries reporting before the mean and median are drawn

# Country groups, at most six lines a chart; the key becomes the chart's file-name tag
COUNTRY_GROUPS = {
    "of_interest": ("Australia", "United States", "Canada", "Germany", "United Kingdom", "Japan"),
    "anglosphere": ("Australia", "United States", "Canada", "New Zealand", "United Kingdom", "Ireland"),
    "major_europe": ("France", "Germany", "Italy", "United Kingdom", "Russia", "Spain"),
    "largest_economies": ("United States", "China", "Japan", "Germany", "United Kingdom", "India"),
    "asia": ("South Korea", "Japan", "China", "India", "Indonesia"),
    "north_europe": ("Denmark", "Sweden", "Norway", "Iceland", "Finland", "United Kingdom"),
    "baltic_europe": ("Latvia", "Lithuania", "Estonia"),
    "central_europe": ("Czech Republic", "Hungary", "Slovakia", "Slovenia", "Poland", "Greece"),
    "west_europe": ("Belgium", "Spain", "Portugal", "Netherlands", "Luxembourg", "France"),
    "italo_germanic_europe": ("Germany", "Austria", "Switzerland", "Italy"),
    "n_america": ("United States", "Canada", "Mexico"),
    "c_s_america": ("Chile", "Brazil", "Colombia", "Costa Rica"),
    "high_inflation": ("Turkey", "Argentina"),
    "other": ("Australia", "New Zealand", "Saudi Arabia", "South Africa", "Israel"),
}

RANGE_BAR_ALPHA = 0.15
RANGE_MARKER_SIZE = 5
RANGE_MARGIN = 0.025  # share of the plotted range added above and below
GOOD, BAD = "darkblue", "darkorange"  # a fall is good for these measures; colour-blind safe


def world_context_axes(data: pd.DataFrame, *, annotate: bool = False, label: str = "OECD monitored") -> Axes:
    """Draw every country thin, the mean and median where enough report, and Australia on top.

    The mean and median include Australia; label names them in the legend (e.g. "BIS monitored").
    Returns the axes for the caller to finalise.
    """
    hidden = data.rename(columns={column: f"_{column}" for column in data.columns})  # "_" keeps out of legend
    ax = line_plot(hidden, width=0.3, color="blue")
    enough = data.notna().sum(axis=1) >= len(data.columns) * MEAN_MEDIAN
    mean = data.mean(axis=1).where(enough).rename(f"{label} mean")
    median = data.median(axis=1).where(enough).rename(f"{label} median")
    line_plot(mean, ax=ax, color="darkblue", style="--", width=2, label_series=True, annotate=annotate)
    line_plot(median, ax=ax, color="darkred", style=":", width=2, label_series=True, annotate=annotate)
    line_plot(data[AUSTRALIA].dropna(), ax=ax, color="darkorange", width=3, label_series=True, annotate=annotate)
    return ax


def country_group_charts(
    data: pd.DataFrame,
    *,
    title: str,
    ylabel: str,
    rfooter: str,
    lfooter: str,
    legend: dict[str, Any] | None = None,
    axhspan: dict[str, Any] | None = None,
) -> None:
    """One line chart per COUNTRY_GROUPS group, of the group's countries present in data.

    Without a legend argument, a chart of more than one line gets mgplot's default legend.
    """
    for tag, group in COUNTRY_GROUPS.items():
        present = sorted(set(group) & set(data.columns), key=COUNTRIES.__getitem__)  # by OECD code
        line_plot_finalise(
            data[present],
            tag=tag,
            xlabel=None,
            dropna=True,
            y0=True,
            width=2,
            title=title,
            ylabel=ylabel,
            rfooter=rfooter,
            lfooter=lfooter,
            legend=legend if legend is not None else len(present) > 1,
            axhspan=axhspan,  # None draws nothing
        )


def recent_ranges(data: pd.DataFrame, months: int, exclude: Sequence[str] = ()) -> pd.DataFrame:
    """Open, high, low and close over each country's last `months` months of data, sorted by close.

    Each row label carries the country's latest month (e.g. "Japan 26-08"). Countries
    without `months` months of history are left out.
    """
    fields = ["Open", "High", "Low", "Close"]
    columns: dict[str, pd.Series] = {}
    for name in data.columns:
        if name in exclude:
            continue
        column = data[name]
        last = column.last_valid_index()
        if not isinstance(last, pd.Period):
            raise TypeError(f"{name}: expected a monthly PeriodIndex")
        window = pd.period_range(end=last, periods=months)
        if window.min() < column.index.min():
            continue
        values = column[window]
        columns[f"{name} {str(last.year)[2:]}-{last.month:02d}"] = pd.Series(
            [values.iloc[0], values.max(), values.min(), values.iloc[-1]], index=fields
        )
    return pd.DataFrame(columns, index=fields).T.sort_values("Close")


def range_chart(
    ranges: pd.DataFrame,
    months: int,
    *,
    title: str,
    ylabel: str,
    rfooter: str,
    axhspan: dict[str, Any] | None = None,
) -> None:
    """Vertical bars spanning each country's low to high, with markers at the first and last prints."""
    _fig, ax = plt.subplots()
    colours = [GOOD if first > last else BAD for first, last in zip(ranges["Open"], ranges["Close"], strict=True)]
    ax.bar(
        ranges.index,
        ranges["High"] - ranges["Low"],
        bottom=ranges["Low"].to_numpy(),
        color=colours,
        linewidth=1.0,
        edgecolor="black",
        label=f"Range of prints through the {months} months",
        alpha=RANGE_BAR_ALPHA,
    )
    ax.plot(
        ranges.index,
        ranges["Open"],
        marker="<",
        linestyle="None",
        label=f"First print in the {months} months",
        color=GOOD,
        markersize=RANGE_MARKER_SIZE,
    )
    ax.plot(
        ranges.index,
        ranges["Close"],
        marker=">",
        linestyle="None",
        label=f"Last print in the {months} months",
        color=BAD,
        markersize=RANGE_MARKER_SIZE,
    )
    ax.tick_params(axis="both", which="major", labelsize="xx-small")
    lowest, highest = min(0, ranges["Low"].min()), ranges["High"].max()  # include zero
    margin = (highest - lowest) * RANGE_MARGIN
    ax.set_ylim(lowest - margin, highest + margin)
    ax.tick_params(axis="x", labelrotation=90)  # the labels are already the country names
    finalise_plot(
        ax,
        lfooter=(
            "Year and month of latest print in the axis labels. "
            f"Range is the {months} months up to and including the latest data. "
        ),
        legend={"loc": "best", "fontsize": "6"},
        y0=True,
        x0=False,
        title=title,
        ylabel=ylabel,
        rfooter=rfooter,
        axhspan=axhspan,  # None draws nothing
    )
