"""Unemployment rates across OECD members and partners: country groups, world context, recent ranges."""

# --- dependencies
import time

import pandas as pd
from mgplot import finalise_plot

from au_econ.charting.international import (
    MEAN_MEDIAN,
    country_group_charts,
    range_chart,
    recent_ranges,
    world_context_axes,
)
from au_econ.sources import oecd

# --- module contract
RELEASE = ("oecd-ur",)
TOPICS = ("international",)
TITLE = "Unemployment Rates"

# --- constants
DATAFLOW = "OECD.SDD.TPS,DSD_LFS@DF_IALFS_UNE_M,"  # unversioned: the latest version
KEYS = ("..._Z.Y._T.Y_GE15..M", "..._Z.Y._T.Y_GE15..Q")  # monthly first; then quarterly (NZ, Switzerland)
START = "2000-01"
WEB_DELAY = 2  # seconds between requests, to be gentle on the OECD server
SOURCE = "OECD: LFS"
GROUPS_FROM_YEAR = 2019
WORLD_FROM_YEAR = 2017
WORLD_TAG = f"since-{WORLD_FROM_YEAR}"
RANGE_MONTHS = (13, 25, 37, 49, 61)


# --- data
def fetch() -> pd.DataFrame:
    """Return monthly unemployment rates (per cent), one column per country label."""
    combined: pd.DataFrame | None = None
    for key in KEYS:
        table = oecd.get_table(DATAFLOW, key, START)
        table.index = pd.PeriodIndex(table.index, freq="M")
        if key.endswith("Q"):
            table = oecd.quarterly_to_monthly(table)
        combined = oecd.combine(combined, table)
        time.sleep(WEB_DELAY)
    if combined is None:
        raise ValueError("OECD unemployment: nothing fetched")
    rates = oecd.national_only(combined)
    oecd.report_missing(rates)
    print(rates.tail())
    return rates.rename(columns=oecd.LABELS)


def _from_year(data: pd.DataFrame, year: int) -> pd.DataFrame:
    """Rows from the start of year."""
    if not isinstance(data.index, pd.PeriodIndex):
        raise TypeError("expected a monthly PeriodIndex")
    return data[data.index.year >= year]


# --- charts
def unemployment_groups(data: pd.DataFrame) -> None:
    """Unemployment rates for each country group, since GROUPS_FROM_YEAR."""
    country_group_charts(
        _from_year(data, GROUPS_FROM_YEAR),
        title="Unemployment rates",
        ylabel="Per cent",
        legend={"loc": "best", "fontsize": "x-small"},
        rfooter=SOURCE,
        lfooter="OECD monitored nations. Seas adj.",
    )


def unemployment_world(data: pd.DataFrame) -> None:
    """Australia's unemployment rate against every country, with their mean and median."""
    ax = world_context_axes(_from_year(data, WORLD_FROM_YEAR))
    finalise_plot(
        ax,
        title="Australian unemployment rate in the world context",
        ylabel="Per cent",
        lfooter=(
            "OECD monitored nations. Mean and median calculated where "
            f"{int(MEAN_MEDIAN * 100)}% or more nations report."
        ),
        xlabel=None,
        y0=True,
        rfooter=SOURCE,
        tag=WORLD_TAG,
        legend={"loc": "best", "fontsize": "xx-small"},
    )


def unemployment_ranges(data: pd.DataFrame) -> None:
    """Each country's range of unemployment rates over recent windows."""
    for months in RANGE_MONTHS:
        range_chart(
            recent_ranges(data, months),
            months,
            title=f"OECD Unemployment Rates - previous {months} months",
            ylabel="Per cent",
            rfooter=SOURCE,
        )


# --- table of contents, in run order
CHARTS = (
    (unemployment_groups, ()),
    (unemployment_world, ()),
    (unemployment_ranges, ()),
)
