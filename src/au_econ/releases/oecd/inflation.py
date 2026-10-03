"""Annual consumer price inflation across OECD members and partners: recent ranges, world context, groups."""

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
RELEASE = ("oecd-cpi",)
TOPICS = ("international",)
TITLE = "Consumer Price Inflation"

# --- constants
DATAFLOWS = (  # COICOP 2018 first: its series win where both have a country; unversioned: the latest
    "OECD.SDD.TPS,DSD_PRICES_COICOP2018@DF_PRICES_C2018_ALL,",
    "OECD.SDD.TPS,DSD_PRICES@DF_PRICES_ALL,",
)
KEYS = (".M.N.CPI.PA._T.N.GY", ".Q.N.CPI.PA._T.N.GY")  # monthly first; year-on-year growth
MONTHLY_DATAFLOW = DATAFLOWS[1]  # has Australia's monthly CPI; COICOP 2018 still treats it as quarterly
AUSTRALIA = "Australia"
NEW_ZEALAND = "New Zealand"  # still reports quarterly
AUS_MONTHLY_KEY = f"{oecd.COUNTRIES[AUSTRALIA]}.M.N.CPI.PA._T.N.GY"
AUS_MONTHLY_START = pd.Period("2025-04", freq="M")  # Australia moved from a quarterly to a monthly CPI
START = "2019-05"
WEB_DELAY = 2  # seconds between requests, to be gentle on the OECD server
SOURCE = "OECD: Prices"
EXCLUDE = ("Turkey", "Russia", "Argentina")  # rampant inflation (Turkey, Argentina); Russia not updating
TARGET = {"ymin": 2, "ymax": 3, "color": "#dddddd", "label": "2-3% inflation target", "zorder": -1}
RANGE_MONTHS = (13, 25, 37, 49)


# --- data
def _align_quarterly(frame: pd.DataFrame) -> pd.DataFrame:
    """Move quarterly reporters from the mid-quarter month, where the OECD places them, to the quarter end.

    Only Australia's quarterly years (before AUS_MONTHLY_START) move; New Zealand moves throughout.
    """
    aus, nzl = oecd.COUNTRIES[AUSTRALIA], oecd.COUNTRIES[NEW_ZEALAND]
    quarterly = frame.index < AUS_MONTHLY_START
    frame.loc[quarterly, aus] = frame[aus].loc[quarterly].shift(1)
    frame[nzl] = frame[nzl].shift(1)
    return frame


def fetch() -> pd.DataFrame:
    """Return annual CPI inflation (per cent), monthly, one column per country label."""
    combined: pd.DataFrame | None = None
    for dataflow in DATAFLOWS:
        for key in KEYS:
            table = oecd.get_table(dataflow, key, START)
            table.index = pd.PeriodIndex(table.index, freq="M")
            if key[1] == "Q":
                table = oecd.quarterly_to_monthly(table)
            combined = oecd.combine(combined, table)
            time.sleep(WEB_DELAY)
    if combined is None:
        raise ValueError("OECD inflation: nothing fetched")
    combined = oecd.national_only(combined)

    aus = oecd.COUNTRIES[AUSTRALIA]
    monthly = oecd.get_table(MONTHLY_DATAFLOW, AUS_MONTHLY_KEY, START)
    monthly.index = pd.PeriodIndex(monthly.index, freq="M")
    combined.loc[monthly.index, aus] = monthly[aus]
    print(f"Patched AUS with monthly data: {monthly.index.min()} to {monthly.index.max()}")

    inflation = _align_quarterly(combined)
    print(inflation[[aus, oecd.COUNTRIES[NEW_ZEALAND]]].tail(12))
    oecd.report_missing(inflation)
    return inflation.rename(columns=oecd.LABELS)


# --- charts
def inflation_ranges(data: pd.DataFrame) -> None:
    """Each country's range of annual inflation prints over recent windows."""
    for months in RANGE_MONTHS:
        range_chart(
            recent_ranges(data, months, exclude=EXCLUDE),
            months,
            title=f"OECD Annual Inflation Rates - previous {months} months",
            ylabel="Per cent",
            axhspan=TARGET,
            rfooter=SOURCE,
        )


def inflation_world(data: pd.DataFrame) -> None:
    """Australia's annual inflation against every other country, with their mean and median."""
    ax = world_context_axes(data.drop(columns=[c for c in EXCLUDE if c in data.columns]))
    finalise_plot(
        ax,
        title="Australian inflation in the world context",
        ylabel="Per cent per year",
        lfooter=(
            f"OECD monitored excluding: {', '.join(EXCLUDE)}. "
            f"Mean/median calculated when >{int(MEAN_MEDIAN * 100)}% of nations report."
        ),
        axhspan=TARGET,
        xlabel=None,
        y0=True,
        rfooter=SOURCE,
        legend={"loc": "best", "fontsize": "xx-small"},
    )


def inflation_groups(data: pd.DataFrame) -> None:
    """Annual inflation for each country group."""
    country_group_charts(
        data,
        title="Annual Consumer Price Inflation",
        ylabel="Per cent per Year",
        axhspan=TARGET,
        rfooter=SOURCE,
        lfooter="OECD monitored nations. Annual change, original series.",
    )


# --- table of contents, in run order
CHARTS = (
    (inflation_ranges, ()),
    (inflation_world, ()),
    (inflation_groups, ()),
)
