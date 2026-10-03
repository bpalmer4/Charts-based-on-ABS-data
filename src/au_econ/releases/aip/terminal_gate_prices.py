"""Fuel terminal gate prices (AIP): national petrol and diesel, major cities, and in today's dollars."""

# --- dependencies
from dataclasses import dataclass
from typing import Any

import pandas as pd
from mgplot import line_plot_finalise

from au_econ.series.prices import get_cpi
from au_econ.sources import aip

# --- module contract
RELEASE = ("aip",)
TOPICS = ("commodities", "prices")
TITLE = "Fuel Terminal Gate Prices"

# --- constants
SOURCE = "AIP"
CPI_SOURCE = "ABS: 6401.0"
PETROL, DIESEL = "Petrol (ULP)", "Diesel"
SHEETS = {PETROL: "Petrol TGP", DIESEL: "Diesel TGP"}
NATIONAL = "National Average"
CITIES = ["Sydney", "Melbourne", "Brisbane", "Perth"]
SHORT_START = pd.Period("2026-01-01", freq="D")
SHORT_TAG = "short"
LOOKBACK_QUARTERS = 2  # CPI growth averaged to project the index to today
LEGEND = {"loc": "best", "fontsize": "x-small"}
YLABEL = "Cents per litre"

# Key events, labelled on the chart itself
HORMUZ_CLOSURE = {
    "x": pd.Period("2026-02-28", freq="D"),
    "color": "grey",
    "linestyle": "--",
    "linewidth": 1,
    "text": "Hormuz closure",
}
EXCISE_CUT = [
    {
        "x": pd.Period("2026-04-01", freq="D"),
        "color": "grey",
        "linestyle": "-.",
        "linewidth": 1,
        "text": "Excise cut ~32c/l",
    },
    {
        "x": pd.Period("2026-07-01", freq="D"),
        "color": "grey",
        "linestyle": "-.",
        "linewidth": 1,
        "text": "Excise extension 16c/l",
    },
    {
        "x": pd.Period("2026-08-03", freq="D"),
        "color": "grey",
        "linestyle": "-.",
        "linewidth": 1,
        "text": "Excise relief ended",
    },
]
CEASEFIRE = [
    {
        "x": pd.Period("2026-06-17", freq="D"),
        "color": "grey",
        "linestyle": ":",
        "linewidth": 1,
        "text": "Ceasefire MOU signed",
    },
    {
        "x": pd.Period("2026-07-08", freq="D"),
        "color": "grey",
        "linestyle": ":",
        "linewidth": 1,
        "text": "Ceasefire declared over",
    },
]
EVENTS: list[dict[str, Any]] = [HORMUZ_CLOSURE, *EXCISE_CUT, *CEASEFIRE]


@dataclass(frozen=True)
class FuelData:
    """The petrol and diesel sheets (cents per litre, by city), and the two national averages."""

    petrol: pd.DataFrame
    diesel: pd.DataFrame
    national: pd.DataFrame


# --- data
def fetch() -> FuelData:
    """Fetch the AIP workbook once; every chart function receives its sheets."""
    workbook = aip.get_workbook()
    petrol, diesel = aip.parse_sheet(workbook, SHEETS[PETROL]), aip.parse_sheet(workbook, SHEETS[DIESEL])
    national = pd.DataFrame({PETROL: petrol[NATIONAL], DIESEL: diesel[NATIONAL]}).dropna()
    if national.empty:
        raise ValueError("AIP: no national average prices")
    print(f"Petrol: {petrol.index[0]} to {petrol.index[-1]}")
    print(f"Diesel: {diesel.index[0]} to {diesel.index[-1]}")
    return FuelData(petrol=petrol, diesel=diesel, national=national)


# --- helpers
def _from_short_start(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows from SHORT_START."""
    return frame[frame.index >= SHORT_START]


def _footer(frame: pd.DataFrame, *, city_level: bool = False) -> str:
    """Build the standard left footer: geography, what the prices are, and the latest date."""
    last = frame.index[-1]
    if not isinstance(last, pd.Period):
        raise TypeError("expected a daily PeriodIndex")
    scope = "business days" if city_level else "national average, business days"
    data_to = last.strftime("%-d-%b-%Y")
    return f"Australia. Daily wholesale terminal gate prices (inc. GST), {scope}. Data to {data_to}."


def _daily_deflator() -> tuple[pd.Series, str]:
    """Multipliers into today's dollars, from the seasonally adjusted CPI, and their label.

    The quarterly CPI is projected to the current quarter at its average growth over the
    last LOOKBACK_QUARTERS quarters, then interpolated to daily.
    """
    cpi, _units, _stype = get_cpi("headline_sa")
    quarterly = cpi.copy()
    growth = quarterly.pct_change().iloc[-LOOKBACK_QUARTERS:].mean()
    today = pd.Timestamp.today().normalize()
    this_quarter = pd.Period(today, freq="Q")
    while quarterly.index[-1] <= this_quarter:
        quarterly[quarterly.index[-1] + 1] = quarterly.iloc[-1] * (1 + growth)
    daily = quarterly.to_timestamp(how="end").resample("D").interpolate(method="linear")
    deflator = daily.loc[today] / daily
    deflator.index = deflator.index.to_period("D")
    return deflator, f"{today.strftime('%B %Y')} dollar terms"


def _national_chart(frame: pd.DataFrame, *, tag: str = "", events: list[dict[str, Any]] | None = None) -> None:
    """National petrol and diesel prices."""
    line_plot_finalise(
        frame,
        title="Fuel Terminal Gate Prices: Petrol and Diesel",
        ylabel=YLABEL,
        xlabel=None,
        legend=LEGEND,
        annotate=True,
        lfooter=_footer(frame),
        rfooter=SOURCE,
        tag=tag,  # "" adds nothing to the file name
        axvline=events,  # None draws nothing
    )


# --- charts
def fuel_prices(data: FuelData) -> None:
    """National petrol and diesel prices: the full history, and since SHORT_START with key events."""
    _national_chart(data.national)
    _national_chart(_from_short_start(data.national), tag=SHORT_TAG, events=EVENTS)


def city_prices(data: FuelData) -> None:
    """Petrol and diesel prices in the major cities, since SHORT_START, with key events."""
    for fuel, sheet in ((PETROL, data.petrol), (DIESEL, data.diesel)):
        city_data = _from_short_start(sheet[CITIES]).dropna()
        line_plot_finalise(
            city_data,
            title=f"{fuel} Terminal Gate Prices: Major Cities",
            ylabel=YLABEL,
            xlabel=None,
            legend=LEGEND,
            annotate=True,
            axvline=EVENTS,
            lfooter=_footer(city_data, city_level=True),
            rfooter=SOURCE,
        )


def real_fuel_prices(data: FuelData) -> None:
    """National petrol and diesel prices in today's dollars."""
    deflator, dollar_label = _daily_deflator()
    real = pd.DataFrame({fuel: data.national[fuel] * deflator for fuel in (PETROL, DIESEL)}).dropna()
    line_plot_finalise(
        real,
        title="Real Fuel Terminal Gate Prices: Petrol and Diesel",
        ylabel=YLABEL,
        xlabel=None,
        legend=LEGEND,
        annotate=True,
        lfooter=f"Australia. Daily wholesale TGP (inc. GST). In {dollar_label}, using All Groups CPI (SA).",
        rfooter=f"{SOURCE}; {CPI_SOURCE}",
    )


# --- table of contents, in run order
CHARTS = (
    (fuel_prices, ()),
    (city_prices, ()),
    (real_fuel_prices, ()),
)
