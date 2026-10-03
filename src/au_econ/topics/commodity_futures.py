"""Commodity futures and indices: front-month prices for metals, grains and softs, and broad commodity indices.

Futures prices are Yahoo Finance front-month closes. Yahoo no longer carries the Bloomberg
Commodity Index (^BCOM), so the broad index charts use two stand-ins: the iPath Bloomberg
Commodity Index Total Return ETN (DJP), a traded note tracking the index, for the recent
daily chart; and the IMF's monthly all-commodities price index, through FRED, for the
long history and the comparison with Australian inflation.
"""

# --- dependencies
from dataclasses import dataclass

import mgplot as mg
import pandas as pd

from au_econ.charting.daily_prices import (
    SOURCE_YAHOO,
    fetch_closes,
    frame_chart,
    last_day,
    single_charts,
    summarise,
)
from au_econ.series.prices import get_cpi
from au_econ.sources import fred

# --- module contract
RELEASE = ("yahoo",)
TOPICS = ("commodities",)
TITLE = "Commodity Futures"

# --- constants
LONG_START = "2024-03-20"  # about two years
METALS = [
    ("GC=F", "Gold", "USD per troy ounce"),
    ("SI=F", "Silver", "USD per troy ounce"),
    ("PL=F", "Platinum", "USD per troy ounce"),
    ("PA=F", "Palladium", "USD per troy ounce"),
    ("HG=F", "Copper", "USD per pound"),
    ("TIO=F", "Iron Ore", "USD per tonne"),
    ("MTF=F", "Coal (API2 Rotterdam)", "USD per tonne"),
]
GRAINS = {"ZW=F": "Wheat", "ZC=F": "Corn"}
SOFTS = [
    ("ZS=F", "Soybeans", "USD cents per bushel"),
    ("KC=F", "Coffee", "USD cents per pound"),
    ("SB=F", "Sugar", "USD cents per pound"),
    ("CC=F", "Cocoa", "USD per tonne"),
    ("CT=F", "Cotton", "USD cents per pound"),
    ("ZR=F", "Rice", "USD per hundredweight"),
    ("UFV=F", "Urea", "USD per tonne"),
]

# broad commodity indices
BCOM_ETN = "DJP"  # iPath Bloomberg Commodity Index Total Return ETN
IMF_ALL_COMMODITIES = "PALLFNFINDEXM"  # FRED: IMF global price index of all commodities, monthly, 2016 = 100
IMF_START = "1992-01-01"
SOURCE_IMF = "Source: IMF via FRED"
MONTHS_IN_YEAR, QUARTERS_IN_YEAR = 12, 4
PERCENT = 100
COMMODITY_SCALE = 10  # divides commodity price growth onto the CPI scale (it swings about 10-14 times as much)
CPI_SHADE_ABOVE = 3.0  # per cent: the top of the RBA's target band
CPI_SHADE = {"color": "goldenrod", "alpha": 0.18}
COMMODITY_LABEL = f"Commodity prices YoY (IMF, monthly, divided by {COMMODITY_SCALE} for scale)"
CPI_LABEL = "AU CPI YoY (quarterly, SA)"


@dataclass(frozen=True)
class FuturesData:
    """Daily closes by Yahoo ticker; the monthly IMF commodity index; and the quarterly SA CPI index."""

    closes: dict[str, pd.Series]
    imf_index: pd.Series
    cpi: pd.Series


# --- data
def fetch() -> FuturesData:
    """Fetch every futures close and the commodity ETN from LONG_START, the IMF index and CPI."""
    tickers = [ticker for ticker, _, _ in METALS] + list(GRAINS) + [ticker for ticker, _, _ in SOFTS] + [BCOM_ETN]
    imf = fred.get_series(IMF_ALL_COMMODITIES, IMF_START)
    imf.index = pd.PeriodIndex(imf.index, freq="M")
    cpi, _, _ = get_cpi("headline_sa")
    return FuturesData(closes=fetch_closes(tickers, LONG_START), imf_index=imf.dropna(), cpi=cpi)


# --- helpers
def _month_label(series: pd.Series) -> str:
    """Return the last period of a monthly series as e.g. "Jul 2026"."""
    last = series.dropna().index[-1]
    if not isinstance(last, pd.Period):
        raise TypeError("Expected a PeriodIndex")
    return last.strftime("%b %Y")


def _cpi_runs_above(cpi_yoy: pd.Series, threshold: float) -> list[tuple[pd.Period, pd.Period]]:
    """Return the (first, last) quarters of each unbroken run with CPI growth above threshold."""
    runs: list[tuple[pd.Period, pd.Period]] = []
    start: pd.Period | None = None
    previous: pd.Period | None = None
    for quarter, value in cpi_yoy.items():
        if not isinstance(quarter, pd.Period):
            raise TypeError("Expected a PeriodIndex")
        if value > threshold and start is None:
            start = quarter
        elif value <= threshold and start is not None and previous is not None:
            runs.append((start, previous))
            start = None
        previous = quarter
    if start is not None and previous is not None:
        runs.append((start, previous))
    return runs


# --- charts
def metals(data: FuturesData) -> None:
    """Precious and base metals, iron ore and coal: one chart each."""
    single_charts(data.closes, METALS)


def grains(data: FuturesData) -> None:
    """Wheat and corn on one chart."""
    frame = pd.DataFrame({label: data.closes[ticker] for ticker, label in GRAINS.items()}).dropna()
    summarise(frame, "Grains")
    frame_chart(
        frame,
        title="Grain Futures: Wheat and Corn",
        ylabel="USD cents per bushel",
        lfooter=f"CBOT front-month futures. Data to {last_day(frame)}.",
    )


def softs(data: FuturesData) -> None:
    """Soybeans, softs, rice and urea: one chart each."""
    single_charts(data.closes, SOFTS)


def commodity_index(data: FuturesData) -> None:
    """Chart the Bloomberg Commodity Index, through the traded note that tracks it (DJP)."""
    if BCOM_ETN not in data.closes:
        print(f"{BCOM_ETN}: no data, skipping")
        return
    etn = data.closes[BCOM_ETN]
    mg.line_plot_finalise(
        etn,
        title="Bloomberg Commodity Index ETN (DJP)",
        ylabel="USD per note",
        xlabel=None,
        annotate=True,
        lfooter=(
            "Daily close of the iPath BCOM Total Return ETN: the index plus collateral interest, "
            f"less fees. Data to {last_day(etn)}."
        ),
        rfooter=SOURCE_YAHOO,
    )


def commodity_index_history(data: FuturesData) -> None:
    """Chart the IMF all-commodities price index since 1992."""
    mg.line_plot_finalise(
        data.imf_index.rename("IMF all commodities"),
        title="Global Commodity Prices: Since 1992",
        ylabel="Index (2016 = 100)",
        xlabel=None,
        annotate=True,
        lfooter=(
            f"IMF global price index of all commodities, monthly average. Data to {_month_label(data.imf_index)}."
        ),
        rfooter=SOURCE_IMF,
    )


def commodities_vs_cpi(data: FuturesData) -> None:
    """Commodity price growth (scaled) against Australian CPI growth, shading CPI above 3 per cent."""
    commodity_yoy = ((data.imf_index / data.imf_index.shift(MONTHS_IN_YEAR)) - 1) * PERCENT
    commodity_yoy = commodity_yoy.dropna() / COMMODITY_SCALE
    cpi_yoy = (((data.cpi / data.cpi.shift(QUARTERS_IN_YEAR)) - 1) * PERCENT).dropna()
    cpi_yoy.index = pd.PeriodIndex(cpi_yoy.index, freq="Q")
    cpi_monthly = pd.Series(cpi_yoy.to_numpy(), index=cpi_yoy.index.asfreq("M", how="E"), name=CPI_LABEL)
    start = commodity_yoy.index[0]
    frame = pd.DataFrame({COMMODITY_LABEL: commodity_yoy, CPI_LABEL: cpi_monthly})
    frame = frame[frame.index >= start].dropna(how="all")
    spans = [
        {"xmin": first.asfreq("M", how="start"), "xmax": last.asfreq("M", how="end"), **CPI_SHADE}
        for first, last in _cpi_runs_above(cpi_yoy[cpi_yoy.index.asfreq("M", how="E") >= start], CPI_SHADE_ABOVE)
    ]
    mg.line_plot_finalise(
        frame,
        title="Commodity Prices vs Australian CPI: Year-on-Year",
        ylabel="Per cent",
        xlabel=None,
        legend={"loc": "upper left", "fontsize": "x-small"},
        annotate=True,
        rounding=1,
        width=[1.2, 1.6],
        color=["steelblue", "crimson"],
        dropna=True,
        y0=True,
        axvspan=spans,
        lfooter=f"CPI plotted at quarter-end months. Shaded: CPI YoY above {CPI_SHADE_ABOVE:.0f}%.",
        rfooter="Source: IMF via FRED, ABS 6401.0",
    )


# --- table of contents, in run order
CHARTS = (
    (metals, ()),
    (grains, ()),
    (softs, ()),
    (commodity_index, ()),
    (commodity_index_history, ()),
    (commodities_vs_cpi, ()),
)
