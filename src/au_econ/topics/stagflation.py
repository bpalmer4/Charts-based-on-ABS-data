"""Stagflation era, 1970-1995: real GDP growth, CPI inflation and unemployment for nine countries.

One figure per country, three stacked panels on a common 1970Q1-1995Q4 axis, with
goldenrod shading over runs of two or more quarters of negative real GDP growth.
FRED for most series; DBnomics (OECD) where FRED's own mirrors start too late.
"""

# --- dependencies
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import readabs as ra
from mgplot import finalise_plot, line_plot

from au_econ.sources import dbnomics, fred

# --- module contract
RELEASE = ("stagflation",)
TOPICS = ("international",)
TITLE = "Stagflation Era 1970-1995"

# --- constants
WINDOW_START = pd.Period("1970Q1", freq="Q")
WINDOW_END = pd.Period("1995Q4", freq="Q")
FRED_START = "1965-01-01"
SOURCE = "FRED; DBnomics: OECD QNA, MEI, EO"
RECESSION_NOTE = "Goldenrod shading: 2+ consecutive quarters of negative real Q/Q GDP growth. "
SERIES_TYPE_NOTE = "GDP and UR seas adj; CPI orig."
MIN_RECESSION_RUN = 2  # consecutive negative quarters
QUARTERS_PER_YEAR = 4
ANNUAL_ANCHOR_QUARTER = 2  # an annual value is placed mid-year, then interpolated
ANNUAL_TAIL_QUARTERS = 2  # quarters carried past the last mid-year anchor
PANEL_HEIGHT_IN = 2.2  # inches per panel
FIGURE_WIDTH_IN = 9.0
SHADING = {"color": "goldenrod", "alpha": 0.30, "zorder": -1}
SPLICE_NOTE = (
    "UR: FRED/OECD harmonised quarterly series (1983Q1+) spliced with OECD Economic Outlook "
    "annual UR (interpolated to quarterly) for 1970-1982."
)


@dataclass(frozen=True)
class Country:
    """Where one country's series come from: "fred:<ID>" or "dbnomics:<path>".

    A tuple of sources is spliced, the first preferred. GDP is seasonally adjusted, CPI
    is original (year-on-year growth removes seasonality), UR is seasonally adjusted.
    """

    gdp: str
    cpi: str
    ur: str | tuple[str, ...]
    end: pd.Period | None = None  # truncate the figure here
    note: str | None = None  # shown as the figure's rheader


COUNTRIES = {
    "USA": Country("fred:GDPC1", "fred:CPIAUCSL", "fred:UNRATE"),
    "UK": Country("fred:NAEXKP01GBQ661S", "fred:GBRCPIALLMINMEI", "fred:LRUNTTTTGBQ156S"),
    "Canada": Country("fred:NGDPRSAXDCCAQ", "fred:CANCPIALLMINMEI", "fred:LRHUTTTTCAQ156S"),
    "Australia": Country("fred:NGDPRSAXDCAUQ", "fred:AUSCPIALLQINMEI", "fred:LRHUTTTTAUQ156S"),
    "New Zealand": Country(
        "dbnomics:OECD/QNA/NZL.B1_GE.VOBARSA.Q",
        "fred:NZLCPIALLQINMEI",
        "dbnomics:OECD/MEI/NZL.LRUNTTTT.STSA.A",
        note="NZ UR is an annual rate (interpolated to quarterly); GDP pre-1987 is interpolated from annual data.",
    ),
    "France": Country(
        "fred:NAEXKP01FRQ661S",
        "fred:FRACPIALLMINMEI",
        ("fred:LRHUTTTTFRQ156S", "dbnomics:OECD/EO/FRA.UNR.A"),
        note=SPLICE_NOTE,
    ),
    "West Germany": Country(
        "fred:NAEXKP01DEQ661S",
        "fred:DEUCPIALLMINMEI",
        "fred:LRUNTTTTDEQ156S",
        end=pd.Period("1991Q2", freq="Q"),
        note=(
            "Truncated at 1991Q2: FRED's German real-GDP series switches from West Germany "
            "to Unified Germany after reunification."
        ),
    ),
    "Italy": Country(
        "dbnomics:OECD/QNA/ITA.B1_GE.VOBARSA.Q",
        "fred:ITACPIALLMINMEI",
        ("fred:LRHUTTTTITQ156S", "dbnomics:OECD/EO/ITA.UNR.A"),
        note=SPLICE_NOTE,
    ),
    "Japan": Country("dbnomics:OECD/QNA/JPN.B1_GE.VOBARSA.Q", "fred:JPNCPIALLMINMEI", "fred:LRHUTTTTJPQ156S"),
}


@dataclass(frozen=True)
class Panel:
    """One country's derived quarterly series, in per cent."""

    gdp_qoq: pd.Series
    cpi_yoy: pd.Series
    ur: pd.Series


# --- data
def _fetch(source: str) -> pd.Series:
    """Fetch one tagged source: "fred:<ID>" or "dbnomics:<path>"."""
    provider, _, key = source.partition(":")
    if provider == "fred":
        return fred.get_series(key, FRED_START)
    if provider == "dbnomics":
        return dbnomics.get_series(key)
    raise ValueError(f"unknown source tag in {source!r}")


def _quarterly(series: pd.Series) -> pd.Series:
    """Convert to a quarterly PeriodIndex.

    Quarterly passes through; annual is placed at Q2 of each year and linearly
    interpolated; monthly or finer (DatetimeIndex) is averaged over each quarter.
    """
    index = series.index
    if isinstance(index, pd.PeriodIndex):
        if index.freqstr.startswith("Q"):
            return series.copy()
        if index.freqstr.startswith(("A", "Y")):
            mid = pd.PeriodIndex(
                [pd.Period(year=period.year, quarter=ANNUAL_ANCHOR_QUARTER, freq="Q") for period in index]
            )
            anchored = pd.Series(series.to_numpy(), index=mid, name=series.name)
            full = pd.period_range(anchored.index.min(), anchored.index.max() + ANNUAL_TAIL_QUARTERS, freq="Q")
            return anchored.reindex(full).interpolate(method="linear")
        raise ValueError(f"unsupported PeriodIndex frequency: {index.freqstr}")
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"expected a PeriodIndex or DatetimeIndex, got {type(index).__name__}")
    out = series.resample("QE").mean()
    out.index = pd.PeriodIndex(out.index, freq="Q")
    return out.dropna()


def _resolve(sources: str | tuple[str, ...]) -> pd.Series:
    """Fetch one source quarterly, or splice several (the first preferred where they overlap)."""
    if isinstance(sources, str):
        return _quarterly(_fetch(sources))
    spliced, _report = ra.splice([_quarterly(_fetch(source)) for source in sources], rebase=False)
    return spliced


def fetch() -> dict[str, Panel]:
    """Fetch and derive every country's panel: Q/Q GDP growth, Y/Y CPI inflation, unemployment rate."""
    panels: dict[str, Panel] = {}
    for name, country in COUNTRIES.items():
        gdp, cpi, ur = _resolve(country.gdp), _resolve(country.cpi), _resolve(country.ur)
        print(f"{name:14s}  GDP={len(gdp):>4}  CPI={len(cpi):>4}  UR={len(ur):>4}")
        panels[name] = Panel(
            gdp_qoq=gdp.pct_change() * 100,
            cpi_yoy=cpi.pct_change(QUARTERS_PER_YEAR) * 100,
            ur=ur,
        )
    return panels


# --- helpers
def _recession_spans(gdp_qoq: pd.Series) -> list[tuple[pd.Period, pd.Period]]:
    """Return (start, end) pairs, inclusive, for runs of MIN_RECESSION_RUN+ negative quarters."""
    growth = gdp_qoq.dropna()
    spans: list[tuple[pd.Period, pd.Period]] = []
    run_start: int | None = None
    for i, negative in enumerate([*(growth < 0).tolist(), False]):  # sentinel closes a final run
        if negative and run_start is None:
            run_start = i
        elif not negative and run_start is not None:
            if i - run_start >= MIN_RECESSION_RUN:
                spans.append((growth.index[run_start], growth.index[i - 1]))
            run_start = None
    return spans


def _window(series: pd.Series, end: pd.Period | None) -> pd.Series | None:
    """Slice to the window (and the country's own end), or None if nothing is left."""
    upper = WINDOW_END if end is None else min(WINDOW_END, end)
    out = series.dropna()
    out = out[(out.index >= WINDOW_START) & (out.index <= upper)]
    return out if not out.empty else None


def _plot_country(name: str, panel: Panel) -> list[tuple[pd.Period, pd.Period]]:
    """Draw one country's figure and return its recession spans."""
    country = COUNTRIES[name]
    cap_suffix = "" if country.end is None else f" (to {country.end})"
    gdp_qoq = _window(panel.gdp_qoq, country.end)
    spans = _recession_spans(gdp_qoq) if gdp_qoq is not None else []
    shading = [{"xmin": start, "xmax": end + 1, **SHADING} for start, end in spans] or None

    full = pd.period_range(WINDOW_START, WINDOW_END, freq="Q")  # every figure spans the whole window
    specs = [
        (series.reindex(full), color, title)
        for series, color, title in (
            (gdp_qoq, "navy", "Real GDP, Q/Q growth"),
            (_window(panel.cpi_yoy, country.end), "darkred", "CPI inflation, year-on-year"),
            (_window(panel.ur, country.end), "darkgreen", "Unemployment rate"),
        )
        if series is not None
    ]

    figsize = (FIGURE_WIDTH_IN, float(round(PANEL_HEIGHT_IN * len(specs), 1)))
    _fig, grid = plt.subplots(len(specs), 1, figsize=figsize, sharex=True, squeeze=False)
    for i, (series, color, title) in enumerate(specs):
        ax = grid[i, 0]
        line_plot(series, ax=ax, color=color, annotate=False)
        if i < len(specs) - 1:
            finalise_plot(ax, title=title, ylabel="Per cent", y0=True, axvspan=shading, axes_only=True)
            continue
        finalise_plot(
            ax,
            title=title,
            ylabel="Per cent",
            y0=True,
            axvspan=shading,
            suptitle=f"{name}: Stagflation era and its aftermath 1970-1995{cap_suffix}",
            rfooter=SOURCE,
            lfooter=f"{RECESSION_NOTE}{SERIES_TYPE_NOTE}",
            rheader=country.note or "",  # mgplot prints a None header as "None"
            figsize=figsize,
            tag=name,
        )
    return spans


# --- charts
def countries(data: dict[str, Panel]) -> None:
    """One three-panel figure per country, then a table of every recession span."""
    rows = []
    for name, panel in data.items():
        print(f"\n=== {name} ===")
        spans = _plot_country(name, panel)
        if not spans:
            print("  No 2+ negative quarters in window.")
            continue
        print("  Recession spans (2+ quarters of negative GDP):")
        for start, end in spans:
            quarters = end.ordinal - start.ordinal + 1
            print(f"    {start}  to  {end}  ({quarters} quarters)")
            rows.append({"Country": name, "Start": str(start), "End": str(end), "Quarters": quarters})
    print(pd.DataFrame(rows).sort_values(["Start", "Country"]).reset_index(drop=True))


# --- table of contents, in run order
CHARTS = ((countries, ()),)
