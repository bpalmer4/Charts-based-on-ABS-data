"""Government bond yields: 10- and 30-year for five markets, China against the US, and AU less US.

The five markets are the US, Japan, Germany, the UK and Australia (10-year only). Each
country's line begins when its data begins. The series are not computed identically: US
yields are constant maturity, Japan's are benchmark bond yields, Germany's, the UK's and
China's are fitted curves, and Australia's are the RBA's interpolated yields. Publication
lags differ, so each chart's footer gives every series' own last date.
"""

# --- dependencies
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mgplot as mg
import pandas as pd

from au_econ.sources import boe, bundesbank, chinabond, mof, rba, yahoo

if TYPE_CHECKING:
    from collections.abc import Callable

# --- module contract
RELEASE = ("bonds",)
TOPICS = ("international",)
TITLE = "Government Bond Yields"

# --- constants
RECENT_TRADING_DAYS = -750  # roughly the last three years
plot_times = 0, RECENT_TRADING_DAYS
EARLIEST_START = "1960-01-01"  # the earliest date asked of Yahoo; each chart's start trims

# Series code per tenor and country: a Yahoo ticker (US), a MOF curve column (Japan), a
# Bundesbank series key (Germany), a BoE curve maturity in years (UK), or an RBA series
# title (Australia). Australia appears only at ten years: neither the RBA nor the AOFM
# publishes a 30-year yield.
CODES: dict[str, dict[str, str]] = {
    "10-year": {
        "United States": "^TNX",
        "Japan": "10Y",
        "Germany": "D.I.ZST.ZI.EUR.S1311.B.A604.R10XX.R.A.A._Z._Z.A",
        "United Kingdom": "10",
        "Australia": "Australian Government 10 year bond",
    },
    "30-year": {
        "United States": "^TYX",
        "Japan": "30Y",
        "Germany": "D.I.ZST.ZI.EUR.S1311.B.A604.R30XX.R.A.A._Z._Z.A",
        "United Kingdom": "30",
    },
}
SHORT_NAMES = {"United States": "US", "United Kingdom": "UK"}  # for chart titles
ABBREVIATIONS = {  # for the footer, which names every series at once
    "United States": "US",
    "Japan": "JP",
    "Germany": "DE",
    "United Kingdom": "UK",
    "Australia": "AU",
    "China": "CN",
}
SOURCES = {  # each country's source, for the right footer
    "United States": "Yahoo",
    "Japan": "MOF",
    "Germany": "Bundesbank",
    "United Kingdom": "BoE",
    "Australia": "RBA: F2",
    "China": "ChinaBond",
}
TENOR_STARTS = {"10-year": "1986-01-01", "30-year": "1999-01-01"}  # Japan joins; the JGB 30-year opens
TENOR_NOTES = {
    "10-year": "",
    "30-year": (
        "US Feb 2002 to Feb 2006: no 30-year issuance, so ^TYX tracks the "
        "longest bond outstanding. UK curve reaches 30 years only from 2016"
    ),
}
CHINA_US_CODES = {"China": "10", "United States": "^TNX"}
CHINA_US_TENOR = "10-year"
CHINA_US_START = "2006-01-01"
CHINA_US_NOTE = "China: ChinaBond fitted CGB curve. US: ^TNX constant maturity"
SPREAD_TENOR = "10-year"
SPREAD_LEGS = ("Australia", "United States")
SPREAD_NOTE = (
    "RBA interpolated 10-year yield less the ^TNX constant maturity yield: "
    "the same tenor, but not identically computed"
)
RBA_CURRENT_TABLE, RBA_HISTORY_TABLE = "F2", "Z:F2-Daily-2013"  # the RBA split its daily yields in 2013


@dataclass(frozen=True)
class BondYields:
    """Daily yields (per cent): one frame per tenor, countries as columns; and China against the US."""

    tenors: dict[str, pd.DataFrame]
    china_us: pd.DataFrame


# --- data
def _australia(title: str) -> pd.Series:
    """RBA daily yield by series title: the current table extended back by the pre-2013 one."""
    current, meta = rba.get_table(RBA_CURRENT_TABLE)
    matches = meta[meta["Title"] == title]
    if len(matches) != 1:
        raise ValueError(f"RBA {RBA_CURRENT_TABLE} holds {len(matches)} series titled {title!r}")
    series_id = str(matches["Series ID"].iloc[0])  # the two tables share IDs but not metadata layouts
    history, _ = rba.get_table(RBA_HISTORY_TABLE)
    if series_id not in history.columns:
        raise ValueError(f"RBA {RBA_HISTORY_TABLE} has no {series_id} column")
    joined = pd.concat([history[series_id].dropna(), current[series_id].dropna()])
    return joined[~joined.index.duplicated(keep="last")].sort_index()  # the current table wins on overlap


FETCHERS: dict[str, Callable[[str], pd.Series]] = {
    "United States": lambda code: yahoo.get_close(code, EARLIEST_START),
    "Japan": mof.get_jgb_yield,
    "Germany": bundesbank.get_series,
    "United Kingdom": lambda code: boe.get_gilt_yield(float(code)),
    "Australia": _australia,
    "China": chinabond.get_cgb_yield,
}


def _yield(country: str, code: str) -> pd.Series:
    """One country's yield, numeric (several sources hand back object columns, which lose end labels)."""
    series = pd.to_numeric(FETCHERS[country](code), errors="coerce").dropna()
    if series.empty:
        raise ValueError(f"No numeric observations for {country} ({code})")
    return series.rename(country)


def _tenor(name: str, codes: dict[str, str], start: str) -> pd.DataFrame:
    """One tenor's countries on a common daily index; gaps (holidays, late starts) left missing."""
    frame = pd.DataFrame({country: _yield(country, code) for country, code in codes.items()})
    frame = frame[frame.index >= pd.Period(start, freq="D")].dropna(how="all")
    if frame.empty:
        raise ValueError(f"No {name} observations on or after {start}")
    for country in frame.columns:
        column = frame[country].dropna()
        print(
            f"{name} {country}: {column.index[0]} to {column.index[-1]}, "
            f"{len(column)} observations, last {column.iloc[-1]:.2f} per cent"
        )
    return frame


def fetch() -> BondYields:
    """Fetch every tenor, and China against the US."""
    tenors = {name: _tenor(name, codes, TENOR_STARTS[name]) for name, codes in CODES.items()}
    return BondYields(tenors=tenors, china_us=_tenor(CHINA_US_TENOR, CHINA_US_CODES, CHINA_US_START))


# --- helpers
def _join_names(names: list[str]) -> str:
    """Join names as a readable list: "a, b and c"."""
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def _title_countries(countries: list[str]) -> str:
    """Countries for a chart title, the long names shortened."""
    return _join_names([SHORT_NAMES.get(country, country) for country in countries])


def _data_to_footer(data: pd.DataFrame) -> str:
    """Every series' last observation date, grouped where they share one, most recent first."""
    ends: dict[pd.Period, list[str]] = {}
    for country in data.columns:
        last = data[country].dropna().index[-1]
        if not isinstance(last, pd.Period):
            raise TypeError(f"{country}: expected a daily PeriodIndex")
        ends.setdefault(last, []).append(ABBREVIATIONS.get(country, country))
    newest_first = sorted(ends.items(), reverse=True)
    parts = [f"{_join_names(codes)} {end.strftime('%-d-%b-%Y')}" for end, codes in newest_first]
    return f"Data to: {'; '.join(parts)}. "


def _source_footer(data: pd.DataFrame) -> str:
    """Name the publisher of every series on the chart."""
    return "; ".join(SOURCES[country] for country in data.columns)


def _yields_chart(name: str, data: pd.DataFrame, note: str) -> None:
    """One tenor over the full history and the recent window."""
    mg.multi_start(
        data,
        function=mg.line_plot_finalise,
        starts=plot_times,
        title=f"{_title_countries(list(data.columns))}: {name} Government Bond Yields",
        ylabel="Per cent per year",
        xlabel=None,
        width=1,
        annotate=True,
        rounding=2,
        legend={"loc": "best", "fontsize": "small"},
        lfooter=_data_to_footer(data),
        rfooter=_source_footer(data),
        rheader=note,  # "" draws nothing
    )


# --- charts
def yields_10_year(data: BondYields) -> None:
    """10-year yields: US, Japan, Germany, UK and Australia."""
    _yields_chart("10-year", data.tenors["10-year"], TENOR_NOTES["10-year"])


def yields_30_year(data: BondYields) -> None:
    """30-year yields: US, Japan, Germany and UK."""
    _yields_chart("30-year", data.tenors["30-year"], TENOR_NOTES["30-year"])


def china_us(data: BondYields) -> None:
    """China against the US at ten years (China's history starts in 2006)."""
    _yields_chart(CHINA_US_TENOR, data.china_us, CHINA_US_NOTE)


def australia_us_spread(data: BondYields) -> None:
    """Australia less US 10-year yield, on days both markets traded (no spread against a stale yield)."""
    first, second = SPREAD_LEGS
    pair = data.tenors[SPREAD_TENOR][[first, second]].dropna()
    if pair.empty:
        raise ValueError(f"No {SPREAD_TENOR} days on which both {first} and {second} traded")
    legs = f"{_title_countries([first])} less {_title_countries([second])}"
    mg.multi_start(
        pair[first] - pair[second],
        function=mg.line_plot_finalise,
        starts=plot_times,
        title=f"{legs}: {SPREAD_TENOR} Government Bond Spread",
        ylabel="Percentage points",
        xlabel=None,
        width=1,
        annotate=True,
        rounding=2,
        legend=False,
        y0=True,
        rheader=SPREAD_NOTE,
        lfooter=_data_to_footer(pair),
        rfooter=_source_footer(pair),
    )


# --- table of contents, in run order
CHARTS = (
    (yields_10_year, ()),
    (yields_30_year, ()),
    (china_us, ()),
    (australia_us_spread, ()),
)
