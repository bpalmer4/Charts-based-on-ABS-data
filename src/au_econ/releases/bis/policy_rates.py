"""Central bank policy rates (BIS): daily since 2018 with hand adjustments for reporting lags; monthly since 1980.

The BIS publishes policy rates with a lag of up to about six weeks. ADJUSTMENTS records
decisions the BIS feed has not yet caught up with.
"""

# --- dependencies
from dataclasses import dataclass

import numpy as np
import pandas as pd
from mgplot import finalise_plot, line_plot_finalise

from au_econ.charting.international import world_context_axes
from au_econ.sources import bis

# --- module contract
RELEASE = ("bis",)
TOPICS = ("international",)
TITLE = "Central Bank Policy Rates"

# --- constants
SOURCE = "BIS: WS_CBPOL"
LFOOTER = "Daily data.  Note: There are lags in BIS data reporting. "
DAILY_START = "2018-01-01"
DAILY = {  # country label: BIS code, for the daily charts
    "Argentina": "AR",
    "Australia": "AU",
    "Brazil": "BR",
    "Canada": "CA",
    "Switzerland": "CH",
    "Chile": "CL",
    "China": "CN",
    "Colombia": "CO",
    "Czech Republic": "CZ",
    "Denmark": "DK",
    "United Kingdom": "GB",
    "Hong Kong": "HK",
    "Croatia": "HR",
    "Hungary": "HU",
    "Indonesia": "ID",
    "Israel": "IL",
    "India": "IN",
    "Iceland": "IS",
    "Japan": "JP",
    "South Korea": "KR",
    "Morocco": "MA",
    "North Macedonia": "MK",
    "Mexico": "MX",
    "Malaysia": "MY",
    "Norway": "NO",
    "New Zealand": "NZ",
    "Peru": "PE",
    "Philippines": "PH",
    "Poland": "PL",
    "Romania": "RO",
    "Serbia": "RS",
    "Russian Federation": "RU",
    "Saudi Arabia": "SA",
    "Sweden": "SE",
    "Thailand": "TH",
    "Turkey": "TR",
    "United States": "US",
    "Euro Area": "XM",
    "South Africa": "ZA",
}

# Policy decisions the BIS feed lags: country label: (rate, effective day). One entry per country.
# Applied while no more than ADJUSTMENT_EXPIRY_DAYS old (from today); a caution is logged when
# the BIS data already reaches the effective day; older entries are ignored and can be deleted.
ADJUSTMENT_EXPIRY_DAYS = 42  # six weeks: the longest BIS reporting lag seen
ADJUSTMENTS = {
    "Australia": (4.60, pd.Period("2026-09-30", freq="D")),  # RBA decision 29 Sep 2026
}

COMPARABLE = ["Australia", "Canada", "Euro Area", "New Zealand", "United Kingdom", "United States"]
BIG_ECONOMIES = ["United States", "China", "India", "Euro Area", "United Kingdom", "Japan", "Canada"]
HIGH_INFLATION = ["Argentina", "Russian Federation", "Turkey"]  # outliers
LINE_WIDTH, AUSTRALIA_WIDTH = 1.5, 3
STYLES = ["-", "--", ":", "-."]
MAX_LINES = 40
FEW_LINES, SOME_LINES = 10, 20  # legend font size steps down as lines increase

MONTHLY_START = "1980-01"
MONTHLY = {  # country label: BIS code, for the monthly charts (G20 and G10 members)
    "United States": "US",
    "United Kingdom": "GB",
    "Japan": "JP",
    "Canada": "CA",
    "Australia": "AU",
    "Sweden": "SE",
    "Switzerland": "CH",
    "South Korea": "KR",
    "India": "IN",
    "Indonesia": "ID",
    "Saudi Arabia": "SA",
    "Brazil": "BR",
    "Mexico": "MX",
    "Argentina": "AR",
    "South Africa": "ZA",
    "Russian Federation": "RU",
    "Turkey": "TR",
    "China": "CN",
    "Euro Area": "XM",
}
MONTHLY_SOURCE = SOURCE
KEY_ECONOMIES = ["United States", "Euro Area", "Japan", "United Kingdom", "Australia", "China"]
RATE_EXCLUDE = ["Argentina", "Russian Federation", "Turkey"]  # high-inflation outliers
MEAN_MEDIAN = 0.80  # share of nations reporting before a mean or median is shown
LONG_HISTORY = ["United States", "United Kingdom", "Japan", "Canada", "Australia", "Sweden", "Switzerland"]
LONG_HISTORY_START = pd.Period("1980-01", freq="M")


@dataclass(frozen=True)
class PolicyRates:
    """Policy rates (per cent): daily since DAILY_START, adjusted for lags; monthly since MONTHLY_START.

    adjusted lists the adjustments applied to the daily rates, e.g. "Australia 4.60 from 30 Sep 2026".
    """

    daily: pd.DataFrame
    monthly: pd.DataFrame
    adjusted: tuple[str, ...]


# --- data
def _fill_daily[T: (pd.Series, pd.DataFrame)](data: T) -> T:
    """Reindex to every day from first to last, sorted, carrying values forward."""
    index = data.index
    if isinstance(index, pd.PeriodIndex):
        days = pd.period_range(start=index.min(), end=index.max(), freq="D")
        return data.reindex(days, fill_value=np.nan).sort_index().ffill()
    if isinstance(index, pd.DatetimeIndex):
        dates = pd.date_range(start=index.min(), end=index.max(), freq="D")
        return data.reindex(dates, fill_value=np.nan).sort_index().ffill()
    raise TypeError("expected a DatetimeIndex or PeriodIndex")


def _daily() -> pd.DataFrame:
    """Daily rates, one column per country, each carried forward within its own reported span."""
    rows = bis.get_policy_rates("D", sorted(DAILY.values()), DAILY_START)
    columns: dict[str, pd.Series] = {}
    finals = []
    for label, code in sorted(DAILY.items(), key=lambda item: item[1]):
        country = rows[rows["REF_AREA"] == code]
        if country.empty:
            print(f"No data: {label}")
            continue
        series = pd.Series(
            country["OBS_VALUE"].to_numpy(), name=label, dtype=float, index=pd.to_datetime(country["TIME_PERIOD"])
        )
        if series.isna().all():
            print(f"Empty: {label}")
            continue
        columns[label] = _fill_daily(series)
        finals.append(f"{code}: {columns[label].iloc[-1]:.2f}")
    frame = pd.DataFrame(columns)
    frame.index = pd.PeriodIndex(frame.index, freq="D")
    print(f"Latest: {', '.join(finals)}")
    print(f"Data shape: {frame.shape}; {frame.index[0]} to {frame.index[-1]}")
    return frame


def _adjusted(
    frame: pd.DataFrame, adjustments: dict[str, tuple[float, pd.Period]], today: pd.Period
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Apply the adjustments that are no more than ADJUSTMENT_EXPIRY_DAYS old; return them, described.

    An adjustment sets its country's rate from its effective day on. If that day is past
    the frame's last day, the frame is first extended to it, carrying every country forward.
    A caution is logged when the BIS data for the country already reaches the effective day.
    """
    reported = {country: frame[country].last_valid_index() for country in adjustments}  # before any extension
    applied = []
    for country, (rate, day) in adjustments.items():
        age = (today - day).n
        if age > ADJUSTMENT_EXPIRY_DAYS:
            print(f"Adjustment ignored ({age} days old; delete it): {country} {rate} from {day}")
            continue
        reported_to = reported[country]
        if isinstance(reported_to, pd.Period) and reported_to >= day:
            print(f"Caution: BIS already reports {country} to {reported_to}; applying {rate} from {day} anyway")
        if day > frame.index[-1]:
            frame.loc[day, country] = rate
            frame = _fill_daily(frame)
        else:
            frame.loc[day, country] = rate
            frame.loc[frame.index > day, country] = np.nan
            frame[country] = frame[country].ffill()
        applied.append(f"{country} {rate:.2f} from {day.strftime('%-d %b %Y')}")
    return frame, tuple(applied)


def _monthly() -> pd.DataFrame:
    """Monthly rates since MONTHLY_START, one column per country."""
    rows = bis.get_policy_rates("M", MONTHLY.values(), MONTHLY_START)
    columns: dict[str, pd.Series] = {}
    for label, code in MONTHLY.items():
        country = rows[rows["REF_AREA"] == code]
        if country.empty:
            print(f"  {label}: no data")
            continue
        series = pd.Series(
            country["OBS_VALUE"].to_numpy(),
            index=pd.PeriodIndex(country["TIME_PERIOD"], freq="M"),
            name=label,
            dtype=float,
        )
        columns[label] = series
        print(f"  {label}: {series.index.min()} to {series.index.max()}")
    return pd.DataFrame(columns).sort_index()


def fetch() -> PolicyRates:
    """Fetch the daily and monthly rates once (one request each); apply the adjustments to the daily."""
    today = pd.Period(pd.Timestamp.today(), freq="D")
    daily, adjusted = _adjusted(_daily(), ADJUSTMENTS, today)
    return PolicyRates(daily=daily, monthly=_monthly(), adjusted=adjusted)


# --- helpers
def _adjusted_note(data: PolicyRates) -> str:
    """Header note naming the lag adjustments applied to the daily rates ("" draws nothing)."""
    return f"Adjusted for BIS lags: {'; '.join(data.adjusted)}." if data.adjusted else ""


def _rates_chart(dataset: pd.DataFrame, tag: str, note: str) -> None:
    """Daily rates since DAILY_START for a group of countries, Australia drawn wider."""
    widths = [LINE_WIDTH] * MAX_LINES
    if "Australia" in dataset.columns:
        widths[list(dataset.columns).index("Australia")] = AUSTRALIA_WIDTH
    count = len(dataset.columns)
    font_size = 10 if count < FEW_LINES else 8 if count < SOME_LINES else 6
    line_plot_finalise(
        dataset[dataset.index >= pd.Period(DAILY_START, freq="D")],
        title="Central Bank Policy Rates",
        ylabel="Annual Policy Rate (%)",
        rfooter=SOURCE,
        lfooter=LFOOTER,
        width=widths,
        style=STYLES * (MAX_LINES // len(STYLES)),
        legend={"ncols": 3, "loc": "upper left", "fontsize": font_size},
        y0=True,
        zero_y=True,
        tag=tag,
        rheader=note,
    )


def _available(frame: pd.DataFrame, wanted: list[str]) -> list[str]:
    """Those of wanted that are columns of frame, in order (BIS coverage varies)."""
    return [column for column in wanted if column in frame.columns]


# --- charts
def daily_rate_groups(data: PolicyRates) -> None:
    """Daily rates: comparable economies, big economies, all but the outliers, and the outliers."""
    daily = data.daily
    groups = {  # file-name tag: countries
        "comparable": COMPARABLE,
        "big-economies": BIG_ECONOMIES,
        "all": sorted(column for column in daily.columns if column not in HIGH_INFLATION),
        "high-inflation": HIGH_INFLATION,
    }
    for tag, group in groups.items():
        _rates_chart(daily[group], tag, _adjusted_note(data))


def daily_world(data: PolicyRates) -> None:
    """Australia's policy rate against every other country, with their mean and median."""
    keep = [column for column in data.daily.columns if column not in HIGH_INFLATION]
    ax = world_context_axes(data.daily[keep], label="BIS monitored")
    finalise_plot(
        ax,
        title="CB Policy Rates: Australia in World Context",
        ylabel="Annual Policy Rate (%)",
        lfooter=LFOOTER + f" Excluded: {', '.join(HIGH_INFLATION)}.",
        rheader=_adjusted_note(data),
        xlabel=None,
        y0=True,
        rfooter=SOURCE,
        legend={"loc": "best", "fontsize": "xx-small"},
    )


def monthly_key_economies(data: PolicyRates) -> None:
    """Monthly policy rates for the key economies."""
    line_plot_finalise(
        data.monthly[_available(data.monthly, KEY_ECONOMIES)].dropna(how="all"),
        title="Central bank policy rates - key economies",
        ylabel="Per cent per annum",
        width=2,
        y0=True,
        zero_y=True,
        rfooter=MONTHLY_SOURCE,
        lfooter="Monthly data.",
        legend={"loc": "best", "fontsize": "x-small", "ncol": 2},
    )


def monthly_g20(data: PolicyRates) -> None:
    """G20 mean and median policy rate (without the high-inflation outliers), against Australia."""
    clean = data.monthly[[column for column in data.monthly.columns if column not in RATE_EXCLUDE]]
    enough = clean.notna().sum(axis=1) >= len(clean.columns) * MEAN_MEDIAN
    summary = pd.DataFrame(
        {"G20 mean": clean.mean(axis=1).where(enough), "G20 median": clean.median(axis=1).where(enough)}
    )
    if "Australia" in data.monthly.columns:
        summary["Australia"] = data.monthly["Australia"]
    line_plot_finalise(
        summary.dropna(how="all"),
        title="G20 policy rates - mean and median",
        ylabel="Per cent per annum",
        width=[2.5, 2.5, 3],
        style=["--", ":", "-"],
        y0=True,
        zero_y=True,
        rfooter=MONTHLY_SOURCE,
        lfooter=(
            f"Excluding: {', '.join(RATE_EXCLUDE)}. "
            f"Mean/median where >{int(MEAN_MEDIAN * 100)}% of nations report. Monthly."
        ),
        legend={"loc": "best", "fontsize": "small"},
    )


def monthly_long_history(data: PolicyRates) -> None:
    """Monthly policy rates for the nations with BIS data back to 1980."""
    rates = data.monthly[_available(data.monthly, LONG_HISTORY)]
    line_plot_finalise(
        rates[rates.index >= LONG_HISTORY_START].dropna(how="all"),
        title="Central bank policy rates since 1980",
        ylabel="Per cent per annum",
        width=2,
        y0=True,
        zero_y=True,
        rfooter=MONTHLY_SOURCE,
        lfooter="Monthly data. Nations with BIS data from 1980.",
        legend={"loc": "best", "fontsize": "x-small", "ncol": 2},
    )


# --- table of contents, in run order
CHARTS = (
    (daily_rate_groups, ()),
    (daily_world, ()),
    (monthly_key_economies, ()),
    (monthly_g20, ()),
    (monthly_long_history, ()),
)
