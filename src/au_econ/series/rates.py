"""Exchange rates and interest rates used by more than one chart module."""

from functools import cache

import pandas as pd
import readabs as ra

from au_econ.sources.rba import get_historical_table

F11_HISTORY = ("f11hist-1969-2009", "f11hist")  # RBA monthly exchange rate workbooks, earliest first
F11_1_HISTORY = (  # RBA daily exchange rate workbooks, earliest first; daily data begin December 1983
    "1983-1986",
    "1987-1990",
    "1991-1994",
    "1995-1998",
    "1999-2002",
    "2003-2006",
    "2007-2009",
    "2010-2013",
    "2014-2017",
    "2018-2022",
    "2023-current",
)
RBA_SERIES = {"US dollars per Australian dollar": "FXRUSD"}  # label: RBA series ID
CLOSED = "CLOSED"  # the RBA's entry for a day the market was closed, in either case


@cache
def _cash_rate() -> pd.Series:
    """Fetch the monthly cash rate target (cached; not for mutation)."""
    rate = ra.read_rba_ocr(monthly=True).astype(float)
    if rate.empty:
        raise ValueError("RBA A2: no cash rate target values")
    return rate.rename("Cash rate")


def get_cash_rate() -> pd.Series:
    """Return the RBA's announced cash rate target, monthly (the last announced value in each month).

    The policy rate itself, a step function that moves only when the Board decides (table
    A2, ARBAMPCNCRT) - not F1.1's interbank overnight rate, a monthly mean of the realised
    rate that blends the old and new targets in any month a decision lands in.
    """
    return _cash_rate().copy()


@cache
def _aud_usd() -> pd.Series:
    """Fetch and join the monthly AUD/USD history (cached; not for mutation)."""
    series_id = RBA_SERIES["US dollars per Australian dollar"]
    parts = []
    for name in F11_HISTORY:
        part = get_historical_table(name)[series_id].dropna()
        part.index = pd.PeriodIndex(part.index, freq="M")
        parts.append(part)
    aud_usd = pd.concat(parts)
    print("latest AUD/USD exchange rate:", aud_usd.index[-1])
    return aud_usd


def get_aud_usd() -> pd.Series:
    """Return US dollars per Australian dollar, monthly, from 1969.

    Each month's value is the rate on its last business day (the 4 pm WM/Reuters fix
    since July 2008), not a monthly average.
    """
    return _aud_usd().copy()


@cache
def _aud_usd_daily() -> pd.Series:
    """Fetch and join the daily AUD/USD history (cached; not for mutation)."""
    series_id = RBA_SERIES["US dollars per Australian dollar"]
    parts = [get_historical_table(name)[series_id].dropna() for name in F11_1_HISTORY]
    daily = pd.concat(parts)
    daily = daily.mask(daily.astype(str).str.upper() == CLOSED)
    daily = pd.to_numeric(daily, errors="raise").dropna()  # anything else non-numeric raises
    daily.index = pd.DatetimeIndex(daily.index)
    return daily.sort_index()


def get_aud_usd_monthly_average() -> tuple[pd.Series, pd.Period]:
    """Return US dollars per Australian dollar as monthly averages of daily rates, and the first averaged month.

    Months before the first month fully covered by daily data (January 1984) take the
    end-of-month rate from get_aud_usd. The latest month is left out until complete.
    """
    daily = _aud_usd_daily()
    days = daily.index
    if not isinstance(days, pd.DatetimeIndex):
        raise TypeError("expected a DatetimeIndex of daily rates")
    first_full = pd.Period(days[0], freq="M") + 1
    last_day = days[-1]
    complete = last_day == last_day + pd.offsets.BMonthEnd(0)  # the month's last business day is in
    last_full = pd.Period(last_day, freq="M") - (0 if complete else 1)
    average = daily.groupby(days.to_period("M")).mean()
    average = average[(average.index >= first_full) & (average.index <= last_full)]
    end_of_month = get_aud_usd()
    joined = pd.concat([end_of_month[end_of_month.index < first_full], average])
    print(f"AUD/USD monthly averages {first_full} to {last_full}; end-of-month before")
    return joined, first_full
