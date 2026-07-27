"""Long-run spliced series that reach back before the modern ABS collections.

Each series joins a published ABS series to older, less comparable evidence, so
that a chart can cover the whole post-war period. The getters are cached per
kernel session and return (series, units, stype); every splice also exposes the
ra.splice() audit report, which should be read rather than assumed:

- get_unemployment_rate() - one monthly unemployment rate back to 1950: the
  published LFS rate, the ABS Modellers' Database rate beneath it, and a
  CES-based backcast beneath that. get_unemployment_splice_report() returns the
  splice audit and get_unemployment_backcast_stats() the regression diagnostics
  for the backcast segment.
- get_productivity_index() - GDP per hour worked back to 1966: the published
  National Accounts index, over a series derived from RBA annual hours data.
  get_productivity_splice_report() returns the splice audit.

The pre-survey segments are estimates, not observations. The unemployment
backcast is a straight line fitted over ten years of overlap and applied to
earlier years; the productivity segment divides quarterly GDP by interpolated
annual hours. Both are honest about where they came from - check the reports
and the fit statistics before leaning on the early years.
"""

# === imports
from __future__ import annotations

import io
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import readabs as ra
from readabs import metacol as mc
from readabs.download_cache import get_file

from abs_gdp import get_gdp

if TYPE_CHECKING:
    from pandas import DataFrame, Series

# === constants
CACHE_DIR = "./CACHE"
JUNE, AUGUST = 6, 8

# === long-run unemployment rate (6202.0 + 1364.0.15.003 + RBA OP8)
_MODELLERS_CAT, _MODELLERS_TABLE = "1364.0.15.003", "1364015003"
_LFS_CAT, _LFS_TABLE = "6202.0", "62020001"

# RBA Occasional Paper 8, table 4.15 (Unemployment). Column 38 is the CES
# registered unemployment rate; column 0 is the financial year, labelled by
# its end year. Data start on row 9. Table note (b): the CES count is taken
# at the end of June each year.
_OP8_URL = "https://www.rba.gov.au/statistics/xls/op8/4-15.xls"
_OP8_SHEET = "4.15"
_OP8_CACHE_PREFIX = "rba_op8"
_OP8_FIRST_DATA_ROW = 9
_OP8_YEAR_COL, _OP8_CES_RATE_COL = 0, 38

# financial years used to fit the CES-to-survey relationship
CALIBRATION_YEARS = (1960, 1970)


@cache
def _get_ces_unemployment_rate() -> Series:
    """CES registered unemployment rate, at end-June each year (RBA OP8 4.15).

    Registered unemployment is not the survey concept: it counts people who
    signed on with the CES seeking full-time work. Table note (b) puts the
    count at the end of June, so each annual figure is placed in June.

    Returns:
        The rate in per cent, on a monthly PeriodIndex of June months. Not for
        mutation (cached).

    """
    content = get_file(_OP8_URL, cache_dir=Path(CACHE_DIR), cache_prefix=_OP8_CACHE_PREFIX)
    raw = pd.read_excel(io.BytesIO(content), sheet_name=_OP8_SHEET, header=None)
    table = raw.loc[_OP8_FIRST_DATA_ROW:, [_OP8_YEAR_COL, _OP8_CES_RATE_COL]].copy()
    table.columns = ["year", "rate"]
    # the year column also holds footnote markers and trailing note text
    table["year"] = pd.to_numeric(table["year"].astype(str).str.extract(r"(\d{4})")[0], errors="coerce")
    table["rate"] = pd.to_numeric(table["rate"], errors="coerce")
    table = table.dropna()
    if table.empty:
        raise ValueError(f"No CES unemployment data found in {_OP8_URL}")

    index = pd.PeriodIndex([pd.Period(year=int(y), month=JUNE, freq="M") for y in table["year"]], freq="M")
    return pd.Series(table["rate"].to_numpy(), index=index, name="CES registered")


@cache
def _get_modellers_unemployment_rate() -> Series:
    """Monthly unemployment rate from the ABS Modellers' Database (1959Q3 on).

    The database publishes quarterly counts rather than the rate, so the rate
    is computed from the two seasonally adjusted series - as in the 6202
    notebook - then interpolated to monthly. The survey was itself quarterly
    until February 1978, so this monthly detail is filled in, not surveyed.

    Returns:
        The rate in per cent, monthly, seasonally adjusted. Not for mutation.

    """
    md, mmeta = ra.read_abs_cat(_MODELLERS_CAT, single_excel_only=_MODELLERS_TABLE, verbose=False)
    unemployed, labour_force = ra.select(
        [
            (md, mmeta, {_MODELLERS_TABLE: mc.table, "Total unemployed ;": mc.did}),
            (md, mmeta, {_MODELLERS_TABLE: mc.table, "Total labour force ;": mc.did}),
        ]
    )  # both are SA counts in '000, so the unit-coherence check passes
    quarterly = (unemployed / labour_force * 100).dropna()
    return ra.qtly_to_monthly(quarterly).rename("Modellers' Database")


@cache
def _get_lfs_unemployment_rate() -> Series:
    """Monthly LFS unemployment rate (6202.0, seasonally adjusted).

    Published monthly from February 1978, when the survey moved off a
    quarterly cycle. Used as published - this is the only genuinely monthly
    segment of the spliced series.

    Returns:
        The rate in per cent, monthly, seasonally adjusted. Not for mutation.

    """
    data, meta = ra.read_abs_cat(_LFS_CAT, single_excel_only=_LFS_TABLE, verbose=False)
    rate = ra.select_one(
        data,
        meta,
        {
            _LFS_TABLE: mc.table,
            "Unemployment rate ;  Persons ;": mc.did,
            "Seasonally Adjusted": mc.stype,
            "Percent": mc.unit,
        },
    )
    return rate.dropna().rename("Labour Force Survey")


def _june_pairs(ces: Series, modellers: Series, window: tuple[int, int]) -> tuple[DataFrame, pd.Period, str]:
    """Pair the CES rate against the survey rate at June, over the fitting window.

    Both sides are taken at June because the CES count is an end-June snapshot;
    fitting it against a financial-year average produces a spurious one-year lag.

    Args:
        ces: the CES registered rate, on June months.
        modellers: the monthly survey rate to calibrate against.
        window: first and last calendar year of the calibration period.

    Returns:
        The paired observations inside the window, the first June of the survey
        series, and the survey series' frequency string.

    """
    index = modellers.index
    if not isinstance(index, pd.PeriodIndex):
        raise TypeError(f"modellers must have a PeriodIndex, got {type(index).__name__}")

    june = modellers[index.month == JUNE]
    paired = pd.DataFrame({"survey": june, "ces": ces}).dropna()
    paired_index = paired.index
    if not isinstance(paired_index, pd.PeriodIndex):
        raise TypeError(f"paired must have a PeriodIndex, got {type(paired_index).__name__}")

    lo, hi = window
    fitted = paired[(paired_index.year >= lo) & (paired_index.year <= hi)]
    if fitted.empty:
        raise ValueError(f"No overlapping June observations in {lo}-{hi} to calibrate on")
    return fitted, june.index[0], index.freqstr


def _backcast_stats(fitted: DataFrame, slope: float, intercept: float, earlier: Series) -> dict[str, float]:
    """Summarise the CES-to-survey fit and the range it is projected over.

    Args:
        fitted: the paired observations the line was fitted to.
        slope: the fitted slope.
        intercept: the fitted intercept.
        earlier: the CES values the line is applied to.

    Returns:
        The fit statistics.

    """
    predicted = slope * fitted["ces"] + intercept
    resid = fitted["survey"] - predicted
    total = ((fitted["survey"] - fitted["survey"].mean()) ** 2).sum()
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(1 - (resid**2).sum() / total),
        "resid_sd": float(resid.std(ddof=2)),
        "n": float(len(fitted)),
        "fitted_ces_min": float(fitted["ces"].min()),
        "fitted_ces_max": float(fitted["ces"].max()),
        "projected_ces_min": float(earlier.min()),
        "projected_ces_max": float(earlier.max()),
    }


def _backcast_unemployment_rate(
    ces: Series,
    modellers: Series,
    window: tuple[int, int] = CALIBRATION_YEARS,
) -> tuple[Series, dict[str, float]]:
    """Model the survey unemployment rate for the years before the Modellers' DB.

    A straight line is fitted from the CES registered rate to the survey rate
    over `window`, then applied to the earlier CES years. Both sides are taken
    at June, because the CES count is an end-June snapshot; fitting it against
    a financial-year average produces a spurious one-year lag. The fitted years
    are the estimate's whole evidence base, so read the returned statistics -
    especially how far the early CES values sit below the fitted range - before
    trusting the result.

    Args:
        ces: the CES registered rate, on June months.
        modellers: the monthly survey rate to calibrate against.
        window: first and last calendar year of the calibration period.

    Returns:
        The monthly estimate for the years before `modellers` begins, and the
        fit statistics (slope, intercept, r2, resid_sd, n, and the fitted CES
        range against the range being projected).

    """
    fitted, first_june, freq = _june_pairs(ces, modellers, window)
    slope, intercept = np.polyfit(fitted["ces"], fitted["survey"], 1)

    earlier = ces[ces.index < first_june]
    estimate = slope * earlier + intercept

    # annual June estimates -> monthly, joining the points without adding turns
    span = pd.period_range(estimate.index[0], estimate.index[-1], freq=freq)
    monthly = estimate.reindex(span).interpolate(method="linear")
    return monthly.rename("Modelled from CES"), _backcast_stats(fitted, slope, intercept, earlier)


@cache
def _get_unemployment_rate() -> tuple[Series, DataFrame, dict[str, float]]:
    """Splice one monthly unemployment rate back to 1950 (cached).

    Priority runs survey-first: the published LFS rate, then the Modellers'
    Database rate, then the CES-based estimate. Rebasing is off - these are
    rates, not index levels. Only the LFS segment is a monthly survey; the
    earlier monthly detail is interpolated rather than observed.

    Returns:
        The spliced rate, the splice report, and the backcast fit statistics.
        Not for mutation (cached).

    """
    lfs = _get_lfs_unemployment_rate()
    modellers = _get_modellers_unemployment_rate()
    ces = _get_ces_unemployment_rate()
    modelled, stats = _backcast_unemployment_rate(ces, modellers)
    rate, report = ra.splice([lfs, modellers, modelled], rebase=False, name="Unemployment rate")
    return rate, report, stats


def get_unemployment_rate() -> tuple[Series, str, str]:
    """Return one monthly unemployment rate spliced back to 1950.

    Three segments, highest priority first: the published LFS rate (6202.0,
    monthly from February 1978), the ABS Modellers' Database rate
    (1364.0.15.003, quarterly counts turned into a rate and interpolated to
    monthly, from 1959Q3), and a CES-based backcast beneath that (RBA
    Occasional Paper 8 table 4.15, calibrated to the survey over 1960-1970).

    Rebasing is off, because these are rates rather than index levels. Only the
    LFS segment is a genuinely monthly survey - read
    get_unemployment_backcast_stats() before relying on the 1950s.

    Returns:
        A (series, units, stype) tuple - the spliced rate (a defensive copy),
        "Percent" and "Seasonally Adjusted".

    """
    rate, _report, _stats = _get_unemployment_rate()
    return rate.copy(), "Percent", "Seasonally Adjusted"


def get_unemployment_splice_report() -> DataFrame:
    """Return the ra.splice() audit report for the unemployment rate.

    Returns:
        The splice report (overlap junctions), a copy.

    """
    _rate, report, _stats = _get_unemployment_rate()
    return report.copy()


def get_unemployment_backcast_stats() -> dict[str, float]:
    """Return the regression diagnostics for the CES backcast segment.

    The fitted CES range against the projected CES range is the key pair: the
    further the early CES values sit outside the range the line was fitted
    over, the more the 1950s estimate is extrapolation.

    Returns:
        The fit statistics (slope, intercept, r2, resid_sd, n, and the fitted
        and projected CES ranges), a copy.

    """
    _rate, _report, stats = _get_unemployment_rate()
    return dict(stats)


# === long-run GDP per hour worked (5206.0 + RBA OP8)
_KEY_AGGREGATES = "5206001_Key_Aggregates"

# RBA Occasional Paper 8, table 4.12 (Aggregate and Average Weekly Hours
# Worked). Column 0 is the calendar year and column 32 the aggregate weekly
# hours worked by all employed persons. The survey is taken in August.
_OP8_HOURS_URL = "https://www.rba.gov.au/statistics/xls/op8/4-12.xls"
_OP8_HOURS_SHEET = "4.12"
_OP8_HOURS_CACHE_PREFIX = "rba_op8_hours"
_OP8_HOURS_YEAR_COL, _OP8_HOURS_TOTAL_COL = 0, 32


@cache
def _get_op8_hours() -> Series:
    """Aggregate weekly hours worked, at August each year (RBA OP8 4.12).

    Returns:
        Aggregate weekly hours, on the quarter containing each August. Not for
        mutation (cached).

    """
    content = get_file(_OP8_HOURS_URL, cache_dir=Path(CACHE_DIR), cache_prefix=_OP8_HOURS_CACHE_PREFIX)
    raw = pd.read_excel(io.BytesIO(content), sheet_name=_OP8_HOURS_SHEET, header=None)
    years = pd.to_numeric(raw[_OP8_HOURS_YEAR_COL].astype(str).str.extract(r"^(\d{4})$")[0], errors="coerce")
    hours = pd.to_numeric(raw[_OP8_HOURS_TOTAL_COL], errors="coerce")
    keep = years.notna() & hours.notna()
    if not keep.any():
        raise ValueError(f"No aggregate hours worked found in {_OP8_HOURS_URL}")
    index = pd.PeriodIndex([pd.Period(year=int(y), month=AUGUST, freq="Q") for y in years[keep]], freq="Q")
    return pd.Series(hours[keep].to_numpy(), index=index, name="Aggregate weekly hours")


def _get_derived_productivity() -> Series:
    """Real GDP per hour worked, built from the annual hours data.

    The hours are an August snapshot, so the quarters between them are
    interpolated, and dividing seasonally adjusted GDP by unadjusted hours
    leaves a level offset against the published index. The rebase in
    _get_productivity_index() removes that offset; only growth is used here.

    Returns:
        Real GDP per aggregate weekly hour, quarterly, on an arbitrary scale.

    """
    gdp, _units = get_gdp(gdp_type="CVM", seasonal="SA")
    hours = _get_op8_hours()
    quarters = pd.period_range(hours.index[0], hours.index[-1], freq="Q")
    gapped = hours.reindex(quarters)
    gapped.index = quarters.to_timestamp()
    filled = gapped.interpolate(method="cubic")
    filled.index = quarters
    return (gdp / filled).dropna().rename("Derived productivity")


@cache
def _get_productivity_index() -> tuple[Series, DataFrame]:
    """Splice GDP per hour worked back to 1966 (cached).

    The published index takes priority and is used unchanged from 1978Q3; the
    derived series fills the years before it. Rebasing is on because this is an
    index on a ratio scale, so rescaling the earlier segment onto the published
    level leaves its growth intact.

    Returns:
        The spliced index and the splice report. Not for mutation (cached).

    """
    data, meta = ra.read_abs_cat("5206.0", single_excel_only=_KEY_AGGREGATES, verbose=False)
    published = ra.select_one(
        data,
        meta,
        {
            _KEY_AGGREGATES: mc.table,
            "GDP per hour worked: Index ;": mc.did,
            "Seasonally Adjusted": mc.stype,
        },
    ).dropna()
    spliced, report = ra.splice([published, _get_derived_productivity()], rebase=True)
    return spliced.rename("GDP per hour worked"), report


def get_productivity_index() -> tuple[Series, str, str]:
    """Return the GDP per hour worked index, spliced back to 1966.

    Two segments: the published National Accounts index (5206.0 Key Aggregates,
    Seasonally Adjusted, unchanged from 1978Q3) over a series derived by
    dividing seasonally adjusted CVM GDP by aggregate weekly hours worked from
    RBA Occasional Paper 8 table 4.12. Those hours are an annual August
    snapshot interpolated to quarters, so the derived segment contributes its
    growth, not its level: the splice rebases it onto the published index.

    Returns:
        A (series, units, stype) tuple - the spliced index (a defensive copy),
        "Index Numbers" and "Seasonally Adjusted".

    """
    spliced, _report = _get_productivity_index()
    return spliced.copy(), "Index Numbers", "Seasonally Adjusted"


def get_productivity_splice_report() -> DataFrame:
    """Return the ra.splice() audit report for the productivity index.

    Returns:
        The splice report (rebase factor and overlap junction), a copy.

    """
    _spliced, report = _get_productivity_index()
    return report.copy()
