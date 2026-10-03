"""Neutral rate proxies: the AOFM and NY Fed 5y5y risk-neutral forwards, the cash rate and trend growth.

The AOFM forward is set against the cash rate and trend growth, then beside the US (NY Fed
ACM) forward. 5y5y = 2 x RNY10 - RNY5: the average expected short rate over the five years beginning five
years from now, with the term premium removed by an affine model. Neither line is r*: the
forward is a price, and trend per-capita growth plus the inflation target is the golden-rule
statement of where a nominal neutral rate should sit. The AOFM series is re-estimated in
full every month, so it is not a real-time series.
"""

# --- dependencies
from dataclasses import dataclass

import mgplot as mg
import pandas as pd
import readabs as ra
from readabs import metacol as mc

from au_econ.series.rates import get_cash_rate
from au_econ.sources import aofm, nyfed
from au_econ.sources.abs import fetch_release

# --- module contract
RELEASE = ("rstar",)
TOPICS = ("international",)
TITLE = "Neutral Rate Proxies"

# --- constants
AOFM_METHOD = "bc"  # bias-corrected, as the AOFM defaults; "ols" is plain ACM
ABS_NA_CAT = "5206.0"
ABS_NA_TABLE = "5206001_Key_Aggregates"
GDP_PER_CAPITA_DID = "GDP per capita: Chain volume measures ;"  # the trailing " ;" excludes "- Percentage changes"
INFLATION_TARGET = 2.5  # the midpoint of the RBA's band: the target, not survey expectations
TREND_WINDOW_QTRS = 40  # ten years, so neither the mining boom nor the pandemic owns the window
QUARTERS_PER_YEAR = 4
COMPLETE_PERIOD_SHARE = 0.8  # a month is complete with this share of the median trading days
RSTAR_START = "1993-01"
CORRELATION_START = "1992-07"
RBA_CASH_TABLE = "A2"

FORWARD_LABEL = "AOFM 5y5y risk-neutral forward"  # the same label on both charts: the same series
US_FORWARD_LABEL = "US ACM 5y5y risk-neutral forward"
CASH_LABEL = "Cash rate"
TREND_G_LABEL = (
    f"Trend GDP per capita growth ({TREND_WINDOW_QTRS // QUARTERS_PER_YEAR}yr avg) + {INFLATION_TARGET}% target"
)
RSTAR_NOTE = (
    "The trend growth line would have been pinned\n"
    "by inflation expectations, which did not settle\n"
    "at the RBA's target until around 1998 - so\n"
    "before then it sits too low."
)
RSTAR_NOTE_XY = (0.02, 0.04)  # axes fractions: the empty bottom-left corner
RSTAR_NOTE_FONTSIZE = "small"
NOTE_BOX = {"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "darkgrey", "alpha": 0.8}


@dataclass(frozen=True)
class NeutralRateData:
    """Daily risk-neutral yield curves (AOFM, NY Fed ACM); the monthly cash rate; quarterly real GDP per capita."""

    aofm: pd.DataFrame
    acm: pd.DataFrame
    cash_rate: pd.Series
    gdp_per_capita: pd.Series


# --- data
def _gdp_per_capita() -> pd.Series:
    """Real GDP per capita: quarterly, seasonally adjusted, chain volume measures."""
    release = fetch_release(ABS_NA_CAT, single_excel_only=ABS_NA_TABLE, verbose=False)
    _, series_id, _ = ra.find_abs_id(
        release.meta,
        {ABS_NA_TABLE: mc.table, GDP_PER_CAPITA_DID: mc.did, "Seasonally Adjusted": mc.stype},
    )
    series = release.data[ABS_NA_TABLE][series_id].dropna()
    if series.empty:
        raise ValueError(f"ABS {ABS_NA_CAT} returned no {GDP_PER_CAPITA_DID!r} values")
    return series


def fetch() -> NeutralRateData:
    """Fetch the two decompositions, the cash rate target and real GDP per capita."""
    return NeutralRateData(
        aofm=aofm.get_term_premium(AOFM_METHOD),
        acm=nyfed.get_acm_daily(),
        cash_rate=get_cash_rate(),
        gdp_per_capita=_gdp_per_capita(),
    )


# --- helpers
def _five_year_five_year(five: pd.Series, ten: pd.Series) -> pd.Series:
    """Return the 5y5y forward from 5- and 10-year zero-coupon risk-neutral yields: 2 x RNY10 - RNY5."""
    forward = (2.0 * ten - five).dropna()
    if forward.empty:
        raise ValueError("The 5- and 10-year risk-neutral yields share no dates")
    return forward


def _complete_months(daily: pd.Series) -> pd.Series:
    """Monthly means of a daily series, with an incomplete final month dropped.

    A mean does not hand the month to whatever moved on its last trading day, and using
    it for both countries keeps the comparison like for like.
    """
    clean = daily.dropna()
    grouped = clean.groupby(pd.PeriodIndex(clean.index, freq="M"))
    counts, means = grouped.count(), grouped.mean()
    if len(counts) and counts.iloc[-1] < COMPLETE_PERIOD_SHARE * counts.median():
        means = means.iloc[:-1]
    return means


def _aofm_forward(data: NeutralRateData) -> pd.Series:
    """Return the AOFM 5y5y forward, monthly means."""
    forward = _five_year_five_year(data.aofm["RNY5"], data.aofm["RNY10"])
    return _complete_months(forward).rename(FORWARD_LABEL)


def _trend_growth_nominal(data: NeutralRateData, index: pd.PeriodIndex) -> pd.Series:
    """Trend real per-capita GDP growth plus the inflation target, on a monthly index.

    Per capita, not aggregate: the consumption-Euler link is about growth per head. Each
    quarter's value sits on its own last month and the months between are interpolated;
    the fill stops at the last published quarter.
    """
    per_capita = data.gdp_per_capita.astype(float)
    per_capita.index = pd.PeriodIndex(per_capita.index, freq="Q")
    yearly = (per_capita / per_capita.shift(QUARTERS_PER_YEAR) - 1) * 100
    trend = (yearly.rolling(TREND_WINDOW_QTRS).mean() + INFLATION_TARGET).dropna()
    trend.index = pd.PeriodIndex(trend.index).asfreq("M", how="E")  # each quarter on its last month
    monthly = trend.reindex(trend.index.union(index)).interpolate("linear", limit_area="inside")
    return monthly.reindex(index).rename(TREND_G_LABEL)


# --- charts
def rstar_proxies(data: NeutralRateData) -> None:
    """Chart the AOFM 5y5y forward against the cash rate target and trend growth plus the target.

    The cash rate is drawn as steps (it moves only when the Board decides); the boxed note
    is the one raw-matplotlib call, since mgplot has no annotation function.
    """
    forward = _aofm_forward(data)
    index = pd.PeriodIndex(forward.index, freq="M")
    frame = pd.DataFrame(
        {
            FORWARD_LABEL: forward,
            CASH_LABEL: data.cash_rate.reindex(index),
            TREND_G_LABEL: _trend_growth_nominal(data, index),
        }
    )
    frame = frame[frame.index >= pd.Period(RSTAR_START, freq="M")]
    spread = (frame[FORWARD_LABEL] - frame[CASH_LABEL]).dropna()
    axes = mg.line_plot(
        frame,
        style=["-", "--", "-."],
        width=[2.0, 1.5, 1.8],
        drawstyle=["default", "steps-post", "default"],
        annotate=True,
        rounding=2,
    )
    axes.text(
        *RSTAR_NOTE_XY,
        RSTAR_NOTE,
        transform=axes.transAxes,
        fontsize=RSTAR_NOTE_FONTSIZE,
        ha="left",
        va="bottom",
        bbox=NOTE_BOX,
    )
    mg.finalise_plot(
        axes,
        title="Macroeconomic Proxies for Nominal r*",
        ylabel="Per cent, nominal",
        xlabel=None,
        legend={"loc": "best", "fontsize": "small"},
        lheader=f"Forward less cash rate: latest {spread.iloc[-1]:+.2f}, mean {spread.mean():+.2f}pp",
        lfooter=(
            "Australia. Monthly. Forward: monthly average; cash rate: target; "
            "growth trend: quarterly, interpolated. "
        ),
        rfooter=f"AOFM ({AOFM_METHOD.upper()}); RBA: {RBA_CASH_TABLE}; ABS: {ABS_NA_CAT}",
    )


def forward_comparison(data: NeutralRateData) -> None:
    """Chart the Australian and US 5y5y forwards, monthly means, with level and change correlations."""
    united_states = _complete_months(_five_year_five_year(data.acm["ACMRNY05"], data.acm["ACMRNY10"]))
    frame = pd.concat([_aofm_forward(data), united_states.rename(US_FORWARD_LABEL)], axis=1).dropna()
    frame = frame[frame.index >= pd.Period(CORRELATION_START, freq="M")]
    levels = frame[FORWARD_LABEL].corr(frame[US_FORWARD_LABEL])
    changes = frame.diff().dropna()
    monthly = changes[FORWARD_LABEL].corr(changes[US_FORWARD_LABEL])
    mg.line_plot_finalise(
        frame,
        title="Australian and US 5y5y Risk-Neutral Forwards",
        ylabel="Per cent, nominal",
        xlabel=None,
        width=1,
        annotate=True,
        rounding=2,
        legend={"loc": "best", "fontsize": "small"},
        lheader=(
            f"Correlation over {frame.index[0]} to {frame.index[-1]}: "
            f"levels {levels:+.2f}, monthly changes {monthly:+.2f}"
        ),
        lfooter=(
            "Monthly averages. Both are 2 x RNY10 - RNY5. "
            f"AU bias-corrected ({AOFM_METHOD.upper()}), US plain ACM. "
        ),
        rfooter="AOFM: term premium; NY Fed: ACM",
    )


# --- table of contents, in run order
CHARTS = (
    (rstar_proxies, ()),
    (forward_comparison, ()),
)
