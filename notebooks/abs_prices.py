"""Price, CPI, wage and house-price getters for the price / numeraire domain.

The getters are cached per kernel session and return (series, units, stype) -
the series type is reported because it is fixed internally (the caller does not
choose it), so callers can label chart footers correctly. All series are
selected by data-item description, never by series ID (ABS series IDs are
fragile):

- get_price_deflator(measure) - published Implicit Price Deflators from the
  National Accounts (5206.0): DFD (domestic final demand - the default), GNE,
  HFCE, GDP. Seasonally Adjusted index numbers (the ABS publishes the IPDs SA
  only); the published index, not a hand-computed nominal/real ratio (which
  would drift under chain-linking). The GDP deflator is compromised as a
  *domestic* gauge by the terms of trade (it embeds export prices).
- get_cpi(measure) - CPI index (6401.0): headline (All groups, Original -
  reconstructed back to 1948 from the reported quarterly change, artefact-free),
  headline_sa, trimmed (Trimmed Mean) or weighted (Weighted Median). Each on its
  native ABS reference base (YoY is base-invariant; rebase for level plots).
- get_wage_index(measure) - WPI (6345.0, SA quarterly index) or AWOTE (6302.0,
  Original biannual $/week).
- get_house_price_index() - a long-run house-price dollar level spliced back to
  1986 (6432.0 mean value + the discontinued 6416.0 RPPI / established-house
  index), optionally extended to 1970 with the BIS index (extend_bis), CPI
  deflated (real) and/or seasonally adjusted (seasonally_adjusted);
  get_house_price_splice_report() returns the splice audit.

- seasonally_adjust(series) - the seasonal adjustment used above, exposed for
  callers that deflate an Original series themselves and need the residual
  seasonality removed before rebasing to a single period.

This module owns the decompose (statsmodels) dependency for the seasonal
adjustment of the spliced house-price level, which is Original and cannot be
rebased to a single quarter without baking that quarter's seasonal factor in.
"""

# === imports
from __future__ import annotations

import io
from functools import cache
from typing import TYPE_CHECKING

import pandas as pd
import readabs as ra
from readabs import metacol as mc
from readabs.download_cache import request_get

from decompose import FINAL_SEASADJ, decompose

if TYPE_CHECKING:
    from pandas import DataFrame, Series

# === constants
_DEFLATOR_TABLE = "5206005_Expenditure_Implicit_Price_Deflators"
_DEFLATOR_DIDS = {
    "DFD": "Domestic final demand ;",
    "GNE": "Gross national expenditure ;",
    "HFCE": "Households ;  Final consumption expenditure ;",
    "GDP": "GROSS DOMESTIC PRODUCT ;",
}


# === implicit price deflators (National Accounts, 5206.0)
@cache
def _get_price_deflator(measure: str) -> tuple[Series, str, str]:
    """Fetch and select a published IPD index (cached per kernel session).

    Args:
        measure: see get_price_deflator.

    Returns:
        The raw index, its units, and series type. Not for mutation (cached).

    """
    if measure not in _DEFLATOR_DIDS:
        choices = tuple(_DEFLATOR_DIDS)
        raise ValueError(f"Unknown deflator: {measure!r}. Choose from {choices}.")
    data, meta = ra.read_abs_cat("5206.0", single_excel_only=_DEFLATOR_TABLE, verbose=False)
    selector = {
        _DEFLATOR_TABLE: mc.table,
        _DEFLATOR_DIDS[measure]: mc.did,
        "Seasonally Adjusted": mc.stype,
        "Index Numbers": mc.unit,
    }
    table, sid, units = ra.find_abs_id(meta, selector, verbose=False)
    return data[table][sid], units, "Seasonally Adjusted"


def get_price_deflator(measure: str = "DFD") -> tuple[Series, str, str]:
    """Return a National-Accounts Implicit Price Deflator index and its units.

    The published deflators are Seasonally Adjusted index numbers (5206.0).

    Args:
        measure: one of "DFD" (domestic final demand - the default and cleanest
            domestic price gauge), "GNE" (gross national expenditure), "HFCE"
            (household final consumption) or "GDP" (whole-economy; compromised as
            a domestic gauge by the terms of trade, which embed export prices).

    Returns:
        A (series, units, stype) tuple - the deflator index (a defensive copy),
        its units, and its series type (always "Seasonally Adjusted").

    """
    series, units, stype = _get_price_deflator(measure)
    return series.copy(), units, stype


# === wage indices (WPI 6345.0, AWOTE 6302.0)
_WPI_TABLE = "634501"
_WPI_DID = (
    "Quarterly Index ;  Total hourly rates of pay excluding bonuses ;  "
    "Australia ;  Private and Public ;  All industries ;"
)
_AWOTE_TABLE = "6302003"
_AWOTE_DID = "Earnings; Persons; Full Time; Adult; Ordinary time earnings ;"


@cache
def _get_wpi() -> tuple[Series, str, str]:
    """Fetch the Wage Price Index, SA quarterly index (6345.0); cached.

    Returns:
        The raw SA WPI index, its units, and series type. Not for mutation.

    """
    data, meta = ra.read_abs_cat("6345.0", single_excel_only=_WPI_TABLE, verbose=False)
    selector = {
        _WPI_TABLE: mc.table,
        _WPI_DID: mc.did,
        "Seasonally Adjusted": mc.stype,
        "Index Numbers": mc.unit,
    }
    table, sid, units = ra.find_abs_id(meta, selector, verbose=False)
    return data[table][sid], units, "Seasonally Adjusted"


@cache
def _get_awote() -> tuple[Series, str, str]:
    """Fetch AWOTE: weekly full-time adult ordinary-time earnings, Persons ($).

    Original series, published biannually (May and November) on a Q-NOV index;
    the index is reinterpreted onto Q-DEC so it aligns with other quarterly
    series. Selected with exact_match so it does not also pick up the matching
    standard-error series.

    Returns:
        The raw AWOTE level ($/week, Q-DEC), its units, and series type. Not for mutation.

    """
    data, meta = ra.read_abs_cat("6302.0", single_excel_only=_AWOTE_TABLE, verbose=False)
    selector = {_AWOTE_TABLE: mc.table, _AWOTE_DID: mc.did, "Original": mc.stype}
    table, sid, units = ra.find_abs_id(meta, selector, exact_match=True, verbose=False)
    series = data[table][sid].dropna()
    series.index = pd.PeriodIndex(series.index, freq="Q-DEC")
    return series, units, "Original"


def get_wage_index(measure: str = "WPI") -> tuple[Series, str, str]:
    """Return a wage series and its units, by measure.

    The two measures are different objects: WPI is a quarterly Seasonally
    Adjusted price index, AWOTE a biannual Original dollar level.

    Args:
        measure: "WPI" (Wage Price Index, total hourly rates excluding bonuses,
            all industries; SA quarterly index) or "AWOTE" (average weekly
            ordinary-time earnings, full-time adults, Persons; Original $/week,
            biannual, reindexed Q-NOV -> Q-DEC).

    Returns:
        A (series, units, stype) tuple - the series (a defensive copy), its
        units, and its series type ("Seasonally Adjusted" WPI, "Original" AWOTE).

    """
    if measure == "WPI":
        series, units, stype = _get_wpi()
    elif measure == "AWOTE":
        series, units, stype = _get_awote()
    else:
        raise ValueError(f"Unknown wage measure: {measure!r}. Choose from ('WPI', 'AWOTE').")
    return series.copy(), units, stype


# === long-run house-price index (spliced 6432.0 mean value + discontinued 6416.0)
# The discontinued 6416.0 final release (Dec 2021) landing page - readabs url override.
_RPPI_URL = (
    "https://www.abs.gov.au/statistics/economy/price-indexes-and-inflation/"
    "residential-property-price-indexes-eight-capital-cities/dec-2021"
)
_MEAN_PRICE_DID = "Mean price of residential dwellings ;  Australia ;"
_RPPI_DID = "Residential Property Price Index ;  Weighted average of eight capital cities ;"
_ESTABLISHED_DID = "Price Index of Established Homes ;  Weighted Average of 8 Capital Cities ;"

# BIS selected residential property prices (dataflow WS_SPP), quarterly nominal
# index for Australia, back to 1970Q1. BIS documents the series as the ABS
# all-dwellings RPPI from 2003Q3, ABS established houses from 1986Q3, and REIA
# median dwelling prices for the state capitals before that - so underneath the
# ABS splice the only thing it contributes is 1970Q1 to 1986Q1, and that part is
# a median rather than a quality-adjusted index: it moves with the composition
# of what sold as well as with prices.
_BIS_URL = "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_SPP/1.0/Q.AU?format=csv"
_BIS_NOMINAL, _BIS_INDEX_UNIT = "N", 628
# Quarters of overlap kept when rebasing the BIS segment onto the ABS dollar
# level. ra.splice() fits a single factor over whatever overlap it is handed,
# and the two series drift apart over the full 40 years (12.1x against 13.3x):
# the whole overlap invents a 2.7 per cent fall at the junction, one quarter
# anchors on a single noisy median print. A year splits the difference.
_BIS_OVERLAP_QUARTERS = 4


@cache
def _get_bis_house_prices() -> Series:
    """Fetch the BIS nominal residential property price index for Australia.

    Quarterly from 1970Q1. The BIS API is used rather than the ABS because the
    pre-1986 REIA median prices are not published in any ABS time series.

    Returns:
        The index, on a quarterly PeriodIndex. Not for mutation (cached).

    """
    # request_get() rather than get_file(): get_file() HEADs the URL first to test
    # freshness against its cache, and the BIS SDMX API answers HEAD with a 500
    # (a plain GET is fine). request_get() is readabs' straight GET - no cache, but
    # the payload is small and a real failure still raises rather than going quiet.
    content = request_get(_BIS_URL)
    raw = pd.read_csv(io.BytesIO(content))
    wanted = raw[(raw["VALUE"] == _BIS_NOMINAL) & (raw["UNIT_MEASURE"] == _BIS_INDEX_UNIT)]
    if wanted.empty:
        raise ValueError("No nominal BIS residential property price index for Australia")

    index = pd.PeriodIndex(wanted["TIME_PERIOD"].str.replace("-", ""), freq="Q-DEC")
    series = pd.Series(wanted["OBS_VALUE"].to_numpy(dtype=float), index=index).sort_index()
    return series.rename("BIS residential property prices")


def _as_quarterly(series: Series) -> Series:
    """Drop missing values and put a series onto a Q-DEC quarterly index.

    The 6416.0 and 6432.0 tables come back on quarterly PeriodIndexes anchored
    on different months, which ra.splice() will not align.

    Args:
        series: a quarterly series, on a PeriodIndex.

    Returns:
        The series, dropna'd, on a Q-DEC PeriodIndex.

    """
    cleaned = series.dropna()
    index = cleaned.index
    if not isinstance(index, pd.PeriodIndex):
        raise TypeError(f"Expected a PeriodIndex for {series.name}, got {type(index).__name__}")
    cleaned.index = index.asfreq("Q-DEC")
    return cleaned


def _select_quarterly(data: dict[str, DataFrame], meta: DataFrame, selector: dict[str, str]) -> Series:
    """Select one series by data-item description and put it on a Q-DEC index.

    Args:
        data: the tables returned by ra.read_abs_cat.
        meta: the matching metadata.
        selector: a find_abs_id selector, {search_value: column}.

    Returns:
        The selected series, dropna'd, on a Q-DEC PeriodIndex.

    """
    table, sid, _units = ra.find_abs_id(meta, selector, verbose=False)
    return _as_quarterly(data[table][sid])


@cache
def _get_abs_house_price_index() -> tuple[Series, DataFrame]:
    """Splice the long-run mean-dwelling-value series and its splice report.

    Highest priority first (see get_house_price_index); rebase=True chains each
    lower-priority index onto the running dollar level so the segments join
    smoothly. Cached per kernel session.

    Returns:
        The spliced series and the ra.splice() audit report. Not for mutation.

    """
    # 1. current all-dwellings mean price (6432.0), $'000 -> $
    d4, m4 = ra.read_abs_cat("6432.0", single_excel_only="643201", verbose=False)
    mean_val = _select_quarterly(d4, m4, {_MEAN_PRICE_DID: mc.did}) * 1_000
    # 2. + 3. discontinued RPPI (8 capitals) and long-run established-house index
    d6, m6 = ra.read_abs_cat("6416.0", url=_RPPI_URL, verbose=False)
    rppi = _select_quarterly(d6, m6, {"641601": mc.table, _RPPI_DID: mc.did})
    est = _select_quarterly(d6, m6, {"641608": mc.table, _ESTABLISHED_DID: mc.did})
    return ra.splice([mean_val, rppi, est], rebase=True, name="House price index")


@cache
def _get_spliced_house_prices(*, extend_bis: bool) -> tuple[Series, DataFrame]:
    """Splice the nominal house-price level, optionally back to 1970Q1.

    The ABS splice carries the dollar level and starts 1986Q2. With extend_bis,
    the BIS index is spliced underneath it, trimmed to _BIS_OVERLAP_QUARTERS past
    the junction so the rebase is anchored there rather than over four decades.

    Args:
        extend_bis: whether to extend the series back to 1970Q1 with BIS/REIA.

    Returns:
        The spliced level and the ra.splice() audit report. Not for mutation.

    """
    abs_level, report = _get_abs_house_price_index()
    if not extend_bis:
        return abs_level, report

    bis = _get_bis_house_prices()
    junction = abs_level.index[0]
    bis = bis[bis.index <= junction + (_BIS_OVERLAP_QUARTERS - 1)]
    if bis.index[0] >= junction:
        raise ValueError("The BIS series adds no quarters before the ABS series starts")
    return ra.splice([abs_level, bis], rebase=True, name="Mean dwelling price")


def _deflate_to_latest(nominal: Series) -> tuple[Series, str]:
    """Deflate a nominal level by the headline CPI, in latest-quarter dollars.

    The CPI is rebased to its own final value, matching the treatment of the
    deflators elsewhere, so the real series reads in the prices of the latest
    CPI quarter. There is no circularity in deflating house prices by the CPI:
    the CPI covers the purchase price of *new* dwellings and rents, not the
    price of established houses.

    Args:
        nominal: a nominal dollar level, on a quarterly PeriodIndex.

    Returns:
        The real level and its units (naming the base quarter).

    """
    cpi, _units, _stype = get_cpi("headline")
    deflator = cpi / cpi.iloc[-1]
    real = (nominal / deflator).dropna()
    if real.empty:
        raise ValueError("No overlap between the house price series and the CPI")
    return real, f"$ ({cpi.index[-1]} prices)"


def seasonally_adjust(series: Series) -> Series:
    """Return the seasonally adjusted component of a price series.

    Multiplicative, since these series are strictly positive levels, and
    ARIMA-extended so the ends are not smoothed against a truncated window.
    Public because deflating any Original series by a price index leaves a
    seasonal residue that has to be removed before the result is rebased to a
    single period.

    Args:
        series: a quarterly price level or index.

    Returns:
        The seasonally adjusted series, named after its input.

    """
    result = decompose(series, model="multiplicative", arima_extend=True)
    if result is None:
        raise ValueError(f"Seasonal decomposition of {series.name} failed")
    name = series.name if isinstance(series.name, str) else str(series.name)
    return result[FINAL_SEASADJ].dropna().rename(name)


@cache
def _get_house_price_index(*, extend_bis: bool, real: bool, seasonally_adjusted: bool) -> tuple[Series, str, str]:
    """Build the house-price level for one combination of options (cached).

    Returns:
        The level, its units, and its series type. Not for mutation (cached).

    """
    series, _report = _get_spliced_house_prices(extend_bis=extend_bis)
    units, stype = "$", "Original"
    if real:
        series, units = _deflate_to_latest(series)
        series = series.rename("Real mean dwelling price")
    if seasonally_adjusted:
        series, stype = seasonally_adjust(series), "Seasonally Adjusted"
    return series, units, stype


def get_house_price_index(
    *, extend_bis: bool = False, real: bool = False, seasonally_adjusted: bool = False
) -> tuple[Series, str, str]:
    """Return a long-run Australian house-price level, spliced back to 1986Q2.

    Splices three ABS measures (highest priority first), rebased to chain
    smoothly onto the current mean-dwelling-value dollar level:
      1. 6432.0 mean price of residential dwellings (all dwellings, 2011Q3+);
      2. 6416.0 Table 1 RPPI, eight-capitals weighted average (2003Q3-2021Q4);
      3. 6416.0 Table 8 established-house index (1986Q2-2005Q2).

    The discontinued 6416.0 is fetched via a readabs URL override (its Dec-2021
    final-release page). The result carries the mean-value dollar level, so the
    units are dollars. See get_house_price_splice_report() for the splice audit
    (rebase factors and overlap junctions).

    Args:
        extend_bis: splice the BIS index underneath, reaching 1970Q1. The 1986Q2
            junction is a real break in method (a REIA median below, a
            quality-adjusted ABS index above). Where the two overlap they agree
            closely - year-ended growth correlates 0.989, mean absolute
            difference 0.62 percentage points - which is what justifies the join.
        real: deflate by the headline CPI, in latest-CPI-quarter dollars.
        seasonally_adjusted: return the seasonally adjusted component. The
            spliced level is Original, so rebasing it to a single quarter would
            bake that quarter's seasonal factor into the base - pass this
            whenever the series is to be indexed to a particular quarter.

    Returns:
        A (series, units, stype) tuple - the level (a defensive copy), its units
        ("$", or "$ (<quarter> prices)" when real) and its series type.

    """
    series, units, stype = _get_house_price_index(
        extend_bis=extend_bis, real=real, seasonally_adjusted=seasonally_adjusted
    )
    return series.copy(), units, stype


def get_house_price_splice_report(*, extend_bis: bool = False) -> DataFrame:
    """Return the ra.splice() audit report for the house-price index.

    Args:
        extend_bis: see get_house_price_index. With the BIS segment, the report
            covers the final ABS-to-BIS junction; the ABS-internal junctions are
            in the report returned without it.

    Returns:
        The splice report (rebase factors and overlap junctions), a copy.

    """
    _spliced, report = _get_spliced_house_prices(extend_bis=extend_bis)
    return report.copy()


# === CPI (6401.0): reconstructed long-run headline + SA analytical measures
_CPI_QTLY_TABLE = "6401017"
_CPI_QOQ_DID = "Percentage Change from Previous Period ;  All groups CPI ;  Australia ;"
_CPI_INDEX_DID = "Index Numbers ;  All groups CPI ;  Australia ;"
_CPI_APPENDIX = "64010Appendix1a"
_CPI_APPENDIX_DIDS = {
    "headline_sa": "Index Numbers ;  All groups CPI, seasonally adjusted ;  Australia ;",
    "trimmed": "Index Numbers ;  Trimmed Mean ;  Australia ;",
    "weighted": "Index Numbers ;  Weighted Median ;  Australia ;",
}


@cache
def _get_cpi_headline() -> tuple[Series, str, str]:
    """Reconstruct the long-run headline CPI index, back to 1948Q4.

    The published quarterly index is rounded to few significant figures in the
    early years, which puts steps into any growth computed off it. The reported
    quarterly % change is finer, so the index is rebuilt by chaining that change
    into a relative index and rebasing it onto the published All-groups index
    level at the latest common quarter - a smooth, artefact-free index on the
    current ABS reference base. Both series are selected by data-item description
    (Original, quarterly, table 6401017); ABS series IDs are fragile.

    Returns:
        The reconstructed headline CPI index, its units, and series type.

    """
    data, meta = ra.read_abs_cat("6401.0", single_excel_only=_CPI_QTLY_TABLE, verbose=False)
    base = {_CPI_QTLY_TABLE: mc.table, "Original": mc.stype, "Quarter": mc.freq}
    _qt, qoq_id, _qu = ra.find_abs_id(meta, base | {_CPI_QOQ_DID: mc.did, "Percent": mc.unit}, verbose=False)
    _it, idx_id, _iu = ra.find_abs_id(
        meta, base | {_CPI_INDEX_DID: mc.did, "Index Numbers": mc.unit}, verbose=False
    )
    qoq = data[_CPI_QTLY_TABLE][qoq_id].dropna() / 100
    published = data[_CPI_QTLY_TABLE][idx_id].dropna()
    rel = (1 + qoq).cumprod()
    anchor = rel.index.intersection(published.index)[-1]
    index = rel / rel.loc[anchor] * published.loc[anchor]
    return index.rename("Headline CPI (reconstructed)"), "Index Numbers", "Original"


@cache
def _get_cpi_appendix(measure: str) -> tuple[Series, str, str]:
    """Fetch an SA analytical CPI index from 64010Appendix1a (cached).

    Returns:
        The raw SA index, its units, and series type. Not for mutation.

    """
    data, meta = ra.read_abs_cat("6401.0", single_excel_only=_CPI_APPENDIX, verbose=False)
    selector = {
        _CPI_APPENDIX: mc.table,
        _CPI_APPENDIX_DIDS[measure]: mc.did,
        "Seasonally Adjusted": mc.stype,
        "Index Numbers": mc.unit,
    }
    table, sid, units = ra.find_abs_id(meta, selector, verbose=False)
    return data[table][sid].dropna(), units, "Seasonally Adjusted"


def get_cpi(measure: str = "headline") -> tuple[Series, str, str]:
    """Return a quarterly CPI index and its units, by measure.

    Args:
        measure: "headline" (All groups, Original - reconstructed back to 1948Q4
            from the reported quarterly change, artefact-free), "headline_sa"
            (All groups, Seasonally Adjusted), "trimmed" (Trimmed Mean, SA) or
            "weighted" (Weighted Median, SA).

    Returns:
        A (series, units, stype) tuple - the index (a defensive copy), "Index
        Numbers", and its series type ("Original" for headline, else "Seasonally
        Adjusted").

    """
    if measure == "headline":
        series, units, stype = _get_cpi_headline()
    elif measure in _CPI_APPENDIX_DIDS:
        series, units, stype = _get_cpi_appendix(measure)
    else:
        choices = ("headline", *_CPI_APPENDIX_DIDS)
        raise ValueError(f"Unknown CPI measure: {measure!r}. Choose from {choices}.")
    return series.copy(), units, stype
