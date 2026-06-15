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
  index); get_house_price_splice_report() returns the splice audit.
"""

# === imports
from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import pandas as pd
import readabs as ra
from readabs import metacol as mc

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
_RPPI_DID = "Residential Property Price Index ;  Weighted average of eight capital cities ;"
_ESTABLISHED_DID = "Price Index of Established Homes ;  Weighted Average of 8 Capital Cities ;"


@cache
def _get_house_price_index() -> tuple[Series, DataFrame]:
    """Splice the long-run mean-dwelling-value series and its splice report.

    Highest priority first (see get_house_price_index); rebase=True chains each
    lower-priority index onto the running dollar level so the segments join
    smoothly. Cached per kernel session.

    Returns:
        The spliced series and the ra.splice() audit report. Not for mutation.

    """
    # 1. current all-dwellings mean price (6432.0), $'000 -> $
    data, meta = ra.read_abs_cat("6432.0", single_excel_only="643201", verbose=False)
    table, sid, _u = ra.find_abs_id(
        meta,
        {"Mean price of residential dwellings ;  Australia ;": mc.did},
        verbose=False,
    )
    mean_val = data[table][sid].dropna() * 1_000
    mean_val.index = mean_val.index.asfreq("Q-DEC")
    # 2. + 3. discontinued RPPI (8 capitals) and long-run established-house index
    d6, m6 = ra.read_abs_cat("6416.0", url=_RPPI_URL, verbose=False)
    rt, rppi_id, _ru = ra.find_abs_id(m6, {"641601": mc.table, _RPPI_DID: mc.did}, verbose=False)
    rppi = d6[rt][rppi_id].dropna()  # RPPI, weighted average of 8 capitals
    rppi.index = rppi.index.asfreq("Q-DEC")
    et, est_id, _eu = ra.find_abs_id(m6, {"641608": mc.table, _ESTABLISHED_DID: mc.did}, verbose=False)
    est = d6[et][est_id].dropna()  # established homes, weighted average of 8 capitals
    est.index = est.index.asfreq("Q-DEC")
    return ra.splice([mean_val, rppi, est], rebase=True, name="House price index")


def get_house_price_index() -> tuple[Series, str, str]:
    """Return a long-run Australian house-price level, spliced back to 1986Q2.

    Splices three ABS measures (highest priority first), rebased to chain
    smoothly onto the current mean-dwelling-value dollar level:
      1. 6432.0 mean price of residential dwellings (all dwellings, 2011Q3+);
      2. 6416.0 Table 1 RPPI, eight-capitals weighted average (2003Q3-2021Q4);
      3. 6416.0 Table 8 established-house index (1986Q2-2005Q2).

    The discontinued 6416.0 is fetched via a readabs URL override (its Dec-2021
    final-release page). The result carries the mean-value dollar level, so the
    units are "$" and the series type is "Original". See
    get_house_price_splice_report() for the splice audit (rebase factors and
    overlap junctions).

    Returns:
        A (series, units, stype) tuple - the spliced level (a defensive copy),
        "$" and "Original".

    """
    spliced, _report = _get_house_price_index()
    return spliced.copy(), "$", "Original"


def get_house_price_splice_report() -> DataFrame:
    """Return the ra.splice() audit report for the house-price index.

    Returns:
        The splice report (rebase factors and overlap junctions), a copy.

    """
    _spliced, report = _get_house_price_index()
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
    _qt, qoq_id, _qu = ra.find_abs_id(
        meta, base | {_CPI_QOQ_DID: mc.did, "Percent": mc.unit}, verbose=False
    )
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
