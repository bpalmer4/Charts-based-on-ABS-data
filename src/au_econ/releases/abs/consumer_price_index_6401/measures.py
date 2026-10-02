"""CPI measures from 6401.0: headline and underlying measures, their growth, core and trend."""

# --- dependencies
from typing import TYPE_CHECKING

import pandas as pd
import readabs as ra
from mgplot import calc_growth, growth_plot_finalise, line_plot_finalise, multi_start
from readabs import metacol as mc

from au_econ.analysis.henderson import hma
from au_econ.charting.footers import SERIES_TYPE_NOTES
from au_econ.charting.targets import ANNUAL_CPI_TARGET_RANGE, MONTHLY_CPI_TARGET, QUARTERLY_CPI_TARGET
from au_econ.charting.windows import (
    MONTHS_PER_YEAR,
    QUARTERS_PER_YEAR,
    monthly_plot_times,
    quarterly_plot_times,
)

if TYPE_CHECKING:
    from au_econ.sources.abs import AbsRelease

# --- constants
APPENDIX = "64010Appendix1a"  # quarterly analytical measures, seasonally adjusted
MONTHLY = "640106"
MONTHLY_GROUPS = "640103"
QUARTERLY_ALL_GROUPS = "6401017"
ORIGINAL, SEASONALLY_ADJUSTED = "Original", "Seasonally Adjusted"
INDEX_NUMBERS = "Index Numbers"
YEAR_ON_YEAR_TERMS = ("Percentage", "revious", "ear")  # the published year-on-year series; ABS case varies
QUARTERLY_CHANGE_DID = "Percentage Change from Previous Period ;  All groups CPI ;  Australia ;"
HENDERSON_TERMS = 13
LONG_HEADLINE = "Qrtly Headline CPI (Orig)"

core_starts = 0, -(8 * MONTHS_PER_YEAR + 1), -(2 * MONTHS_PER_YEAR + 1)
trend_plot_from = -(5 * MONTHS_PER_YEAR)

LFOOTER = "Australia. Orig = Original series. SA = Seasonally adjusted series. "
APPENDIX_NOTE = "Quarterly CPI measures are from Appendix 1a. "
ANNUAL_PCT = "CPI (annual % change)"
CORE_COLOURS = ("darkblue", "darkorange", "darkblue", "darkorange")
CORE_STYLES = ("-", "-", "--", "--")
TREND_COLOURS = ("brown", "cornflowerblue", "darkorange")
TREND_WIDTHS = (0.75, 2.5)  # thin monthly lines, thick trend lines

# label: (table, description search term, series type)
MEASURES = {
    "Qrtly Headline CPI (SA)": (APPENDIX, "All groups CPI, seasonally adjusted", SEASONALLY_ADJUSTED),
    "Qrtly Trimmed Mean CPI (SA)": (APPENDIX, "Trimmed Mean", SEASONALLY_ADJUSTED),
    "Qrtly Weighted Median CPI (SA)": (APPENDIX, "Weighted Median", SEASONALLY_ADJUSTED),
    "Monthly Headline CPI (Orig)": (MONTHLY, "All groups CPI ;", ORIGINAL),
    "Monthly Headline CPI (SA)": (MONTHLY, "All groups CPI, seasonally adjusted", SEASONALLY_ADJUSTED),
    "Monthly Trimmed Mean CPI (SA)": (MONTHLY, "Trimmed Mean", SEASONALLY_ADJUSTED),
    "Monthly Weighted Median CPI (SA)": (MONTHLY, "Weighted Median", SEASONALLY_ADJUSTED),
}
CORE = {
    "Qrtly Trimmed Mean CPI (SA)": (APPENDIX, "Trimmed Mean", SEASONALLY_ADJUSTED),
    "Qrtly Weighted Median CPI (SA)": (APPENDIX, "Weighted Median", SEASONALLY_ADJUSTED),
    "Monthly Trimmed Mean CPI (SA)": (MONTHLY, "Trimmed Mean", SEASONALLY_ADJUSTED),
    "Monthly Weighted Median CPI (SA)": (MONTHLY, "Weighted Median", SEASONALLY_ADJUSTED),
}
TREND = {
    "Headline (SA)": (MONTHLY, "All groups CPI, seasonally adjusted", SEASONALLY_ADJUSTED),
    "Trimmed Mean (SA)": (MONTHLY, "Trimmed Mean", SEASONALLY_ADJUSTED),
    "Weighted Median (SA)": (MONTHLY, "Weighted Median", SEASONALLY_ADJUSTED),
}
# (labels left out, file tag, title) for each trend chart
TREND_CHARTS = (
    ((), "", "CPI Measures: Annualised Monthly Growth with Henderson Trend"),
    (("Headline (SA)",), "underlying-only", "Core CPI Measures: Annualised Monthly Growth with Henderson Trend"),
)
GROWTH = {
    "Qrtly Headline CPI (SA, pre-2025 methodology)": (
        APPENDIX,
        "All groups CPI, seasonally adjusted",
        SEASONALLY_ADJUSTED,
    ),
    "Qrtly Trimmed Mean CPI (SA, pre-2025 methodology)": (APPENDIX, "Trimmed Mean", SEASONALLY_ADJUSTED),
    "Qrtly Weighted Median CPI (SA, pre-2025 methodology)": (APPENDIX, "Weighted Median", SEASONALLY_ADJUSTED),
    "Monthly Headline CPI (SA)": (MONTHLY, "All groups CPI, seasonally adjusted", SEASONALLY_ADJUSTED),
    "Monthly Trimmed Mean CPI (SA)": (MONTHLY, "Trimmed Mean", SEASONALLY_ADJUSTED),
    "Monthly Weighted Median CPI (SA)": (MONTHLY, "Weighted Median", SEASONALLY_ADJUSTED),
    "Monthly CPI Headline (Orig)": (MONTHLY, "All groups CPI ;", ORIGINAL),
    "Monthly CPI Goods (Orig)": (MONTHLY, "All groups, goods component", ORIGINAL),
    "Monthly CPI Services (Orig)": (MONTHLY, "All groups, services component", ORIGINAL),
    "Monthly CPI Tradables (Orig)": (MONTHLY, "Tradables ;", ORIGINAL),
    "Monthly CPI Non-tradables (Orig)": (MONTHLY, "Non-tradables ;", ORIGINAL),
    "Monthly CPI Discretionary (Orig)": (MONTHLY, ";  Discretionary ;", ORIGINAL),
    "Monthly CPI Non-Discretionary (Orig)": (MONTHLY, "Non-Discretionary", ORIGINAL),
    "Monthly CPI Discretionary excl. tobacco (Orig)": (MONTHLY, "Discretionary excluding tobacco", ORIGINAL),
    "Monthly CPI excl. volatile items (Orig)": (MONTHLY, "excluding 'volatile items' ;", ORIGINAL),
    "Monthly CPI excl. volatile items and holiday travel (Orig)": (
        MONTHLY,
        "volatile items' and holiday travel",
        ORIGINAL,
    ),
    "Monthly CPI excl. food and energy (Orig)": (MONTHLY, "excluding food and energy", ORIGINAL),
    "Monthly CPI Market goods excl. volatile items (Orig)": (MONTHLY, "- Total ;", ORIGINAL),
    "Monthly CPI Market goods excl. volatile items - Goods (Orig)": (MONTHLY, "- Goods ;", ORIGINAL),
    "Monthly CPI Market goods excl. volatile items - Services (Orig)": (MONTHLY, "- Services ;", ORIGINAL),
    "Monthly CPI excl. Alcohol and tobacco (Orig)": (MONTHLY, "excluding Alcohol and tobacco", ORIGINAL),
    "Monthly CPI excl. Clothing and footwear (Orig)": (MONTHLY, "excluding Clothing and footwear", ORIGINAL),
    "Monthly CPI excl. Communication (Orig)": (MONTHLY, "excluding Communication", ORIGINAL),
    "Monthly CPI excl. Education (Orig)": (MONTHLY, "excluding Education", ORIGINAL),
    "Monthly CPI excl. Food and non-alcoholic beverages (Orig)": (
        MONTHLY,
        "excluding Food and non-alcoholic beverages",
        ORIGINAL,
    ),
    "Monthly CPI excl. Furnishings (Orig)": (MONTHLY, "excluding Furnishings", ORIGINAL),
    "Monthly CPI excl. Health (Orig)": (MONTHLY, "excluding Health ;", ORIGINAL),
    "Monthly CPI excl. Housing (Orig)": (MONTHLY, "excluding Housing ;", ORIGINAL),
    "Monthly CPI excl. Housing and Insurance (Orig)": (MONTHLY, "excluding Housing and Insurance", ORIGINAL),
    "Monthly CPI excl. Insurance and financial services (Orig)": (
        MONTHLY,
        "excluding Insurance and financial services",
        ORIGINAL,
    ),
    "Monthly CPI excl. Medical and hospital services (Orig)": (
        MONTHLY,
        "excluding Medical and hospital services",
        ORIGINAL,
    ),
    "Monthly CPI excl. Recreation and culture (Orig)": (MONTHLY, "excluding Recreation and culture", ORIGINAL),
    "Monthly CPI excl. Transport (Orig)": (MONTHLY, "excluding Transport", ORIGINAL),
    "Monthly Automotive fuel (Orig)": (MONTHLY_GROUPS, "Automotive fuel ;", ORIGINAL),
    "Monthly Electricity (Orig)": (MONTHLY_GROUPS, "Electricity ;", ORIGINAL),
}


# --- helpers
def _select(release: AbsRelease, spec: tuple[str, str, str], *, unit: str = "", year_on_year: bool) -> pd.Series:
    """Return one series from a table, by description and series type, trailing gaps trimmed.

    With year_on_year, the ABS's published year-on-year growth series is chosen.
    """
    table, did, stype = spec
    selector = {did: mc.did, stype: mc.stype}
    if unit:
        selector[unit] = mc.unit
    if year_on_year:
        selector |= dict.fromkeys(YEAR_ON_YEAR_TERMS, mc.did)
    meta = release.meta[release.meta[mc.table] == table]
    _table, series_id, _units = ra.find_abs_id(meta, selector, verbose=False)
    series = release.data[table][series_id]
    last_valid = series.last_valid_index()
    return series if last_valid is None else series.loc[:last_valid]


def _monthly(series: pd.Series) -> pd.Series:
    """Return a monthly series unchanged, or a quarterly one interpolated to months."""
    freq = series.index.freqstr[0] if isinstance(series.index, pd.PeriodIndex) else ""
    if freq == "M":
        return series
    if freq == "Q":
        return ra.qtly_to_monthly(series)
    raise ValueError(f"Unexpected frequency {freq!r} in {series.name!r}")


def _annual_growth(release: AbsRelease, specs: dict[str, tuple[str, str, str]]) -> dict[str, pd.Series]:
    """Return published year-on-year growth for each labelled measure, on a monthly axis."""
    return {label: _monthly(_select(release, spec, year_on_year=True)) for label, spec in specs.items()}


def _long_headline(release: AbsRelease) -> pd.Series:
    """Return quarterly headline year-on-year growth back to 1949, compounded from quarterly change.

    Compounding the published quarterly change avoids the rounding steps that the early,
    coarsely rounded index numbers would put into growth computed from the index.
    """
    selector = {QUARTERLY_ALL_GROUPS: mc.table, QUARTERLY_CHANGE_DID: mc.did, ORIGINAL: mc.stype}
    _table, series_id, _units = ra.find_abs_id(release.meta, selector, exact_match=True, verbose=False)
    quarterly = release.data[QUARTERLY_ALL_GROUPS][series_id].dropna() / 100
    annual = ((1 + quarterly).rolling(QUARTERS_PER_YEAR).apply(lambda x: x.prod()) - 1) * 100
    return annual.dropna().rename(LONG_HEADLINE)


# --- charts
def cpi_measures(release: AbsRelease) -> None:
    """Annual growth in the CPI measures, monthly and quarterly; full history and recent."""
    long_headline = {LONG_HEADLINE: ra.qtly_to_monthly(_long_headline(release))}
    multi_start(
        pd.DataFrame(long_headline | _annual_growth(release, MEASURES)),
        function=line_plot_finalise,
        starts=monthly_plot_times,
        title="Australian Consumer Price Index (CPI) Measures",
        ylabel=ANNUAL_PCT,
        axhspan=ANNUAL_CPI_TARGET_RANGE,
        legend={"loc": "best", "ncol": 2, "fontsize": "x-small"},
        lfooter=LFOOTER + APPENDIX_NOTE,
        rfooter=release.source,
        y0=True,
    )


def growth(release: AbsRelease) -> None:
    """Periodic and annual growth for each CPI measure and analytical series; recent."""
    for label, spec in GROWTH.items():
        series = _select(release, spec, unit=INDEX_NUMBERS, year_on_year=False)
        monthly = isinstance(series.index, pd.PeriodIndex) and series.index.freqstr[0] == "M"
        growth_plot_finalise(
            calc_growth(series),
            plot_from=(monthly_plot_times if monthly else quarterly_plot_times)[1],
            title=f"Growth: {label}",
            ylabel="Per cent",
            lfooter=f"Australia. {SERIES_TYPE_NOTES[spec[2]]}",
            axhline=MONTHLY_CPI_TARGET if monthly else QUARTERLY_CPI_TARGET,
            axhspan=ANNUAL_CPI_TARGET_RANGE,
            rfooter=release.source,
            y0=True,
        )


def core_measures(release: AbsRelease) -> None:
    """Trimmed mean against weighted median, monthly and quarterly; three windows."""
    multi_start(
        pd.DataFrame(_annual_growth(release, CORE)),
        function=line_plot_finalise,
        starts=core_starts,
        title="Australian Inflation Measures: Trimmed Mean vs Weighted Median",
        ylabel=ANNUAL_PCT,
        axhspan=ANNUAL_CPI_TARGET_RANGE,
        legend={"loc": "best", "ncol": 2, "fontsize": "x-small"},
        lfooter=LFOOTER + APPENDIX_NOTE,
        color=list(CORE_COLOURS),
        style=list(CORE_STYLES),
        rfooter=release.source,
        y0=True,
    )


def trend_annualised(release: AbsRelease) -> None:
    """Annualised monthly growth with a Henderson trend: all measures, then core only."""
    colour_of = dict(zip(TREND, TREND_COLOURS, strict=True))
    for left_out, tag, title in TREND_CHARTS:
        monthly: dict[str, pd.Series] = {}
        trends: dict[str, pd.Series] = {}
        for label, spec in TREND.items():
            if label in left_out:
                continue
            index = _select(release, spec, unit=INDEX_NUMBERS, year_on_year=False)
            annualised = ra.annualise_rates(index / index.shift(1) - 1, periods_per_year=MONTHS_PER_YEAR)
            monthly[label] = annualised
            trends[f"{label} trend"] = hma(annualised.dropna(), HENDERSON_TERMS)
        count = len(monthly)
        thin, thick = TREND_WIDTHS
        line_plot_finalise(
            pd.DataFrame(monthly | trends),
            plot_from=trend_plot_from,
            title=title,
            ylabel="Per cent per annum",
            color=[colour_of[label] for label in monthly] * 2,
            width=[thin] * count + [thick] * count,
            style=["-"] * (2 * count),
            annotate=[False] * count + [True] * count,  # label the trend lines only
            rounding=1,
            axhspan=ANNUAL_CPI_TARGET_RANGE,
            legend={"loc": "best", "ncol": 2, "fontsize": "x-small"},
            lfooter="Australia. Seasonally adjusted. Trend: 13-term Henderson moving average. ",
            rfooter=release.source,
            y0=True,
            tag=tag,
        )


# --- table of contents, in run order
CHARTS = (
    (cpi_measures, ()),
    (growth, ()),
    (core_measures, ()),
    (trend_annualised, ()),
)
