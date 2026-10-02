"""Wage Price Index, Australia (6345.0): wage growth by sector, and wages against prices.

The WPI follows price changes in a fixed "basket" of jobs.
"""

# --- dependencies
import re

import pandas as pd
import readabs as ra
from mgplot import bar_plot_finalise, line_plot_finalise, multi_start, series_growth_plot_finalise
from readabs import metacol as mc

from au_econ.charting.footers import SERIES_TYPE_NOTES
from au_econ.charting.windows import QUARTERS_PER_YEAR, quarterly_plot_times
from au_econ.series.prices import get_cpi, get_living_cost_index
from au_econ.sources.abs import AbsRelease, fetch_release

# --- module contract
RELEASE = ("6345", "wpi")
TOPICS = ("wages", "prices")
TITLE = "Wage Price Index"

# --- constants
CATALOGUE = "6345.0"
TABLE = "634501"
CPI_CATALOGUE = "6401.0"
LCI_CATALOGUE = "6467.0"
ORIGINAL = "Original"
SEASONALLY_ADJUSTED = "Seasonally Adjusted"
STYPE_ABBREVIATIONS = {ORIGINAL: "Orig", SEASONALLY_ADJUSTED: "Seas Adj"}
ANNUAL_CHANGE_DID = "Percentage Change from Corresponding Quarter of Previous Year"
INDEX_DID = "Index"
WPI_MEASURE = "Total hourly rates of pay excluding bonuses ;  Australia"
INDEX_DID_PREFIX = "Quarterly Index ;  Total hourly rates of pay excluding bonuses ;  Australia ;  "
WPI_SA_SELECTOR = {
    "Quarterly Index": mc.did,
    "Private and Public": mc.did,
    "All industries": mc.did,
    SEASONALLY_ADJUSTED: mc.stype,
}
SECTORS = ("Private All industries", "Public All industries")  # public v private chart
COMPARISON_BASE = pd.Period("2019Q4", freq="Q")
QUARTERS_PER_PERIOD = 1
GROWTH_TITLE = "Annual Wage Price Growth"
PER_CENT_PA = "Per cent per annum"
SA_LFOOTER = f"Australia. {SERIES_TYPE_NOTES[SEASONALLY_ADJUSTED]}"
WPI_LFOOTER = f"Australia. Total hourly rates of pay excluding bonuses. {SERIES_TYPE_NOTES[ORIGINAL]}"
SMALL_LEGEND_UPPER_LEFT = {"loc": "upper left", "fontsize": "small"}
SMALL_LEGEND_BEST = {"loc": "best", "fontsize": "small"}


# --- data
def fetch() -> AbsRelease:
    """Fetch the release once; every chart function receives it."""
    return fetch_release(CATALOGUE)


# --- helpers
def _select(release: AbsRelease, stype: str, did_contains: str) -> list[tuple[str, str]]:
    """Return (series ID, description) for the table's series of one type and description."""
    meta = release.meta
    selected = meta[
        (meta[mc.table] == TABLE) & (meta[mc.stype] == stype) & meta[mc.did].str.contains(did_contains)
    ]
    return list(zip(selected[mc.id], selected[mc.did], strict=True))


def _annual_growth(release: AbsRelease) -> dict[str, pd.Series]:
    """Return annual WPI growth (Original) by sector, keyed by a short sector label."""
    growth: dict[str, pd.Series] = {}
    for series_id, did in _select(release, ORIGINAL, ANNUAL_CHANGE_DID):
        label = (
            did.replace(ANNUAL_CHANGE_DID, "")
            .replace(WPI_MEASURE, "")
            .replace(";", "")
            .replace("Private and Public", "All sectors")
            .strip()
        )
        growth[re.sub(" +", " ", label)] = release.data[TABLE][series_id].dropna()
    return growth


def _wpi_sa_index(release: AbsRelease) -> pd.Series:
    """Return the seasonally adjusted WPI index, all sectors and industries."""
    _table, series_id, _units = ra.find_abs_id(release.meta, {TABLE: mc.table} | WPI_SA_SELECTOR)
    return release.data[TABLE][series_id]


# --- charts
def annual_growth(release: AbsRelease) -> None:
    """Annual wage price growth for all sectors, private and public; full and recent."""
    for label, series in _annual_growth(release).items():
        multi_start(
            series,
            function=line_plot_finalise,
            starts=quarterly_plot_times,
            title=f"{GROWTH_TITLE}: {label}",
            ylabel=PER_CENT_PA,
            rfooter=release.source,
            lfooter=WPI_LFOOTER,
        )


def public_private(release: AbsRelease) -> None:
    """Private against public sector annual wage price growth; full and recent."""
    growth = _annual_growth(release)
    multi_start(
        pd.DataFrame({sector: growth[sector] for sector in SECTORS}),
        function=line_plot_finalise,
        starts=quarterly_plot_times,
        title=GROWTH_TITLE,
        ylabel=PER_CENT_PA,
        rfooter=release.source,
        lfooter=WPI_LFOOTER,
    )


def quarterly_growth(release: AbsRelease) -> None:
    """WPI growth bars by sector, original and seasonally adjusted; recent only."""
    for stype in (ORIGINAL, SEASONALLY_ADJUSTED):
        for series_id, did in _select(release, stype, INDEX_DID):
            label = did.replace(INDEX_DID_PREFIX, "").replace(" ;", "").replace("  ", " ")
            series_growth_plot_finalise(
                release.data[TABLE][series_id],
                plot_from=quarterly_plot_times[1],
                tag="recent",
                title=f"WPI Growth: {label} ({STYPE_ABBREVIATIONS[stype]})",
                rfooter=release.source,
                lfooter=f"Australia. WPI = Wage Price Index. {SERIES_TYPE_NOTES[stype]}",
                y0=True,
            )


def cpi_comparison(release: AbsRelease) -> None:
    """CPI and WPI since December 2019: rebased index, quarterly and annual growth."""
    cpi, _units, _stype = get_cpi("headline_sa")
    combined = pd.DataFrame({"CPI": cpi, "WPI": _wpi_sa_index(release)}).dropna()
    since = combined[combined.index >= COMPARISON_BASE]
    if since.empty or since.index[0] != COMPARISON_BASE:
        raise ValueError(f"CPI and WPI: no common observation for {COMPARISON_BASE}")
    rfooter = f"{release.source}, {CPI_CATALOGUE}"

    line_plot_finalise(
        since / since.iloc[0] * 100,
        title="CPI and WPI since December 2019",
        ylabel=f"Index ({COMPARISON_BASE} = 100)",
        rfooter=rfooter,
        lfooter=SA_LFOOTER,
        legend=SMALL_LEGEND_UPPER_LEFT,
    )
    bar_plot_finalise(
        (since.pct_change(QUARTERS_PER_PERIOD, fill_method=None) * 100).iloc[QUARTERS_PER_PERIOD:],
        title="CPI and WPI: Quarter-on-Quarter Growth",
        ylabel="Per cent",
        rfooter=rfooter,
        lfooter=SA_LFOOTER,
        legend=SMALL_LEGEND_BEST,
        y0=True,
    )
    line_plot_finalise(
        (since.pct_change(QUARTERS_PER_YEAR, fill_method=None) * 100).iloc[QUARTERS_PER_YEAR:],
        title="CPI and WPI: Through-the-Year Growth",
        ylabel=PER_CENT_PA,
        rfooter=rfooter,
        lfooter=SA_LFOOTER,
        legend=SMALL_LEGEND_BEST,
        y0=True,
    )


def real_wages(release: AbsRelease) -> None:
    """WPI deflated by the CPI and by the Living Cost Index, rebased; full and recent."""
    wpi = _wpi_sa_index(release)
    cpi, _cpi_units, _cpi_stype = get_cpi("headline_sa")
    lci, _lci_units, _lci_stype = get_living_cost_index()
    deflators = (
        (
            "CPI",
            cpi,
            CPI_CATALOGUE,
            f"Australia. {SERIES_TYPE_NOTES[SEASONALLY_ADJUSTED]} Real WPI = WPI / CPI, rebased.",
            "Real Wage Price Index",
        ),
        (
            "LCI",
            lci,
            LCI_CATALOGUE,
            (
                "Australia. LCI = Living Cost Index (Employee households). Real WPI = WPI / LCI, rebased. "
                "WPI seasonally adjusted, LCI original."
            ),
            "Real Wage Price Index: LCI Deflated",
        ),
    )
    for name, deflator, catalogue, lfooter, title in deflators:
        frame = pd.DataFrame({name: deflator, "WPI": wpi}).dropna()
        real = frame["WPI"] / frame[name]
        real = (real / real.iloc[0] * 100).rename(f"Real WPI ({name})")
        multi_start(
            real,
            function=line_plot_finalise,
            starts=quarterly_plot_times,
            title=title,
            ylabel="Index (start = 100)",
            annotate=True,
            rfooter=f"{release.source}, {catalogue}",
            lfooter=lfooter,
        )


# --- table of contents, in run order
CHARTS = (
    (annual_growth, ()),
    (public_private, ()),
    (quarterly_growth, ()),
    (cpi_comparison, ()),
    (real_wages, ()),
)
