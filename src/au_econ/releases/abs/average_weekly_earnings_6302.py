"""Average Weekly Earnings, Australia (6302.0): full-time adult ordinary time earnings."""

# --- dependencies
import pandas as pd
import readabs as ra
from mgplot import line_plot_finalise, seastrend_plot_finalise
from readabs import metacol as mc

from au_econ.charting.footers import SERIES_TYPE_NOTES
from au_econ.sources.abs import AbsRelease, fetch_release

# --- module contract
RELEASE = ("6302", "awe")
TOPICS = ("wages",)
TITLE = "Average Weekly Earnings"

# --- constants
CATALOGUE = "6302.0"
SEASONALLY_ADJUSTED = "Seasonally Adjusted"
TABLES = {SEASONALLY_ADJUSTED: "6302002", "Trend": "6302001"}  # series type: table
UNITS = "$"
EARNINGS_DID = ("Earnings", "Persons", "Full Time", "Adult", "Ordinary time earnings")
ANNUAL_PERIODS = 2  # bi-annual data: two observations a year
CHART_TITLE = "Average Weekly Full-time Adult Ordinary Time Earnings"
LFOOTER = "Australia.  Bi-annual data. "


# --- data
def fetch() -> AbsRelease:
    """Fetch the release once; every chart function receives it."""
    return fetch_release(CATALOGUE)


# --- helpers
def _earnings(release: AbsRelease) -> tuple[pd.DataFrame, str]:
    """Return seasonally adjusted and trend earnings side by side, recalibrated, with units."""
    columns: dict[str, pd.Series] = {}
    for stype, table in TABLES.items():
        selector = dict.fromkeys(EARNINGS_DID, mc.did) | {stype: mc.stype, UNITS: mc.unit, table: mc.table}
        _table, series_id, _units = ra.find_abs_id(release.meta, selector, verbose=False)
        columns[stype] = release.data[table][series_id]
    return ra.recalibrate(pd.DataFrame(columns), UNITS)


# --- charts
def earnings(release: AbsRelease) -> None:
    """Earnings level: seasonally adjusted and trend."""
    frame, units = _earnings(release)
    seastrend_plot_finalise(frame, title=CHART_TITLE, ylabel=units, lfooter=LFOOTER, rfooter=release.source)


def earnings_growth(release: AbsRelease) -> None:
    """Annual growth in seasonally adjusted earnings."""
    frame, _units = _earnings(release)
    growth = frame[SEASONALLY_ADJUSTED].pct_change(ANNUAL_PERIODS) * 100
    line_plot_finalise(
        growth,
        title=f"Growth Rate: {CHART_TITLE}",
        ylabel="Annual Growth Rate (%)",
        lfooter=f"{LFOOTER}{SERIES_TYPE_NOTES[SEASONALLY_ADJUSTED]}",
        rfooter=release.source,
        annotate=True,
    )


# --- table of contents, in run order
CHARTS = (
    (earnings, ()),
    (earnings_growth, ()),
)
