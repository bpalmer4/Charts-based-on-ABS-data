"""Selected Living Cost Indexes, Australia (6467.0): cost of living by household type.

The household types and interest/insurance components are read from the metadata, so
a type the ABS adds is charted without a code change.
"""

# --- dependencies
import textwrap

import pandas as pd
import readabs as ra
from mgplot import growth_plot_finalise, line_plot_finalise, multi_start
from readabs import metacol as mc

from au_econ.charting.footers import SERIES_TYPE_NOTES
from au_econ.charting.targets import ANNUAL_CPI_TARGET_RANGE, QUARTERLY_CPI_TARGET
from au_econ.charting.windows import quarterly_plot_times
from au_econ.sources.abs import AbsRelease, fetch_release

# --- module contract
RELEASE = ("6467", "lci")
TOPICS = ("prices",)
TITLE = "Living Cost Indexes"

# --- constants
CATALOGUE = "6467.0"
HOUSEHOLDS_TABLE = "646701"
COMPONENTS_TABLE = "646703"
QUARTERLY_DID = "Percentage Change from Previous Period"
ANNUAL_DID = "Percentage Change from Corresponding Quarter of Previous Year"
HOUSEHOLD_PART, COMPONENT_PART = 1, 2  # positions in the ";"-separated description
TITLE_WIDTH = 60
LFOOTER = f"Australia. {SERIES_TYPE_NOTES['Original']}"
PER_CENT = "Per cent"


# --- data
def fetch() -> AbsRelease:
    """Fetch the release once (the ABS publishes no complete zip file for 6467.0)."""
    return fetch_release(CATALOGUE, get_zip=False, get_excel=True)


# --- helpers
def _description_parts(release: AbsRelease, table: str, part: int) -> list[str]:
    """Return one ";"-separated part of the table's annual-change descriptions, in order."""
    meta = release.meta
    descriptions = meta[meta[mc.did].str.contains(ANNUAL_DID) & (meta[mc.table] == table)][mc.did]
    return list(dict.fromkeys(did.split(";")[part].strip() for did in descriptions))


def _household_types(release: AbsRelease) -> list[str]:
    """Return the household types, e.g. "Employee households"."""
    return _description_parts(release, HOUSEHOLDS_TABLE, HOUSEHOLD_PART)


def _series(release: AbsRelease, table: str, *terms: str) -> pd.Series:
    """Return the one series in a table whose description contains all the terms."""
    selector = {table: mc.table} | dict.fromkeys(terms, mc.did)
    _table, series_id, _units = ra.find_abs_id(release.meta, selector, verbose=False)
    return release.data[table][series_id]


# --- charts
def households(release: AbsRelease) -> None:
    """Annual and quarterly LCI growth for each household type; full and recent."""
    for household in _household_types(release):
        growth = pd.DataFrame(
            {
                "Annual": _series(release, HOUSEHOLDS_TABLE, household, ANNUAL_DID),
                "Quarterly": _series(release, HOUSEHOLDS_TABLE, household, QUARTERLY_DID),
            }
        )
        multi_start(
            growth,
            starts=quarterly_plot_times,
            function=growth_plot_finalise,
            title=textwrap.fill(f"Living Cost Index: {household}", TITLE_WIDTH),
            axhspan=ANNUAL_CPI_TARGET_RANGE,
            axhline=QUARTERLY_CPI_TARGET,
            legend={"fontsize": 8, "loc": "best"},
            rfooter=release.source,
            lfooter=LFOOTER,
            y0=True,
        )


def household_comparison(release: AbsRelease) -> None:
    """Annual LCI growth for every household type on one chart."""
    annual = {
        household: _series(release, HOUSEHOLDS_TABLE, household, ANNUAL_DID)
        for household in _household_types(release)
    }
    line_plot_finalise(
        pd.DataFrame(annual),
        title="Annual Growth in Living Cost Indexes",
        ylabel=PER_CENT,
        axhspan=ANNUAL_CPI_TARGET_RANGE,
        legend={"fontsize": 8, "loc": "best", "ncol": 2},
        rfooter=release.source,
        lfooter=LFOOTER,
    )


def components(release: AbsRelease) -> None:
    """Annual growth in each interest and insurance component, by household type."""
    for component in _description_parts(release, COMPONENTS_TABLE, COMPONENT_PART):
        annual = {
            household: _series(release, COMPONENTS_TABLE, household, component, ANNUAL_DID)
            for household in _household_types(release)
        }
        line_plot_finalise(
            pd.DataFrame(annual),
            title=f"LCI Annual Growth: {component}",
            ylabel=PER_CENT,
            axhspan=ANNUAL_CPI_TARGET_RANGE,
            legend={"fontsize": "xx-small", "ncol": 2},
            rfooter=release.source,
            lfooter=LFOOTER,
        )


# --- table of contents, in run order
CHARTS = (
    (households, ()),
    (household_comparison, ()),
    (components, ()),
)
