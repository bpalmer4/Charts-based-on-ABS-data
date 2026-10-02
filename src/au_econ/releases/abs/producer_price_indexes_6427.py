"""Producer Price Indexes, Australia (6427.0): price growth for final demand and selected industries."""

# --- dependencies
import readabs as ra
from mgplot import multi_start, series_growth_plot_finalise
from readabs import metacol as mc

from au_econ.charting.windows import quarterly_plot_times
from au_econ.sources.abs import AbsRelease, fetch_release

# --- module contract
RELEASE = ("6427", "ppi")
TOPICS = ("prices",)
TITLE = "Producer Price Indexes"

# --- constants
CATALOGUE = "6427.0"
INDEX_DID = "Index Number"
LFOOTER = "Australia. Original series."
# (description search term, table, chart label) for each series
FINAL_DEMAND = ("Final", "642701", "Final Demand (Prices)")
COAL_MINING = ("Coal mining", "6427011", "Input prices to the Coal mining industry")
MANUFACTURING = ("Manufacturing division", "6427012", "Output prices of the Manufacturing industries")
BUILDING_CONSTRUCTION = ("30 Building construction Australia ;", "6427017", "Building construction prices")
ROAD_FREIGHT = ("Road freight transport ", "6427021", "Road freight transport prices")
EMPLOYMENT_SERVICES = ("Employment services", "6427025", "Employment services prices")


# --- data
def fetch() -> AbsRelease:
    """Fetch the release once; every chart function receives it."""
    return fetch_release(CATALOGUE)


# --- helpers
def _growth(release: AbsRelease, series: tuple[str, str, str]) -> None:
    """Plot quarterly and annual growth in one PPI series; full history and recent."""
    search_term, table, label = series
    selector = {table: mc.table, INDEX_DID: mc.did, search_term: mc.did}
    _table, series_id, _units = ra.find_abs_id(release.meta, selector, verbose=False)
    multi_start(
        release.data[table][series_id],
        function=series_growth_plot_finalise,
        starts=quarterly_plot_times,
        title=f"Growth in {label} PPI",
        lfooter=LFOOTER,
        rfooter=release.source,
        y0=True,
    )


# --- charts
def final_demand(release: AbsRelease) -> None:
    """Plot growth in final demand producer prices."""
    _growth(release, FINAL_DEMAND)


def coal_mining(release: AbsRelease) -> None:
    """Plot growth in input prices to the coal mining industry."""
    _growth(release, COAL_MINING)


def manufacturing(release: AbsRelease) -> None:
    """Plot growth in output prices of the manufacturing industries."""
    _growth(release, MANUFACTURING)


def building_construction(release: AbsRelease) -> None:
    """Plot growth in building construction prices."""
    _growth(release, BUILDING_CONSTRUCTION)


def road_freight(release: AbsRelease) -> None:
    """Plot growth in road freight transport prices."""
    _growth(release, ROAD_FREIGHT)


def employment_services(release: AbsRelease) -> None:
    """Plot growth in employment services prices."""
    _growth(release, EMPLOYMENT_SERVICES)


# --- table of contents, in run order
CHARTS = (
    (final_demand, ()),
    (coal_mining, ()),
    (manufacturing, ()),
    (building_construction, ()),
    (road_freight, ()),
    (employment_services, ()),
)
