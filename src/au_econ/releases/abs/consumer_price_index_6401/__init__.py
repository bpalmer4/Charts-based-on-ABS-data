"""Consumer Price Index, Australia (6401.0): the CPI measures, and later the expenditure classes.

Charts from 6401.0 data only. Anything combining the CPI with other releases (the
discontinued monthly indicator 6484.0, PPI, WPI, deflators) is in a topic module.
"""

# --- dependencies
from au_econ.releases.abs.consumer_price_index_6401 import classes, measures
from au_econ.sources.abs import AbsRelease, fetch_release

# --- module contract
RELEASE = ("6401", "cpi")
TOPICS = ("prices",)
TITLE = "Consumer Price Index"

# --- constants
CATALOGUE = "6401.0"


# --- data
def fetch() -> AbsRelease:
    """Fetch the whole release once; every chart function receives it."""
    return fetch_release(CATALOGUE)


# --- table of contents, in run order
CHARTS = (*measures.CHARTS, *classes.CHARTS)
