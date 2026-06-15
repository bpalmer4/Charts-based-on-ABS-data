"""GDP convenience getter built on the ABS National Accounts (5206.0).

Split out of abs_helper.py so the National-Accounts volume measures live with
their own concept rather than in the catch-all helper. The fetch is cached for
the life of the kernel session (readabs handles the on-disk, between-run cache);
the public getter returns a defensive copy so the cached series cannot be
mutated by a caller.
"""

# === imports
from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import readabs as ra

if TYPE_CHECKING:
    from pandas import Series


# === GDP from the National Accounts key aggregates
@cache
def _get_gdp(gdp_type: str, seasonal: str) -> tuple[Series, str]:
    """Fetch and select the GDP series (cached per kernel session).

    Args:
        gdp_type: see get_gdp.
        seasonal: see get_gdp.

    Returns:
        The raw selected series and its units. Callers must not mutate the
        returned series in place - it is the shared cached object.

    """
    # validate the arguments
    did_cvm = "Gross domestic product: Chain volume measures ;"
    did_cp = "Gross domestic product: Current prices ;"
    gdp_types = {
        # gdp-type: data-item-description in the ABS data
        "CP": did_cp,
        "Current price": did_cp,
        "Current prices": did_cp,
        "CVM": did_cvm,
        "Volumetric": did_cvm,
    }
    seasonals = {
        "SA": "Seasonally Adjusted",
        "S": "Seasonally Adjusted",
        "T": "Trend",
        "O": "Original",
    }
    if gdp_type not in gdp_types:
        raise ValueError(f"Invalid GDP type: {gdp_type}")
    if seasonal not in seasonals:
        raise ValueError(f"Invalid seasonal adjustment type: {seasonal}")

    # get the series
    cat = "5206.0"
    seo = "5206001_Key_Aggregates"
    gdp_data, gdp_meta = ra.read_abs_cat(cat, single_excel_only=seo, verbose=False)
    selector = {
        gdp_types[gdp_type]: ra.metacol.did,
        seasonals[seasonal]: ra.metacol.stype,
    }
    table, series_id, units = ra.find_abs_id(gdp_meta, selector, verbose=False)
    gdp = gdp_data[table][series_id]

    return gdp, units


def get_gdp(gdp_type: str = "CP", seasonal: str = "SA") -> tuple[Series, str]:
    """Return the ABS GDP Series (from the key aggregates table).

    Args:
        gdp_type: The type of series to return - 'CP' (current prices) or
            'CVM' (chain volume measures).
        seasonal: The seasonal adjustment type - 'SA' (seasonally adjusted),
            'T' (trend) or 'O' (original).

    Returns:
        A tuple of the selected GDP series (a defensive copy of the cached
        fetch) and the units of that series.

    """
    gdp, units = _get_gdp(gdp_type, seasonal)
    return gdp.copy(), units
