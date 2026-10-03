"""GDP from the ABS National Accounts key aggregates (5206.0), wanted by more than one module.

The getter is cached for the run and returns (series, units), with the series a copy, so a
caller changing it cannot corrupt the cache.
"""

from functools import cache
from typing import TYPE_CHECKING

import readabs as ra
from readabs import metacol as mc

if TYPE_CHECKING:
    from pandas import Series

GDP_CATALOGUE = "5206.0"
GDP_TABLE = "5206001_Key_Aggregates"
GDP_MEASURES = {
    "CP": "Gross domestic product: Current prices ;",
    "CVM": "Gross domestic product: Chain volume measures ;",
}
SERIES_TYPES = {"SA": "Seasonally Adjusted", "T": "Trend", "O": "Original"}


@cache
def _gdp(measure: str, series_type: str) -> tuple[Series, str]:
    """Fetch one GDP series (cached; not for mutation)."""
    if measure not in GDP_MEASURES:
        raise ValueError(f"Unknown GDP measure {measure!r}: choose from {sorted(GDP_MEASURES)}")
    if series_type not in SERIES_TYPES:
        raise ValueError(f"Unknown series type {series_type!r}: choose from {sorted(SERIES_TYPES)}")
    data, meta = ra.read_abs_cat(GDP_CATALOGUE, single_excel_only=GDP_TABLE, verbose=False)
    selector = {GDP_MEASURES[measure]: mc.did, SERIES_TYPES[series_type]: mc.stype}
    table, series_id, units = ra.find_abs_id(meta, selector, verbose=False)
    series = data[table][series_id]
    if series.dropna().empty:
        raise ValueError(f"ABS {GDP_CATALOGUE} returned no {measure} {series_type} GDP values")
    return series, units


def get_gdp(measure: str = "CP", series_type: str = "SA") -> tuple[Series, str]:
    """Return quarterly GDP and its units: measure "CP" (current prices) or "CVM"; series type "SA", "T" or "O"."""
    series, units = _gdp(measure, series_type)
    return series.copy(), units
