"""Price indexes wanted by more than one module: CPI measures and the Living Cost Index.

Each getter is cached for the run and returns (series, units, series type), with the
series a copy, so a caller changing it cannot corrupt the cache.
"""

from functools import cache
from typing import TYPE_CHECKING

import readabs as ra
from readabs import metacol as mc

if TYPE_CHECKING:
    from pandas import Series

# --- CPI (6401.0): seasonally adjusted measures from the analytical appendix table
CPI_CATALOGUE = "6401.0"
CPI_APPENDIX_TABLE = "64010Appendix1a"
CPI_MEASURES = {
    "headline_sa": "Index Numbers ;  All groups CPI, seasonally adjusted ;  Australia ;",
    "trimmed": "Index Numbers ;  Trimmed Mean ;  Australia ;",
    "weighted": "Index Numbers ;  Weighted Median ;  Australia ;",
}
SEASONALLY_ADJUSTED = "Seasonally Adjusted"
INDEX_NUMBERS = "Index Numbers"

# --- Living Cost Index (6467.0): employee households
LCI_CATALOGUE = "6467.0"
LCI_TABLE = "646701"
LCI_SELECTOR = {"Index Numbers": mc.did, "Employee households": mc.did, "All groups": mc.did}


@cache
def _cpi(measure: str) -> tuple[Series, str, str]:
    """Fetch one seasonally adjusted CPI index (cached; not for mutation)."""
    data, meta = ra.read_abs_cat(CPI_CATALOGUE, single_excel_only=CPI_APPENDIX_TABLE, verbose=False)
    selector = {
        CPI_APPENDIX_TABLE: mc.table,
        CPI_MEASURES[measure]: mc.did,
        SEASONALLY_ADJUSTED: mc.stype,
        INDEX_NUMBERS: mc.unit,
    }
    table, series_id, units = ra.find_abs_id(meta, selector, verbose=False)
    series = data[table][series_id].dropna()
    if series.empty:
        raise ValueError(f"CPI {measure}: no data in {CPI_APPENDIX_TABLE}")
    return series, units, SEASONALLY_ADJUSTED


def get_cpi(measure: str) -> tuple[Series, str, str]:
    """Return a quarterly CPI index: "headline_sa", "trimmed" or "weighted" (all SA)."""
    if measure not in CPI_MEASURES:
        raise ValueError(f"Unknown CPI measure {measure!r}; choose from {tuple(CPI_MEASURES)}")
    series, units, stype = _cpi(measure)
    return series.copy(), units, stype


@cache
def _living_cost_index() -> tuple[Series, str, str]:
    """Fetch the employee households Living Cost Index (cached; not for mutation)."""
    data, meta = ra.read_abs_cat(LCI_CATALOGUE, get_zip=False, get_excel=True)
    table, series_id, units = ra.find_abs_id(meta, {LCI_TABLE: mc.table} | LCI_SELECTOR)
    series = data[table][series_id]
    if series.dropna().empty:
        raise ValueError(f"LCI: no data in {LCI_TABLE}")
    stype = str(meta.loc[meta[mc.id] == series_id, mc.stype].iloc[0])
    return series, units, stype


def get_living_cost_index() -> tuple[Series, str, str]:
    """Return the quarterly Living Cost Index for employee households (All groups)."""
    series, units, stype = _living_cost_index()
    return series.copy(), units, stype
