"""Population series wanted by more than one module.

Each getter is cached for the run and returns (series, units), with the series a copy, so
a caller changing it cannot corrupt the cache.
"""

from functools import cache
from typing import TYPE_CHECKING

import readabs as ra
from readabs import metacol as mc

if TYPE_CHECKING:
    from pandas import Series

# --- Estimated Resident Population (3101.0): quarterly, persons
ERP_CATALOGUE = "3101.0"
ERP_TABLE = "310104"
ERP_SELECTOR = {
    ";  Australia ;": mc.did,  # the bare name would also match every state's "Australian"
    "Estimated Resident Population ;  Persons ;  ": mc.did,
}


@cache
def _erp() -> tuple[Series, str]:
    """Fetch national ERP (cached; not for mutation)."""
    data, meta = ra.read_abs_cat(ERP_CATALOGUE, single_excel_only=ERP_TABLE, verbose=False)
    _, series_id, units = ra.find_abs_id(meta, ERP_SELECTOR, verbose=False)
    series = data[ERP_TABLE][series_id].dropna()
    if series.empty:
        raise ValueError(f"ABS {ERP_CATALOGUE} returned no national ERP values")
    return series, units


def get_erp(project_quarters: int = 0) -> tuple[Series, str]:
    """National Estimated Resident Population, optionally extended at its latest quarterly growth rate.

    ERP is published about six months after its reference quarter, so a few quarters of
    projection let it divide series that are more current.
    """
    series, units = _erp()
    series = series.copy()
    rate = series.iloc[-1] / series.iloc[-2]
    last = series.index[-1]
    for step in range(1, project_quarters + 1):
        series[last + step] = series[last + step - 1] * rate
    return series, units
