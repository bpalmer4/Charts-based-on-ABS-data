"""Bank for International Settlements: central bank policy rates (dataflow WS_CBPOL), via the BIS SDMX API.

One request covers every country ("+"-joined keys). The API sends no Last-Modified, so
responses are cached for RECENT_MAX_AGE by file age.
"""

import io
from typing import TYPE_CHECKING

import pandas as pd

from au_econ.sources.http_cache import RECENT_MAX_AGE, get_recent

if TYPE_CHECKING:
    from collections.abc import Iterable

POLICY_RATES_URL = "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_CBPOL/1.0/{frequency}.{areas}"
TIMEOUT = 120  # seconds
COLUMNS = ["REF_AREA", "TIME_PERIOD", "OBS_VALUE"]


def get_policy_rates(frequency: str, areas: Iterable[str], start: str) -> pd.DataFrame:
    """Return policy rates (per cent), one row per country and period: REF_AREA, TIME_PERIOD, OBS_VALUE.

    frequency is "D" (daily) or "M" (monthly); areas are BIS two-letter codes ("XM" is the euro area).
    """
    url = POLICY_RATES_URL.format(frequency=frequency, areas="+".join(areas))
    content = get_recent(url, {"startPeriod": start, "format": "csv"}, "bis", RECENT_MAX_AGE, TIMEOUT)
    rows = pd.read_csv(io.BytesIO(content))
    if rows.empty:
        raise ValueError(f"BIS policy rates {frequency}: no data from {start}")
    return rows[COLUMNS]
