"""Deutsche Bundesbank: time series from its statistics API, as CSV.

The API sends no Last-Modified, so downloads are cached for RECENT_MAX_AGE by file age.
It throttles bursts of requests, so a failed download falls back to the cached copy.
The CSV puts a variable number of metadata rows ahead of the data, and marks a
non-trading day with a bare full stop.
"""

import re

import pandas as pd

from au_econ.sources.http_cache import BROWSER_HEADERS, RECENT_MAX_AGE, get_recent

BASE_URL = "https://api.statistiken.bundesbank.de/rest/download/BBSIS/"
DATA_ROW = re.compile(r"^\d{4}-\d{2}-\d{2},")  # data rows start with an ISO date
MISSING = (".", "")


def get_series(key: str) -> pd.Series:
    """Return a daily Bundesbank series (e.g. a Bund yield) with a daily PeriodIndex."""
    content = get_recent(
        BASE_URL + key,
        {"format": "csv", "lang": "en"},
        "bundesbank",
        RECENT_MAX_AGE,
        headers=BROWSER_HEADERS,
        fallback=True,
    )
    rows = (line.split(",") for line in content.decode("utf-8-sig").splitlines() if DATA_ROW.match(line))
    observations = {pd.Period(row[0], freq="D"): float(row[1]) for row in rows if row[1] not in MISSING}
    if not observations:
        raise ValueError(f"Bundesbank returned no observations for {key}")
    return pd.Series(observations).sort_index()
