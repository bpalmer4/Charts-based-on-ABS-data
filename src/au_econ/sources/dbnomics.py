"""DBnomics: series by PROVIDER/DATASET/SERIES path, through the public API.

Fetched directly, not through http_cache: DBnomics sends no Last-Modified header, so a
cached copy would never be refreshed.
"""

import time

import pandas as pd
import requests

SERIES_URL = "https://api.db.nomics.world/v22/series"
TIMEOUT = 60  # seconds
RETRIES = 4
SERVER_ERROR = 500
SERVER_BACKOFF = 1.0  # seconds, times the attempt number
NETWORK_BACKOFF = 1.5  # seconds, times the attempt number
MISSING = ("", "NA", "na", ".")


def _to_float(value: object) -> float:
    """Coerce a DBnomics observation to float; "NA", None and blanks become NaN."""
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str) and value.strip() not in MISSING:
        try:
            return float(value)
        except ValueError:
            return float("nan")
    return float("nan")


def _get(path: str) -> requests.Response:
    """GET one series with its observations, retrying on timeouts and server errors."""
    for attempt in range(1, RETRIES + 1):
        last = attempt == RETRIES
        try:
            response = requests.get(f"{SERIES_URL}/{path}", params={"observations": "1"}, timeout=TIMEOUT)
        except requests.Timeout, requests.ConnectionError:
            if last:
                raise
            time.sleep(NETWORK_BACKOFF * attempt)
            continue
        if response.status_code >= SERVER_ERROR and not last:
            time.sleep(SERVER_BACKOFF * attempt)
            continue
        response.raise_for_status()
        return response
    raise RuntimeError(f"DBnomics {path}: no response")


def get_series(path: str) -> pd.Series:
    """Return a DBnomics series with a PeriodIndex at its native frequency, missing values dropped."""
    docs = _get(path).json().get("series", {}).get("docs", [])
    if not docs:
        raise ValueError(f"DBnomics {path}: no data")
    periods, values = docs[0].get("period", []), docs[0].get("value", [])
    if not periods:
        raise ValueError(f"DBnomics {path}: empty series")
    freq = pd.Period(periods[0]).freqstr
    index = pd.PeriodIndex([pd.Period(period, freq=freq) for period in periods])
    return pd.Series([_to_float(value) for value in values], index=index, dtype="float64", name=path).dropna()
