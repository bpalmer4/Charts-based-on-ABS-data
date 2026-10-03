"""FRED (Federal Reserve Bank of St. Louis): series observations through the API, cached on disk.

FRED selects series by ID only. Its IDs are stable mnemonics, unlike ABS series IDs, so
FRED modules name their series by ID.
"""

import json
from functools import cache

import pandas as pd

from au_econ import paths
from au_econ.sources.http_cache import get_file

OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
KEY_FILE = paths.KEYS_DIR / "fred.api"
MISSING = "."  # FRED's marker for a missing observation


@cache
def _api_key() -> str:
    """Read the API key once per run."""
    return KEY_FILE.read_text().strip()


def get_series(series_id: str, start: str, frequency: str | None = None) -> pd.Series:
    """Return a FRED series from start (YYYY-MM-DD), with a DatetimeIndex and missing values dropped.

    frequency (e.g. "q") asks FRED to aggregate to a lower frequency; None leaves the native one.
    """
    params = {"series_id": series_id, "api_key": _api_key(), "file_type": "json", "observation_start": start}
    if frequency is not None:
        params["frequency"] = frequency
    observations = json.loads(get_file(OBSERVATIONS_URL, params, prefix="fred"))["observations"]
    series = pd.Series(
        [float("nan") if obs["value"] == MISSING else float(obs["value"]) for obs in observations],
        index=pd.to_datetime([obs["date"] for obs in observations]),
        name=series_id,
    ).dropna()
    if series.empty:
        raise ValueError(f"FRED {series_id}: no observations from {start}")
    return series
