"""Japan Ministry of Finance: the daily JGB yield curve (history file plus the current month).

The files send Last-Modified, so they go through http_cache.get_file, with the cached
copy as a fallback when a download fails. The "English" CSVs still carry Shift-JIS bytes.
"""

import io
from functools import cache

import pandas as pd

from au_econ.sources.http_cache import get_file

BASE_URL = "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/"
HISTORY = "historical/jgbcme_all.csv"
CURRENT = "jgbcme.csv"
ENCODING = "cp932"


def _curve_file(path: str) -> pd.DataFrame:
    """Read one MOF curve CSV into a dated frame of per-cent yields (row 0 is a title; "-" is missing)."""
    text = get_file(BASE_URL + path, prefix="mof", fallback=True).decode(ENCODING)
    frame = pd.read_csv(io.StringIO(text), skiprows=1, na_values=["-"])
    frame["Date"] = pd.to_datetime(frame["Date"], format="%Y/%m/%d", errors="coerce")
    frame = frame.dropna(subset=["Date"]).set_index("Date")
    return frame.apply(pd.to_numeric, errors="coerce")


@cache
def _curve() -> pd.DataFrame:
    """Return the whole daily curve: history extended by the current month (cached; not for mutation)."""
    history, current = _curve_file(HISTORY), _curve_file(CURRENT)
    curve = pd.concat([history, current[~current.index.isin(history.index)]]).sort_index()
    if curve.empty:
        raise ValueError("MOF returned an empty yield curve")
    curve.index = pd.PeriodIndex(curve.index, freq="D")
    return curve


def get_jgb_yield(tenor: str) -> pd.Series:
    """Return one tenor (e.g. "10Y") of the daily JGB curve, per cent."""
    curve = _curve()
    if tenor not in curve.columns:
        raise ValueError(f"MOF curve has no {tenor} column: {list(curve.columns)}")
    series = curve[tenor].dropna()
    if series.empty:
        raise ValueError(f"MOF returned no {tenor} observations")
    return series.copy()
