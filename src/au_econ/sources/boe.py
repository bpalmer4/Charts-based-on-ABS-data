"""Bank of England: the daily fitted nominal gilt curve (spot), from its yield-curve archives.

The BoE refuses a request without a browser User-Agent. The history archive is about
39 MB and changes only as years roll over, so it is fetched by Last-Modified; the
current-month archive is cached for RECENT_MAX_AGE. Both fall back to the cached copy
when a download fails.
"""

import io
import zipfile
from functools import cache

import pandas as pd

from au_econ.sources.http_cache import BROWSER_HEADERS, RECENT_MAX_AGE, get_file, get_recent

BASE_URL = "https://www.bankofengland.co.uk/-/media/boe/files/statistics/yield-curves/"
HISTORY = "glcnominalddata.zip"
CURRENT = "latest-yield-curve-data.zip"
TIMEOUT = 300  # seconds: the history archive is large
MATURITY_ROW = 3  # the spot sheet's header row holds the maturities, in years


def _spot_curve(archive_bytes: bytes) -> pd.DataFrame:
    """Read the nominal spot curve out of every workbook in a BoE archive.

    The archives hold real and inflation curves too; the spot sheet was renamed from
    "4. nominal spot curve" to "4. spot curve" from 2005, so it is matched on its
    suffix. The current workbook carries a literal "Refresh" row where a date should
    be, so the dates and the yields are both coerced.
    """
    archive = zipfile.ZipFile(io.BytesIO(archive_bytes))
    frames = []
    for name in archive.namelist():
        if "Nominal" not in name:
            continue
        book = pd.ExcelFile(io.BytesIO(archive.read(name)))
        sheets = [
            sheet
            for sheet in book.sheet_names
            if isinstance(sheet, str) and sheet.endswith("spot curve") and "short end" not in sheet
        ]
        if not sheets:
            raise ValueError(f"No nominal spot curve sheet in {name}")
        frame = book.parse(sheets[0], header=MATURITY_ROW)
        frame = frame.rename(columns={frame.columns[0]: "Date"})
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce", format="mixed")
        frame = frame.dropna(subset=["Date"]).set_index("Date")
        frames.append(frame.apply(pd.to_numeric, errors="coerce"))
    if not frames:
        raise ValueError("BoE archive held no nominal curve workbook")
    return pd.concat(frames).sort_index()


@cache
def _curve() -> pd.DataFrame:
    """Return the daily nominal spot curve: history plus this month (cached; not for mutation)."""
    history = _spot_curve(
        get_file(BASE_URL + HISTORY, prefix="boe", timeout=TIMEOUT, headers=BROWSER_HEADERS, fallback=True)
    )
    current = _spot_curve(
        get_recent(
            BASE_URL + CURRENT, None, "boe", RECENT_MAX_AGE, TIMEOUT, headers=BROWSER_HEADERS, fallback=True
        )
    )
    curve = pd.concat([history, current[~current.index.isin(history.index)]]).sort_index()
    curve.index = pd.PeriodIndex(curve.index, freq="D")
    return curve


def get_gilt_yield(maturity: float) -> pd.Series:
    """Return the nominal spot yield at one maturity (years), daily, per cent."""
    curve = _curve()
    if maturity not in curve.columns:
        raise ValueError(f"BoE curve has no {maturity}-year maturity")
    series = curve[maturity].dropna()
    if series.empty:
        raise ValueError(f"BoE returned no {maturity}-year observations")
    return series.copy()
