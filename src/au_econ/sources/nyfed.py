"""New York Fed: the Adrian-Crump-Moench (ACM) term premium decomposition of the US Treasury curve.

Read from the Fed's workbook. The server sends no Last-Modified (a Last-Modified rule would
serve the first download forever), so the workbook is cached for RECENT_MAX_AGE, with the
cached copy as a fallback when a download fails.
"""

import io

import pandas as pd

from au_econ.sources.http_cache import RECENT_MAX_AGE, get_recent

ACM_URL = "https://www.newyorkfed.org/medialibrary/media/research/data_indicators/ACMTermPremium.xls"
DAILY_SHEET = "ACM Daily"  # named: "ACM Monthly" comes first in the workbook


def get_acm_daily() -> pd.DataFrame:
    """Return the ACM daily sheet (e.g. ACMRNY05, ACMRNY10), numeric, on a DatetimeIndex."""
    workbook = get_recent(ACM_URL, None, "nyfed", RECENT_MAX_AGE, fallback=True)
    frame = pd.read_excel(io.BytesIO(workbook), sheet_name=DAILY_SHEET)
    dates = pd.to_datetime(frame["DATE"], format="%d-%b-%Y", errors="coerce")  # anything else is a footer row
    frame = frame.loc[dates.notna()].copy()
    frame.index = pd.DatetimeIndex(dates.loc[dates.notna()])
    frame = frame.drop(columns="DATE").apply(pd.to_numeric, errors="coerce")
    if frame.empty:
        raise ValueError(f"The NY Fed {DAILY_SHEET} sheet holds no dated rows")
    return frame
