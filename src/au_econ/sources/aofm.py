"""AOFM: the daily term premium decomposition of the nominal Treasury Bond curve.

Fitted yield (FY), term premium (TP) and risk-neutral yield (RNY), tenors 1 to 10
years, from July 1992. The file is served from a path holding the node's creation date, so a site rebuild moves
it; if the known URL fails, the link is re-found on the data hub page. Downloads go by
Last-Modified, with the cached copy as a fallback.
"""

import io
import re
from functools import cache

import pandas as pd
import requests

from au_econ.sources.http_cache import HttpError, get_file

WORKBOOK_URL = "https://www.aofm.gov.au/sites/default/files/2025-06-06/term%20premium.xlsx"
HUB_URL = "https://www.aofm.gov.au/data-hub"
SITE = "https://www.aofm.gov.au"
HUB_LINK = re.compile(r'href="([^"]*term[%20_ ]*premium[^"]*\.xlsx)"', flags=re.IGNORECASE)
METHOD_SHEETS = {"bc": "TermPremiumBC", "ols": "TermPremiumOLS"}  # bias-corrected, or plain ACM
HEADER_ROW = 1  # row 0 is the method's title banner


@cache
def _workbook() -> bytes:
    """Return the workbook, re-finding its link on the data hub if the known URL fails."""
    try:
        return get_file(WORKBOOK_URL, prefix="aofm", fallback=True)
    except (requests.RequestException, HttpError) as error:
        print(f"AOFM workbook not available at {WORKBOOK_URL} ({type(error).__name__})")
    page = get_file(HUB_URL, prefix="aofm", fallback=True).decode("utf-8", errors="replace")
    links = HUB_LINK.findall(page)
    if not links:
        raise ValueError(f"No term premium workbook at {WORKBOOK_URL} or linked from {HUB_URL}; check by hand")
    url = links[0] if links[0].startswith("http") else SITE + links[0]
    return get_file(url, prefix="aofm", fallback=True)


def get_term_premium(method: str) -> pd.DataFrame:
    """Return one decomposition sheet ("bc" or "ols"), daily, numeric, on a DatetimeIndex."""
    if method not in METHOD_SHEETS:
        raise ValueError(f"Unknown AOFM method {method!r}; expected one of {', '.join(sorted(METHOD_SHEETS))}")
    raw = pd.read_excel(io.BytesIO(_workbook()), sheet_name=METHOD_SHEETS[method], header=HEADER_ROW)
    frame = raw.rename(columns={raw.columns[0]: "DATE"})
    dates = pd.to_datetime(frame["DATE"], errors="coerce")
    frame = frame.loc[dates.notna()].copy()
    frame.index = pd.DatetimeIndex(dates.loc[dates.notna()])
    frame = frame.drop(columns="DATE").apply(pd.to_numeric, errors="coerce")
    if frame.empty:
        raise ValueError(f"The AOFM {METHOD_SHEETS[method]} sheet holds no dated rows")
    return frame
