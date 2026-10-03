"""Australian Institute of Petroleum (AIP): the weekly terminal gate price (TGP) workbook.

AIP moved to WordPress in August 2026: the upload folder reflects when a file was
uploaded, not the date in its name, so the URL cannot be constructed. The current link
is read off the resources page. Each workbook is cached under its own name, which carries
the data date. If the page cannot be read or has no workbook link (as can happen in a
week when Friday is a public holiday), the newest cached workbook is used, with a
warning; the charts' "Data to" footers show its date.
"""

import io
import re

import pandas as pd
import requests

from au_econ import paths

INDEX_URL = "https://aip.com.au/resources/historical-ulp-and-diesel-tgp-data/"
LINK = re.compile(r"https://\S+?/AIP_TGP_Data_[^\"']+?\.xlsx")
HEADERS = {"User-Agent": "Mozilla/5.0"}
TIMEOUT = 30  # seconds
CACHE_DIR = paths.CACHE_DIR / "AIP"
FILE_PREFIX = "AIP_TGP_Data_"
FILE_DATE_FORMAT = "%d-%b-%Y"  # e.g. AIP_TGP_Data_28-Aug-2026.xlsx


def _current_url() -> str:
    """Find the current TGP workbook link on the AIP resources page."""
    response = requests.get(INDEX_URL, headers=HEADERS, timeout=TIMEOUT)
    response.raise_for_status()
    match = LINK.search(response.text)
    if match is None:
        raise FileNotFoundError(f"No AIP_TGP_Data link found at {INDEX_URL}")
    return match.group(0)


def _latest_cached() -> bytes:
    """Return the newest cached workbook, by the date in its name (not alphabetical)."""
    cached = sorted(
        CACHE_DIR.glob(f"{FILE_PREFIX}*.xlsx"),
        key=lambda path: pd.to_datetime(path.stem.removeprefix(FILE_PREFIX), format=FILE_DATE_FORMAT),
    )
    if not cached:
        raise FileNotFoundError(f"AIP unreachable and no cached data in {CACHE_DIR}")
    print(f"Using cached {cached[-1].name}")
    return cached[-1].read_bytes()


def get_workbook() -> bytes:
    """Return the current AIP TGP workbook, downloading it once; fall back to the newest cached one."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        url = _current_url()
    except (requests.RequestException, FileNotFoundError) as err:
        print(f"WARNING: cannot reach AIP ({err})")
        return _latest_cached()
    file = CACHE_DIR / url.rsplit("/", 1)[-1]
    if file.exists():
        print(f"Using cached: {file.name}")
        return file.read_bytes()
    response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    response.raise_for_status()
    print(f"Downloaded: {file.name}")
    file.write_bytes(response.content)
    return response.content


def parse_sheet(workbook: bytes, sheet: str) -> pd.DataFrame:
    """Return one sheet (e.g. "Petrol TGP") with a daily PeriodIndex and tidied column names."""
    frame = pd.read_excel(io.BytesIO(workbook), sheet_name=sheet, header=0, index_col=0, parse_dates=True)
    frame.columns = frame.columns.str.strip().str.replace("\n", " ")
    frame.index = pd.DatetimeIndex(frame.index).to_period("D")
    return frame
