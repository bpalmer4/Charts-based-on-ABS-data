"""Department of Home Affairs: temporary visa holders in Australia (BP0019), via data.gov.au.

Each quarterly release is a new workbook, found through the data.gov.au catalogue (CKAN)
API; the download goes through http_cache.
"""

import io

import pandas as pd
import requests

from au_econ.sources.http_cache import get_file

PACKAGE_URL = "https://data.gov.au/data/api/3/action/package_show"
PACKAGE_ID = "temporary-entrants-visa-holders"
RESOURCE_MARK = "bp0019"  # in the workbook's URL
HEADERS = {"Accept": "application/json", "User-Agent": "Mozilla/5.0"}
TIMEOUT = 30  # seconds
SHEET = "Visa Holders"
DATE_ROW, FIRST_DATA_ROW = 9, 10  # dates across the date row; one visa category per row below
TOTAL = "Grand Total"


def _current_url() -> str:
    """Find the current BP0019 workbook in the data.gov.au catalogue."""
    response = requests.get(PACKAGE_URL, params={"id": PACKAGE_ID}, headers=HEADERS, timeout=TIMEOUT)
    response.raise_for_status()
    for resource in response.json()["result"]["resources"]:
        if resource.get("format", "").upper() == "XLSX" and RESOURCE_MARK in resource.get("url", "").lower():
            return str(resource["url"])
    raise RuntimeError("No BP0019 XLSX resource found in the data.gov.au dataset")


def get_visa_stock() -> tuple[pd.DataFrame, pd.Timestamp]:
    """Return temporary visa holders (quarters by visa category, no grand total) and the latest date.

    Stocks are at quarter ends, except the latest, which can fall mid-quarter (e.g. 31 August);
    it is shown in its quarter, and its actual date is returned for footers.
    """
    url = _current_url()
    print(f"Visa stock workbook: {url.rsplit('/', 1)[-1]}")
    raw = pd.read_excel(io.BytesIO(get_file(url, prefix="homeaffairs")), sheet_name=SHEET, header=None)
    dates = pd.to_datetime(raw.iloc[DATE_ROW, 1:].tolist())
    categories = raw.iloc[FIRST_DATA_ROW:, 0].astype(str).tolist()
    values = raw.iloc[FIRST_DATA_ROW:, 1:].apply(pd.to_numeric, errors="coerce")
    stock = pd.DataFrame(values.to_numpy(), index=categories, columns=dates).rename_axis("Visa Category")
    stock = stock.drop(index=TOTAL, errors="ignore").T
    if stock.empty:
        raise ValueError("Home Affairs BP0019: no visa stock")
    stock_dates = pd.DatetimeIndex(stock.index)
    stock.index = stock_dates.to_period("Q")
    return stock, stock_dates[-1]
