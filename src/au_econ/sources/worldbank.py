"""World Bank: Commodity Markets ("Pink Sheet") monthly prices, and World Development Indicators.

Pink Sheet: the workbook's URL changes with each monthly release, so the current link is
read off the Commodity Markets page; the download goes through http_cache.

Indicators: the JSON API, cached for RECENT_MAX_AGE by file age. Its Last-Modified is the
time of the response, not of the data.
"""

import io
import json
import re
from typing import TYPE_CHECKING

import pandas as pd
import requests

from au_econ.sources.http_cache import RECENT_MAX_AGE, get_file, get_recent

if TYPE_CHECKING:
    from collections.abc import Iterable

LANDING_PAGE = "https://www.worldbank.org/en/research/commodity-markets"
LINK = re.compile(r"https://thedocs\.worldbank\.org/en/doc/[^\"']+/CMO-Historical-Data-Monthly\.xlsx")
HEADERS = {"User-Agent": "Mozilla/5.0"}
TIMEOUT = 30  # seconds
SHEET = "Monthly Prices"
MISSING = ["N/A", "missing", "-", "…", "..."]
LABEL_ROW, UNIT_ROW, FIRST_DATA_ROW = 4, 5, 7
INDICATOR_URL = "https://api.worldbank.org/v2/country/{countries}/indicator/{indicator}"
PER_PAGE = 1000
RESPONSE_PARTS = 2  # a data page is [page metadata, rows]


def _current_url() -> str:
    """Find the current monthly workbook link on the Commodity Markets page."""
    response = requests.get(LANDING_PAGE, headers=HEADERS, timeout=TIMEOUT)
    response.raise_for_status()
    match = LINK.search(response.text)
    if match is None:
        raise ValueError(f"Could not find CMO-Historical-Data-Monthly.xlsx URL on {LANDING_PAGE}")
    return match.group(0)


def get_commodity_prices() -> tuple[pd.DataFrame, pd.Series]:
    """Return monthly prices (one column per commodity label) and each commodity's units (US$ ...)."""
    url = _current_url()
    print(f"Data URL: {url}")
    workbook = get_file(url, prefix="worldbank")
    sheet = pd.read_excel(io.BytesIO(workbook), sheet_name=SHEET, header=None, na_values=MISSING, index_col=0)
    labels = sheet.iloc[LABEL_ROW]
    units = sheet.iloc[UNIT_ROW].str.replace("$", "US$").str.replace("(", "").str.replace(")", "")
    units.index = labels
    sheet.columns = labels
    prices = sheet.iloc[FIRST_DATA_ROW:].dropna(axis=0, how="all")
    prices.index = pd.PeriodIndex(prices.index.str.replace("M", "-"), freq="M")
    if prices.empty:
        raise ValueError(f"World Bank {SHEET}: no prices")
    print("latest commodity data:", prices.index[-1])
    return prices, units


def get_indicator(indicator: str, countries: Iterable[str], start: int, end: int) -> pd.DataFrame:
    """Return an annual World Development Indicator: years by country (the API's country names).

    countries are ISO3 codes (or an aggregate such as "WLD"); null values are left out.
    """
    url = INDICATOR_URL.format(countries=";".join(countries), indicator=indicator)
    records = []
    page = 1
    while True:
        params = {"format": "json", "date": f"{start}:{end}", "per_page": str(PER_PAGE), "page": str(page)}
        response = json.loads(get_recent(url, params, "worldbank", RECENT_MAX_AGE))
        if len(response) < RESPONSE_PARTS or response[1] is None:
            break
        records += [
            {"country": row["country"]["value"], "year": int(row["date"]), "value": float(row["value"])}
            for row in response[1]
            if row["value"] is not None
        ]
        if page >= response[0]["pages"]:
            break
        page += 1
    if not records:
        raise ValueError(f"World Bank {indicator}: no data for {start}-{end}")
    rows = pd.DataFrame(records)
    if rows.duplicated(["country", "year"]).any():  # pivot_table would average them
        raise ValueError(f"World Bank {indicator}: more than one value for a country and year")
    table = rows.pivot_table(index="year", columns="country", values="value")
    table.index = pd.PeriodIndex(table.index.astype(str), freq="Y")
    return table.sort_index().dropna(how="all", axis=0).dropna(how="all", axis=1)
