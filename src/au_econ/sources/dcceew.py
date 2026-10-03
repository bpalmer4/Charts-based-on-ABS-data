"""DCCEEW: Australian Petroleum Statistics (energy.gov.au) and the quarterly greenhouse gas inventory.

Petroleum: each release is a new workbook linked from that year's publication page; the
newest is chosen by the release month in its file name, then downloaded through
http_cache. The new year's page does not exist until around March, so the previous
year's is also tried.

Greenhouse gas inventory: each quarter's update has its own page, named for the quarter.
The department's list of updates can lag a release, so the pages are tried directly,
newest quarter first, and the one workbook each links to is downloaded through http_cache.
"""

import io
import re
from urllib.parse import unquote

import pandas as pd
import requests

from au_econ.sources.http_cache import get_file

BASE_URL = "https://www.energy.gov.au"
PAGE_TEMPLATE = BASE_URL + "/publications/australian-petroleum-statistics-{year}"
WORKBOOK_LINK = re.compile(r'href="([^"]+\.xlsx)"')
MONTHS = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)
TIMEOUT = 60  # seconds
OK = 200
MISSING = "n.a."

INVENTORY_SITE = "https://www.dcceew.gov.au"
INVENTORY_PAGE = (
    INVENTORY_SITE + "/climate-change/publications/" + "national-greenhouse-gas-inventory-quarterly-update-{}"
)
INVENTORY_LOOKBACK = 6  # quarters tried; an update follows its quarter by about five months
INVENTORY_HEADER_ROWS = 5  # title rows above each figure sheet's column headings
SEPTEMBER = 9  # some September pages are named "sept"


def _file_name(url: str) -> str:
    """Return the decoded file name at the end of a URL."""
    return unquote(url.rsplit("/", maxsplit=1)[-1])


def _release_month(url: str) -> tuple[int, int]:
    """Return the (year, month) of the release named in a workbook URL; (0, 0) if there is none.

    File names are inconsistently cased and URL-encoded, separate words with spaces,
    underscores or hyphens, and some carry a "(re-issued)" or "_0" suffix, so the
    month name and year are matched directly, across any of those separators.
    """
    name = _file_name(url).lower()
    match = re.search(rf"({'|'.join(MONTHS)})[\s_-]+(\d{{4}})", name)
    return (0, 0) if match is None else (int(match.group(2)), MONTHS.index(match.group(1)) + 1)


def _workbook_url() -> str:
    """Return the URL of the most recent data-extract workbook."""
    year = pd.Timestamp.today().year
    for try_year in (year, year - 1):
        page = requests.get(PAGE_TEMPLATE.format(year=try_year), timeout=TIMEOUT)
        if page.status_code != OK:
            continue
        links = WORKBOOK_LINK.findall(page.text)
        for link in links:
            if _release_month(link) == (0, 0):  # file names are hand-made: a new pattern would be skipped
                print(f"WARNING: cannot date petroleum workbook {_file_name(link)}; newer data may be missed")
        dated = [link for link in links if _release_month(link) != (0, 0)]
        if dated:
            newest = max(dated, key=_release_month)
            return newest if newest.startswith("http") else BASE_URL + newest
    raise ValueError("No petroleum statistics workbook found for the current or previous year")


def get_workbook() -> bytes:
    """Return the most recent Australian Petroleum Statistics workbook."""
    url = _workbook_url()
    print(f"Petroleum statistics workbook: {_file_name(url)}")
    return get_file(url, prefix="dcceew")


def monthly_sheet(workbook: bytes, sheet: str) -> pd.DataFrame:
    """Return a monthly sheet with a monthly PeriodIndex, "n.a." as missing."""
    frame = pd.read_excel(io.BytesIO(workbook), sheet_name=sheet).dropna(how="all", axis=0).set_index("Month")
    frame.index = pd.PeriodIndex(frame.index, freq="M")
    return frame.replace(MISSING, pd.NA)


def fuel_price_sheet(workbook: bytes) -> pd.DataFrame:
    """Return the quarterly Australian fuel prices sheet with a quarterly PeriodIndex."""
    frame = pd.read_excel(io.BytesIO(workbook), sheet_name="Australian fuel prices").dropna(how="all", axis=0)
    frame.index = pd.PeriodIndex(frame["Year"].astype(str) + frame["Quarter"], freq="Q")
    return frame.drop(columns=["Year", "Quarter"])


def _inventory_slugs(quarter: pd.Period) -> list[str]:
    """Page-name endings for a quarter's update, e.g. "march-2026"; September is sometimes "sept"."""
    month = quarter.end_time.month
    names = [MONTHS[month - 1]] + (["sept"] if month == SEPTEMBER else [])
    return [f"{name}-{quarter.year}" for name in names]


def get_inventory_workbook() -> tuple[bytes, pd.Period]:
    """Return the most recent greenhouse gas inventory workbook and the quarter it reports."""
    current = pd.Timestamp.today().to_period("Q")
    for quarter in (current - lag for lag in range(INVENTORY_LOOKBACK)):
        for slug in _inventory_slugs(quarter):
            page = requests.get(INVENTORY_PAGE.format(slug), timeout=TIMEOUT)
            if page.status_code != OK:
                continue
            links = sorted(set(WORKBOOK_LINK.findall(page.text)))
            if len(links) != 1:
                raise ValueError(f"Greenhouse gas inventory {slug}: expected one workbook link, found {links}")
            url = links[0] if links[0].startswith("http") else INVENTORY_SITE + links[0]
            print(f"Greenhouse gas inventory workbook: {_file_name(url)}")
            return get_file(url, prefix="dcceew"), quarter
    raise ValueError(f"No greenhouse gas inventory update found in the last {INVENTORY_LOOKBACK} quarters")


def inventory_sheet(workbook: bytes, sheet: str) -> pd.DataFrame:
    """Return a quarterly figure sheet (e.g. "Figure 1") with a quarterly PeriodIndex."""
    frame = pd.read_excel(
        io.BytesIO(workbook), sheet_name=sheet, index_col=0, skiprows=INVENTORY_HEADER_ROWS
    ).dropna(how="all", axis=0)
    frame.index = pd.PeriodIndex(frame.index, freq="Q")
    return frame
