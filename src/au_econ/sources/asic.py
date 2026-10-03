"""ASIC: corporate insolvency statistics workbooks, from asic.gov.au.

Workbook links carry a random media token, so each is found by its file name on its
publication page, then downloaded through http_cache. The current workbook's name gives
its publication date, so each release is a new file. The 2022 workbooks hold the history
before the current series; they no longer change, so they are cached for good. When a
page cannot be reached, the newest cached copy of the workbook is used, with a warning.
"""

import re
from typing import TYPE_CHECKING

import pandas as pd
import requests

from au_econ import paths
from au_econ.sources.http_cache import BROWSER_HEADERS, get_file, get_recent

if TYPE_CHECKING:
    from pathlib import Path

PREFIX = "asic"
TIMEOUT = 60  # seconds
CURRENT_PAGE = (
    "https://asic.gov.au/regulatory-resources/find-a-document/statistics/"
    "insolvency-statistics/insolvency-statistics-current/"
)
HISTORY_PAGE = (
    "https://asic.gov.au/about-asic/corporate-publications/statistics/"
    "insolvency-statistics/insolvency-statistics-up-to-31-july-2022"
)
CURRENT_STEM = "asic-insolvency-statistics-series-1-and-series-2-published-"
HISTORY_SERIES_1 = "asic-insolvency-statistics-series-1-published-8-september-2022.xlsx"
HISTORY_SERIES_1A = "asic-insolvency-statistics-series-1a-published-8-september-2022.xlsx"
LINK = re.compile(r'href="(https://download\.asic\.gov\.au/[^"]+\.xlsx)"')
PUBLISHED = re.compile(r"published-(\d{1,2}-[a-z]+-\d{4})\.xlsx$")


def _published(name: str) -> pd.Timestamp:
    """Return the publication date in a workbook's file name; the earliest date if it has none."""
    match = PUBLISHED.search(name)
    return pd.Timestamp.min if match is None else pd.to_datetime(match.group(1).replace("-", " "), dayfirst=True)


def _newest_cached(name_start: str, error: Exception) -> bytes:
    """Return the newest cached workbook whose file name starts name_start, saying so."""
    cached: list[Path] = [
        file
        for file in paths.CACHE_DIR.glob(f"{PREFIX}--*")
        if file.name.split("--", maxsplit=2)[-1].startswith(name_start)
    ]
    if not cached:
        raise RuntimeError(f"ASIC page unreachable and no cached {name_start} workbook") from error
    newest = max(cached, key=lambda file: _published(file.name))
    print(f"WARNING: could not check the ASIC page ({type(error).__name__}); using cached {newest.name}")
    return newest.read_bytes()


def _workbook_url(page: str, name_start: str) -> str:
    """Return the one workbook link on page whose file name starts name_start."""
    response = requests.get(page, headers=BROWSER_HEADERS, timeout=TIMEOUT)
    response.raise_for_status()
    links = sorted(
        {link for link in LINK.findall(response.text) if link.rsplit("/", 1)[-1].startswith(name_start)}
    )
    if len(links) != 1:
        raise ValueError(f"ASIC {page}: expected one {name_start} workbook link, found {links}")
    return links[0]


def get_current_workbook() -> bytes:
    """Return the latest series 1 and series 2 workbook."""
    try:
        url = _workbook_url(CURRENT_PAGE, CURRENT_STEM)
    except (requests.RequestException, ValueError) as error:
        return _newest_cached(CURRENT_STEM, error)
    print(f"ASIC insolvency workbook: {url.rsplit('/', 1)[-1]}")
    return get_file(url, prefix=PREFIX, timeout=TIMEOUT, headers=BROWSER_HEADERS, fallback=True)


def get_history_workbook(name: str) -> bytes:
    """Return one of the 2022 history workbooks (HISTORY_SERIES_1 or HISTORY_SERIES_1A)."""
    try:
        url = _workbook_url(HISTORY_PAGE, name)
    except (requests.RequestException, ValueError) as error:
        return _newest_cached(name, error)
    return get_recent(url, None, PREFIX, None, TIMEOUT, headers=BROWSER_HEADERS)
