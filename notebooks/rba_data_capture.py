"""Get data from the Reserve Bank of Australia (RBA)."""

# system imports
import io
import re
import sys
from pathlib import Path
from typing import Iterable

# analytic imports
import pandas as pd
from bs4 import BeautifulSoup

# local imports
import common

# -- Establish an RBA cache directory
CACHE_DIR = "./RBA_CACHE/"
CACHE_PATH = Path(CACHE_DIR)
CACHE_PATH.mkdir(parents=True, exist_ok=True)


def clear_cache() -> None:
    """Remove all files from the cache directory."""
    for f in CACHE_PATH.iterdir():
        if f.is_file():
            f.unlink()


# -- get webpage addresses for the RBA links
#    but ignore the CSV data links (ie collect XLS links)
def _get_rba_links() -> dict[str, str]:
    """Scrape RBA website for links to datasets."""

    excel = re.compile(r"\.xlsx?$")  # ends in .xls or .xlsx

    # get key web-page
    url = "https://www.rba.gov.au/statistics/tables/"
    page = common.request_get(url)
    if page is None:
        sys.exit(f"Could not get page from {url}")

    # extract web addresses
    white_space = re.compile(r"\s+")
    extracted_links = {}  # link: decription
    soup = BeautifulSoup(page, features="lxml")
    main_list = soup.find_all(
        "div",
        {
            "id": "tables-list",
        },
    )
    assert len(main_list) == 1
    links = main_list[0].find_all("a")
    for k in links:
        text = re.sub(white_space, " ", k.getText()).strip()
        link = k["href"]
        if link and re.search(excel, link):
            extracted_links[link] = text
    return extracted_links


_extracted_links = _get_rba_links()


# -- build a truely unique links table
def _truely_unique(iterable_: Iterable[str]) -> set[str]:
    """Ensure unique mapping."""

    found: set[str] = set()
    bad: set[str] = set()
    for item in iterable_:
        if item in bad:
            continue
        if item in found:
            found.remove(item)
            bad.add(item)
            continue
        found.add(item)
    return found


_uniqueness = _truely_unique(_extracted_links.values())
_unique_links = {
    value: key for key, value in _extracted_links.items() if value in _uniqueness
}


# -- download a specific RBA table using a label.
def get_data_table_labels() -> Iterable[str]:
    """Return a list of labels for accessing RBA data."""
    return _uniqueness


def get_data_file(label: str) -> bytes:
    """Ensure latest version of data is in the cache file.
    Return cache file name."""
    if label not in _unique_links.keys():
        raise ValueError(f"Data for {label} not found")

    # get last-modified time from file-header from RBA
    url = _unique_links[label]
    front = "https://www.rba.gov.au"
    if front not in url:
        url = front + url

    # return the data
    return common.get_file(url, CACHE_PATH)


def get_data(label: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Get the data."""

    # convert label to an updated-cache file name
    data_bytes = io.BytesIO(get_data_file(label))

    # separate meta-data from the data
    # assume the series identifier is the last meta-data row
    frame = pd.read_excel(data_bytes)
    frame = frame.set_index(frame.columns[0])
    series_id = "Series ID"
    if series_id not in frame.index:
        series_id = "Mnemonic"
        if series_id not in frame.index:
            print(f"Series identifier not found for {label}")
            return None
    sid_row = (frame.index.get_loc(series_id)) + 1
    meta = frame.iloc[0:sid_row]
    data = frame[sid_row:].dropna(how="all", axis=0).copy()

    # index columns by the Series ID
    sid = meta[meta.index == series_id].squeeze()
    if isinstance(sid, str):
        sid = [sid]  # unsqueeze
    meta.columns = sid
    data.columns = sid

    # Date adjustments - but not for distributional data
    if "istribution" not in label:
        data.index = pd.to_datetime(data.index)

    return meta.T.dropna(how="all", axis=1), data
