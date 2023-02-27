"""Get data from the Reserve Bank of Australia (RBA)."""

# system imports
from typing import Iterable
import re
from datetime import datetime
from pathlib import Path
import pytz

# analytic imports
import pandas as pd
from bs4 import BeautifulSoup
import requests

# local imports
import common


# -- Establish an RBA cache directory
CACHE_DIR = "./RBA_CACHE/"
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)


def clear_cache():
    """Remove all files from the cache directory."""

    path = Path(CACHE_DIR)
    _ = [f.unlink() for f in path.iterdir() if f.is_file()]


# -- get webpage addresses for the RBA links
#    but ignore the CSV data links (ie collect XLS links)
def _get_rba_links():
    """Scrape RBA website for links to datasets."""

    excel = re.compile(r"\.xlsx?$")  # ends in .xls or .xlsx

    # get key web-page
    url = "https://www.rba.gov.au/statistics/tables/"
    page = common.request_get(url)
    if page is None:
        return None

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
def _truely_unique(iterable_):
    """Ensure unique mapping."""

    found, bad = set(), set()
    for item in iterable_:
        if item in bad:
            continue
        if item in found:
            found.remove(item)
            bad.add(item)
            continue
        found.add(item)
    return found


_truely_unique = _truely_unique(_extracted_links.values())
_unique_links = {
    value: key for key, value in _extracted_links.items() if value in _truely_unique
}


# -- download a specific RBA table using a label.
def get_data_table_labels() -> Iterable[str]:
    """Return a list of labels for accessing RBA data."""
    return _truely_unique


def get_data_file(label: str) -> str | None:
    """Ensure latest version of data is in the cache file.
    Return cache file name."""
    if label not in _unique_links.keys():
        print(f"Data for {label} not found")
        return None

    # get last-modified time from file-header from RBA
    url = _unique_links[label]
    front = "https://www.rba.gov.au"
    if front not in url:
        url = front + url
    response = requests.head(url, timeout=20)  # only get header
    code = response.status_code
    if (
        code != 200
        or response.headers is None
        or "Last-Modified" not in response.headers
    ):
        print(f"Could not get web page header ({url}), error code: {code}")
        return None
    source_mtime_str = response.headers["Last-Modified"]
    source_mtime = pd.to_datetime(source_mtime_str, utc=True)

    # check cache version and use if fresh
    use_cache = True
    cache_stem = common.cachefy_name(url)
    cache_file_name = CACHE_DIR + cache_stem
    if not (path := Path(cache_file_name)).is_file():
        use_cache = False
        print(f'Cache data not stored in cache for "{label}"')
    if use_cache:
        stat = path.stat()
        cache_mtime = pd.Timestamp(datetime.fromtimestamp(stat.st_mtime, tz=pytz.utc))
        if cache_mtime < source_mtime:
            use_cache = False
            print(f'Cache data for "{label}" looks too old')
    if use_cache:
        print(f'Using cached data for "{label}"')

    # download file and save to cache
    if not use_cache:
        print(f'Downloading data for "{label}"')
        file_bytes = common.request_get(url)
        common.save_to_cache(path, file_bytes)

    return cache_file_name


def get_data(label: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Get the data."""

    # convert label to an updated-cache file name
    cache_file_name = get_data_file(label)
    if cache_file_name is None:
        return None

    # separate meta-data from the data
    # assume the series identifier is the last meta-data row
    frame = pd.read_excel(cache_file_name)
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
