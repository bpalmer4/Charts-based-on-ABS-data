# rba_common.py - retrieve data from the RBA

import numpy as np
import pandas as pd
import matplotlib as mpl
from datetime import datetime

from pathlib import Path
import requests
from bs4 import BeautifulSoup
import re
from typing import Iterable, Optional, Tuple
from datetime import datetime
import pytz

# local imports 
import common


# -- Establish an RBA cache directory
CACHE_DIR = "./RBA_CACHE/"
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)


# -- get webpage addresses for the RBA links
#    but ignore the CSV data links (ie collect XLS links)
def _get_rba_links():
    XLS = re.compile('\.xlsx?$') # ends in .xls or .xlsx

    # get key web-page
    url = "https://www.rba.gov.au/statistics/tables/"
    page = common.request_get(url)
    if page is None: return None
    
    # extract web addresses
    white_space = re.compile("\s\s+")
    extracted_links = {} # link: decription
    soup = BeautifulSoup(page, features="lxml")
    main_list = soup.find_all("div", {"id": "tables-list",})
    assert len(main_list) == 1
    links = main_list[0].find_all("a")
    for k in links:
        text = re.sub(white_space, " ", k.getText()).strip()
        link = k['href']
        if link and re.search(XLS, link):
            extracted_links[link] = text
    return extracted_links
_extracted_links = _get_rba_links()


# -- build a truely unique links table
def _truely_unique(iterable_):
    found, bad = set(), set()
    for i in iterable_:
        if i in bad: 
            continue
        if i in found:
            del found[i]
            bad.add(i)
            continue
        found.add(i)
    return found
_truely_unique = _truely_unique(_extracted_links.values())
_unique_links = {value: key for key, value in _extracted_links.items() if value in _truely_unique}


# -- download a specific RBA table using a label.
def get_data_table_labels() -> Iterable[str]:
    """Return a list of labels for accessing RBA data."""
    return _truely_unique

def get_data_file(label: str) -> Optional[str]:
    """Ensure latest version of data is in the cache file. 
       Return cache file name."""
    if label not in _unique_links.keys():
        print(f'Data for {label} not found')
        return (None)

    # get last-modified time from file-header from RBA
    url = _unique_links[label]
    front = 'https://www.rba.gov.au'
    if front not in url:
        url = front + url
    response = requests.head(url) # only get header
    code = response.status_code
    if code != 200 or response.headers is None or 'Last-Modified' not in response.headers:
        print(f'Could not get web page header ({url}), error code: {code}')
        return None
    source_mtime_str = response.headers['Last-Modified']
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


def get_data(label: str) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    # convert label to an updated-cache file name
    cache_file_name = get_data_file(label)
    if cache_file_name is None:
        return None

    # separate meta-data from the data
    # assume the series identifier is the last meta-data row
    df = pd.read_excel(cache_file_name)
    df = df.set_index(df.columns[0])
    SERIES_ID = 'Series ID'
    if SERIES_ID not in df.index:
        SERIES_ID = 'Mnemonic'
        if SERIES_ID not in df.index:
            print(f'Series identifier not found for {label}')
            return None
    sid_row = (df.index.get_loc(SERIES_ID)) + 1
    meta = df.iloc[0:sid_row]
    data = df[sid_row:].dropna(how='all', axis=0).copy()

    # index columns by the Series ID
    sid = meta[meta.index == SERIES_ID].squeeze()
    if type(sid) == str:
        sid = [sid] # unsqueeze
    meta.columns = sid
    data.columns = sid 

    # Date adjustments - but not for distributional data
    if 'istribution' not in label:
        data.index = pd.to_datetime(data.index)

    return meta.T.dropna(how='all', axis=1), data


###### quick testing
if __name__ == "__main__":
    for label in _unique_links.keys():
        print('===============================')
        print(label)
        tup = get_data(label)
        if tup is not None:
            meta, data = tup
            print('---------------')
            print(meta)
            print('---------------')
            print(data.head())