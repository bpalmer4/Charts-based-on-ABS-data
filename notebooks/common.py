# common.py
import pandas as pd

import re
import requests
from typing import Optional, Tuple
from pathlib import Path

_bad_cache_pattern = re.compile('[~"#%&*:<>?/\\{|}\.]+')
def cachefy_name(name:str) -> str:
    new_name = re.sub(_bad_cache_pattern, '_', name)
    return new_name


def request_get(url:str) -> Optional[bytes]:
    gotten = requests.get(url, allow_redirects=True)
    code = gotten.status_code
    if code != 200:
        print(f'Could not get web page ({url}), error code: {code}')
        return None
    contents = gotten.content # bytes
    return contents


def save_to_cache(path:Path, contents:bytes):
    path.open(mode='w', buffering=-1, encoding=None, errors=None, newline=None)
    path.write_bytes(contents)

    
# --- get Bank of International Settlements data
def get_bis_cbpr() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get the Central Bank Policy Rate data from the Bank
       of International Settlements (BIS).
       Returns a Tuple of two pandas Dataframes. The first 
       DataFrame contains meta-data. The second Dataframe  
       contains the actual data.
       NOTE: the BIS data is not always up-to-date. It may 
             contain a number of NA values in the last
             few rows for some nations."""

    url = "https://www.bis.org/statistics/full_cbpol_d_csv_row.zip"
    bis = pd.read_csv(url, low_memory=False, header=None)

    META_ROWS = 9
    bis_meta = bis[:META_ROWS].copy()
    bis_meta = bis_meta.set_index(0).T
    #display(bis_meta.head())

    bis_data = bis[META_ROWS:].copy()
    bis_data = bis_data.set_index(0)
    bis_data.index = pd.PeriodIndex(bis_data.index, freq='D')
    names = bis_meta['Reference area'].str[3:]
    bis_data.columns = names
    bis_data = bis_data.astype(float)
    
    return (bis_meta, bis_data)

