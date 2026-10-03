"""Reserve Bank of Australia: historical statistical tables that readabs' RBA catalogue does not list.

readabs (read_rba_table) remains the way to read the current RBA tables. Its catalogue
lists the current daily exchange rates (F11.1, from 2023) but not the monthly F11 history
back to 1969, so those workbooks are fetched here, through http_cache.
"""

import io

import pandas as pd
import readabs as ra

from au_econ.sources.http_cache import get_file

HISTORICAL_URL = "https://www.rba.gov.au/statistics/tables/xls-hist/{name}.xls"
SERIES_ID_ROW = 10  # the Data sheet's header row holding the series IDs


def get_table(table: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return an RBA statistical table (e.g. "F2", "Z:F2-Daily-2013") and its metadata, via readabs."""
    data, meta = ra.read_rba_table(table)
    if data.empty:
        raise ValueError(f"RBA {table}: no data")
    return data, meta


def get_historical_table(name: str) -> pd.DataFrame:
    """Return the Data sheet of an RBA historical workbook (e.g. "f11hist"), one column per series ID."""
    workbook = get_file(HISTORICAL_URL.format(name=name), prefix="rba")
    table = pd.read_excel(io.BytesIO(workbook), sheet_name="Data", header=SERIES_ID_ROW, index_col=0)
    if table.empty:
        raise ValueError(f"RBA {name}: no data")
    return table
