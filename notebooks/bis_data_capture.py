"""Get central bank policy rate data from the Bank of
   International Settlements (BIS)."""

# --- imports
import pandas as pd
import common
from pathlib import Path
import io


# --- functions
def get_bis_cbpr() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Get central bank policy rate data from the Bank of
    International Settlements (BIS).
    Returns a Tuple of two pandas Dataframes. The first
    DataFrame contains meta-data. The second Dataframe
    contains the actual data.
    NOTE: the BIS data is not always up-to-date. It may
          contain a number of NA values in the last
          few rows for some nations."""

    url = "https://www.bis.org/statistics/full_cbpol_d_csv_row.zip"
    cache_dir = Path("./BIS_CACHE")
    cache_dir.mkdir(parents=True, exist_ok=True)
    zipfile = common.get_file(url, cache_dir)
    bis = pd.read_csv(io.BytesIO(zipfile), compression='zip', low_memory=False, header=None)

    num_meta_rows = 9
    bis_meta = bis[:num_meta_rows].copy()
    bis_meta = bis_meta.set_index(0).T
    #display(bis_meta)

    bis_data = bis[num_meta_rows:].copy()
    bis_data = bis_data.set_index(0)
    bis_data.index = pd.PeriodIndex(bis_data.index, freq="D")
    bis_data.columns = pd.Index(bis_meta["Reference area"].str[3:])
    bis_data = bis_data.astype(float)
    #display(bis_data)                         

    return (bis_meta, bis_data)
