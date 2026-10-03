"""U.S. Energy Information Administration: petroleum price history spreadsheets, from eia.gov.

Each series is an .xls workbook with a "Data 1" sheet of dates and prices, downloaded
through http_cache (EIA sends Last-Modified). No API key is needed.
"""

import io

import pandas as pd

from au_econ.sources.http_cache import get_file

PREFIX = "eia"
BASE_URL = "https://www.eia.gov/dnav/pet/hist_xls/"
SHEET = "Data 1"
HEADER_ROWS = 2

# workbook names, by series
WTI_SPOT = "RWTCd.xls"  # WTI Cushing, daily, USD per barrel
BRENT_SPOT = "RBRTEd.xls"  # Brent Europe, daily, USD per barrel
GULF_JET_SPOT = "eer_epjk_pf4_rgc_dpgd.xls"  # US Gulf Coast kerosene-type jet fuel, daily, USD per gallon
RETAIL_PETROL = "EMM_EPMR_PTE_NUS_DPGw.xls"  # US regular petrol, all formulations, weekly, USD per gallon


def get_prices(workbook: str) -> pd.DataFrame:
    """Return a price history workbook as a date/price frame, oldest first."""
    content = get_file(BASE_URL + workbook, prefix=PREFIX)
    frame = pd.read_excel(io.BytesIO(content), sheet_name=SHEET, skiprows=HEADER_ROWS)
    frame.columns = pd.Index(["date", "price"])
    frame = frame.dropna()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["price"] = frame["price"].astype(float)
    if frame.empty:
        raise ValueError(f"EIA {workbook}: no rows")
    return frame.sort_values("date").reset_index(drop=True)


def get_series(workbook: str, name: str, scale: float = 1.0) -> pd.Series:
    """Return a price history as a daily-PeriodIndex series (times scale), one value per date."""
    frame = get_prices(workbook)
    series = pd.Series(frame["price"].to_numpy() * scale, index=pd.PeriodIndex(frame["date"], freq="D"), name=name)
    return series[~series.index.duplicated(keep="last")]
