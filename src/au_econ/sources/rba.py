"""Reserve Bank of Australia: tables that readabs does not read, and the SOMP forecast tables.

readabs (read_rba_table) remains the way to read the current RBA tables. Its catalogue
lists the current daily exchange rates (F11.1, from 2023) but not the monthly F11 history
back to 1969, so those workbooks are fetched here, through http_cache. So are the forecast
tables in each Statement on Monetary Policy (SOMP), which are web pages, not workbooks.
"""

import io

import pandas as pd
import readabs as ra

from au_econ.sources.http_cache import HttpError, get_file, get_recent

HISTORICAL_URL = "https://www.rba.gov.au/statistics/tables/xls-hist/{name}.xls"
SERIES_ID_ROW = 10  # the Data sheet's header row holding the series IDs

SOMP_URL = "https://www.rba.gov.au/publications/smp/{year}/{month}/{page}.html"
SOMP_MONTHS = {1: "feb", 2: "may", 3: "aug", 4: "nov"}  # report month for each quarter
SOMP_OUTLOOK_FROM = 2024  # the forecast table moved from forecasts.html to outlook.html


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


def get_somp_table(year: int, quarter: int) -> pd.DataFrame | None:
    """Return the forecast table (the page's last table) of one SOMP, raw; None if the page cannot be had.

    A published report does not change, so each page is cached forever. A page that
    cannot be fetched is usually a report not yet published.
    """
    page = "outlook" if year >= SOMP_OUTLOOK_FROM else "forecasts"
    url = SOMP_URL.format(year=year, month=SOMP_MONTHS[quarter], page=page)
    try:
        html = get_recent(url, None, prefix="rba-somp", max_age=None)
    except HttpError as error:
        print(f"{year}-Q{quarter} skipped: {error}")
        return None
    try:
        tables = pd.read_html(io.StringIO(html.decode("utf-8")))
    except ValueError:
        print(f"{year}-Q{quarter} skipped: no tables at {url}")
        return None
    return tables[-1]
