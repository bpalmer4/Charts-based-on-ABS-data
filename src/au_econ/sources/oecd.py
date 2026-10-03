"""OECD: dataflows from the OECD Data Explorer SDMX API, as CSV with codes and labels.

Cached for RECENT_MAX_AGE by the file's age, not by Last-Modified: the API's Last-Modified
is a fixed, malformed date ("2023-12-05") that does not move when the data does. The cache
keeps repeated runs inside the API's rate limit for anonymous use.
"""

import io

import pandas as pd

from au_econ.sources.http_cache import RECENT_MAX_AGE, get_recent

DATA_URL = "https://sdmx.oecd.org/public/rest/data"
TIMEOUT = 120  # seconds; the API can be slow
MAX_GAP_MONTHS = 2  # months interpolated between mid-quarter values

COUNTRIES = {  # country label: OECD reference area; OECD members and partner economies
    "Australia": "AUS",
    "Austria": "AUT",
    "Belgium": "BEL",
    "Canada": "CAN",
    "Chile": "CHL",
    "Czech Republic": "CZE",
    "Denmark": "DNK",
    "Estonia": "EST",
    "Finland": "FIN",
    "France": "FRA",
    "Germany": "DEU",
    "Greece": "GRC",
    "Hungary": "HUN",
    "Iceland": "ISL",
    "Ireland": "IRL",
    "Israel": "ISR",
    "Italy": "ITA",
    "Japan": "JPN",
    "South Korea": "KOR",
    "Latvia": "LVA",
    "Luxembourg": "LUX",
    "Mexico": "MEX",
    "Netherlands": "NLD",
    "New Zealand": "NZL",
    "Norway": "NOR",
    "Poland": "POL",
    "Portugal": "PRT",
    "Slovakia": "SVK",
    "Slovenia": "SVN",
    "Spain": "ESP",
    "Sweden": "SWE",
    "Switzerland": "CHE",
    "Turkey": "TUR",
    "United Kingdom": "GBR",
    "United States": "USA",
    "Argentina": "ARG",
    "Brazil": "BRA",
    "China": "CHN",
    "Colombia": "COL",
    "Costa Rica": "CRI",
    "India": "IND",
    "Indonesia": "IDN",
    "Lithuania": "LTU",
    "Russia": "RUS",
    "Saudi Arabia": "SAU",
    "South Africa": "ZAF",
    "Romania": "ROU",
    "Bulgaria": "BGR",
    "Croatia": "HRV",
}
LABELS = {code: label for label, code in COUNTRIES.items()}


def get_data(dataflow: str, key: str = "all", start: str | None = None) -> pd.DataFrame:
    """Return one row per observation of an OECD dataflow (e.g. "OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA,").

    key selects series by dimension codes, dot-separated, "+" for several, empty for all.
    start is an SDMX period, e.g. "2021-Q4".
    """
    params = {"format": "csvfilewithlabels"}
    if start is not None:
        params["startPeriod"] = start
    content = get_recent(f"{DATA_URL}/{dataflow}/{key}", params, "oecd", RECENT_MAX_AGE, TIMEOUT)
    frame = pd.read_csv(io.BytesIO(content))
    if frame.empty:
        raise ValueError(f"OECD {dataflow} {key}: no data")
    return frame


def get_table(dataflow: str, key: str, start: str) -> pd.DataFrame:
    """Return an OECD dataflow as a table: one row per period (as published), one column per area code."""
    rows = get_data(dataflow, key, start)
    duplicated = rows[rows.duplicated(["REF_AREA", "TIME_PERIOD"])]["REF_AREA"].unique()
    if len(duplicated):  # pivot_table would average them
        raise ValueError(f"OECD {dataflow} {key}: more than one series for {sorted(duplicated)}")
    table = rows.pivot_table(index="TIME_PERIOD", columns="REF_AREA", values="OBS_VALUE")
    return table.dropna(how="all", axis=1).dropna(how="all", axis=0)


def combine(left: pd.DataFrame | None, right: pd.DataFrame) -> pd.DataFrame:
    """Join two tables side by side, keeping left's column wherever both have a country.

    Fetch the preferred dataflow first.
    """
    if left is None:
        return right
    return pd.concat([left, right.drop(left.columns.intersection(right.columns), axis=1)], axis=1)


def quarterly_to_monthly(frame: pd.DataFrame) -> pd.DataFrame:
    """Place quarterly values in the mid-quarter month, as the OECD does, and interpolate between them.

    The frame arrives with a monthly PeriodIndex holding each quarter at its first month.
    """
    shifted = frame.set_axis(frame.index + 1)
    if not isinstance(shifted.index, pd.PeriodIndex):
        raise TypeError("expected a monthly PeriodIndex")
    monthly = shifted.reindex(pd.period_range(start=shifted.index.min(), end=shifted.index.max()))
    return monthly.interpolate(limit_area="inside", limit=MAX_GAP_MONTHS, axis=0)


def national_only(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop aggregate columns (EU, G7, OECD, ...), keeping the countries in COUNTRIES."""
    remove = frame.columns.difference(pd.Index(COUNTRIES.values()))
    if len(remove):
        print(f"Removing columns: {remove}")
        frame = frame.drop(remove, axis=1)
    return frame


def report_missing(frame: pd.DataFrame) -> None:
    """Print the countries with no data at all, and those missing the latest period."""
    missing = list(set(COUNTRIES.values()) - set(frame.columns))
    if missing:
        print(f"Missing national data for {', '.join(LABELS[code] for code in missing)}")
    final_row = frame.iloc[-1]
    missing_count = final_row.isna().sum()
    if missing_count:
        print(f"Final period: {final_row.name}")
        print(f"Missing data count for final period: {missing_count}")
        print(f"Missing data belongs to: {frame.columns[final_row.isna()].to_list()}")
        print(f"Nations with final data: {frame.columns[final_row.notna()].to_list()}")
