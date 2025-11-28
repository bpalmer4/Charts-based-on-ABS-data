"""ABS data series loader.

Load and process time series data from the Australian Bureau of Statistics (ABS)
using the readabs library. Supports fetching data by catalogue number or from
local zip files, with optional growth rate calculations.

Types:
    ReqsTuple: NamedTuple specifying series selection requirements
        (catalogue, table, description, series type, unit, growth options, zip file).
    ReqsDict: Type alias for dict[str, ReqsTuple].

Constants:
    stype_codes: Mapping of series type abbreviations to full names
        (O=Original, S/SA=Seasonally Adjusted, T=Trend).

Functions:
    load_series(input_tuple: ReqsTuple) -> Series
        Load a single ABS series based on selection requirements.
    
    get_abs_data(wanted: ReqsDict) -> dict[str, Series]
        Load multiple ABS series into a dictionary.

Example:
    >>> reqs = ReqsTuple("", "640106", "All groups CPI", "O", "", False, False, "")
    >>> series = load_series(reqs)
    
    >>> wanted = {"CPI": reqs}
    >>> data = get_abs_data(wanted)

Dependencies:
    - readabs
    - pandas
"""


from typing import NamedTuple, cast
from functools import cache
import readabs as ra
from readabs import metacol as mc
from pandas import Series, DataFrame, PeriodIndex


# --- Types and codes for specifying selection requirements ---

stype_codes = {
    "O": "Original",
    "S": "Seasonally Adjusted",
    "SA": "Seasonally Adjusted",
    "T": "Trend",
}

class ReqsTuple(NamedTuple):
    """NamedTuple for specifying selection requirements."""
    cat: str                # ABS catalogue number
    table: str              # ABS table id
    did: str                # Desired text in Data Item Description
    stype: str              # Series Type: O, S, SA, T
    unit: str               # Unit of Measure
    seek_yr_growth: bool    # Whether to seek yearly growth series
    calc_growth: bool       # Whether to calculate growth rates
    zip_file: str           # Zip file for data retrieval ("" for none)

type ReqsDict = dict[str, ReqsTuple]


# --- private code ---
@cache
def get_zip_table(zip_file: str, table: str) -> tuple[DataFrame, DataFrame]:
    """Get a table from a ABS zip file of all tables.

    Args:
        zip_file (str): Path to the ABS zip file.
        table (str): Table identifier.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: data and metadata DataFrames.
    """
    dictionary, meta = ra.read_abs_cat(cat="", zip_file=zip_file, single_excel_only=table)
    data = dictionary[table]
    meta = meta[meta[mc.table] == table]
    return (data, meta)


@cache
def get_table(cat: str, table: str) -> tuple[DataFrame, DataFrame]:
    """Get ABS data table and metadata for a given ABS catalogue-id and table-id."""

    dictionary, meta = ra.read_abs_cat(cat, single_excel_only=table)
    data = dictionary[table]
    return (data, meta)


# --- public code ---
@cache
def load_series(input_tuple: ReqsTuple) -> Series:
    """Load an ABS data-series and return as a pandas Series.

    Args: 
        input_tuple (ReqsTuple): Tuple of selection requirements.

    Returns: 
        A pandas Series.

    Raises:
        ValueError: If neither cat nor zip_file is provided (need one, not both)
        ValueError: If calc_growth is requested for unsupported periodicity.
    """

    cat, table, did, stype, unit, seek_yr_growth, calc_growth, zip_file = input_tuple
    stype = stype if stype not in stype_codes else stype_codes[stype]

    # Fudge to use old CPI data until new data is available
    if cat:
        data, meta = get_table(cat, table)
    elif zip_file:
        data, meta = get_zip_table(zip_file, table)
    else:
        raise ValueError("Either cat or zip_file must be provided.")

    selector = {
        did: mc.did,
        stype: mc.stype,
    }
    if unit:
        selector[unit] = mc.unit
    if seek_yr_growth:
        # ABS inconsistent capitalisation ...
        selector["Percentage"] = mc.did
        selector["revious"] = mc.did
        selector["ear"] = mc.did
    _table, series_id, _units = ra.find_abs_id(meta, selector, verbose=False)
    series = data[series_id]
    if calc_growth:
        periodicity = cast(PeriodIndex, series.index).freqstr[0]
        p_map = {"Q": 4, "M": 12}
        if periodicity not in p_map:
            raise ValueError(f"Cannot calculate growth for periodicity '{periodicity}'")
        series = series.pct_change(periods=p_map[periodicity]) * 100.0

    return series


def get_abs_data(wanted: ReqsDict) -> dict[str, Series]:
    """Load all the ABS data series specified in the dictionary of requirements.
    
    Args:
        wanted (ReqsDict): Dictionary of desired series with names as keys.
        
    Returns:
        dict[str, Series]: Dictionary of loaded series.
    """

    box = {}
    for (w, t) in wanted.items():
        series = load_series(t)
        box[w] = series

    return box


# --- Example usage ---
if __name__ == "__main__":

    # --- extract for a single series ---
    cpi = ReqsTuple("6401.0", "640106", "All groups CPI, seasonally adjusted", "S", "", True, False, "")
    cpi_series = load_series(cpi)
    print(cpi_series.tail())

    # --- extract multiple series ---
    sought_after: ReqsDict = {
        "Monthly CPI (SA)": cpi,
        "Unemployment rate monthly (SA)":
            ReqsTuple("6202.0", "6202001", "Unemployment rate ;  Persons ;", "S", "", False, False, ""),
    }
    dataset = get_abs_data(sought_after)
    for (series_name, series_data) in dataset.items():
        print(f"{series_name}:\n{series_data.tail()}\n")