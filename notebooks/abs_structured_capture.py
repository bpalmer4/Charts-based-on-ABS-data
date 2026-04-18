"""ABS data series loader.

Load and process time series data from the Australian Bureau of Statistics (ABS)
using the readabs library. Supports fetching data by catalogue number or from
local zip files, with optional growth rate calculations.

Types:
    ReqsTuple: NamedTuple specifying series selection requirements
        (catalogue, table, description, series type, unit, growth options, zip file).
    ReqsDict: Type alias for dict[str, ReqsTuple].
    AbsSeries: dataclass bundling the loaded series with its ABS metadata
        (unit, did, series_id, table, cat, stype, freq).

Constants:
    stype_codes: Mapping of series type abbreviations to full names
        (O=Original, S/SA=Seasonally Adjusted, T=Trend).

Functions:
    load_series(input_tuple: ReqsTuple) -> AbsSeries
        Load a single ABS series based on selection requirements.

    get_abs_data(wanted: ReqsDict) -> dict[str, AbsSeries]
        Load multiple ABS series into a dictionary.

    to_scale_word(unit: str) -> str
        Translate an ABS unit string to a readabs.recalibrate-compatible scale word.

Example:
    >>> reqs = ReqsTuple("6401.0", "640106", "All groups CPI", "S", "", True, False, "")
    >>> record = load_series(reqs)
    >>> record.series.tail()
    >>> record.unit            # e.g. 'Percent'

Dependencies:
    - readabs
    - pandas
"""


from dataclasses import dataclass
from typing import NamedTuple, cast

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
    unit: str               # Unit of Measure filter (matches metadata Unit column)
    seek_yr_growth: bool    # Whether to seek yearly growth series
    calc_growth: bool       # Whether to calculate growth rates
    zip_file: str           # Zip file for data retrieval ("" for none)

type ReqsDict = dict[str, ReqsTuple]


@dataclass(frozen=True)
class AbsSeries:
    """An ABS-published time series bundled with its metadata.

    `unit` is the raw ABS unit string (e.g. '000', 'Number', 'Percent',
    'Index Numbers'). Use `to_scale_word(unit)` to get a recalibrate-compatible
    starting unit for chart axes.
    """
    series: Series          # the time series itself
    unit: str               # raw ABS Unit string
    did: str                # full Data Item Description (as requested)
    series_id: str          # ABS series ID (e.g. 'A2133255F')
    table: str              # ABS table ID (e.g. '310101')
    cat: str                # ABS catalogue (e.g. '3101.0')
    stype: str              # series type (Original / Seasonally Adjusted / Trend)
    freq: str               # frequency code, one char: 'Q', 'M', 'A', ...


# Map ABS raw Unit strings to scale words understood by readabs.recalibrate.
# Only scale-bearing units need translation; 'Percent', 'Index Numbers', etc.
# pass through unchanged (recalibrate leaves them alone).
_ABS_UNIT_MAP = {
    "000": "Thousands",
    "$Millions": "Millions",
    "$ Millions": "Millions",
    "$Billions": "Billions",
    "$ Billions": "Billions",
}


def to_scale_word(unit: str) -> str:
    """Translate an ABS unit string to a recalibrate-compatible scale word."""
    return _ABS_UNIT_MAP.get(unit, unit)


# --- private code ---
def get_zip_table(zip_file: str, table: str, **kwargs) -> tuple[DataFrame, DataFrame]:
    """Get a table from a ABS zip file of all tables.

    Args:
        zip_file (str): Path to the ABS zip file.
        table (str): Table identifier.
        **kwargs: Additional keyword arguments passed to ra.read_abs_cat.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: data and metadata DataFrames.
    """
    dictionary, meta = ra.read_abs_cat(cat="", zip_file=zip_file, single_excel_only=table, **kwargs)
    data = dictionary[table]
    meta = meta[meta[mc.table] == table]
    return (data, meta)


def get_table(cat: str, table: str, **kwargs) -> tuple[DataFrame, DataFrame]:
    """Get ABS data table and metadata for a given ABS catalogue-id and table-id."""

    dictionary, meta = ra.read_abs_cat(cat, single_excel_only=table, verbose=False, **kwargs)
    data = dictionary[table]
    meta = meta[meta[mc.table] == table]
    return (data, meta)


# --- public code ---
def load_series(input_tuple: ReqsTuple, verbose=False, **kwargs) -> AbsSeries:
    """Load an ABS data-series and return as an AbsSeries record.

    Args:
        input_tuple (ReqsTuple): Tuple of selection requirements.
        verbose (bool): Whether to print verbose output.

    Returns:
        AbsSeries: the series plus its ABS metadata.

    Raises:
        ValueError: If neither cat nor zip_file is provided (need one, not both)
        ValueError: If calc_growth is requested for unsupported periodicity.
    """

    cat, table, did, stype, unit, seek_yr_growth, calc_growth, zip_file = input_tuple
    stype = stype if stype not in stype_codes else stype_codes[stype]

    if cat:
        data, meta = get_table(cat, table, **kwargs)
    elif zip_file:
        data, meta = get_zip_table(zip_file, table, **kwargs)
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
    _table, series_id, series_unit = ra.find_abs_id(meta, selector, verbose=verbose)
    series = data[series_id]

    # Trim trailing NaN values (e.g. Appendix1a extends beyond published data)
    last_valid = series.last_valid_index()
    if last_valid is not None:
        series = series.loc[:last_valid]

    if calc_growth:
        periodicity = cast(PeriodIndex, series.index).freqstr[0]
        p_map = {"Q": 4, "M": 12}
        if periodicity not in p_map:
            raise ValueError(f"Cannot calculate growth for periodicity '{periodicity}'")
        series = series.pct_change(periods=p_map[periodicity]) * 100.0
        series_unit = "Percent"   # values are now an annual % change

    freq = cast(PeriodIndex, series.index).freqstr[0]

    return AbsSeries(
        series=series,
        unit=series_unit,
        did=did,
        series_id=series_id,
        table=table,
        cat=cat,
        stype=stype,
        freq=freq,
    )


def get_abs_data(wanted: ReqsDict, verbose=False, **kwargs) -> dict[str, AbsSeries]:
    """Load all the ABS data series specified in the dictionary of requirements.

    Args:
        wanted (ReqsDict): Dictionary of desired series with names as keys.
        verbose (bool): Whether to print verbose output.

    Returns:
        dict[str, AbsSeries]: Dictionary of loaded AbsSeries records.
    """

    return {
        name: load_series(req, verbose=verbose, **kwargs)
        for name, req in wanted.items()
    }


# --- Example usage ---
if __name__ == "__main__":

    # --- extract for a single series ---
    cpi = ReqsTuple("6401.0", "640106", "All groups CPI, seasonally adjusted", "S", "", True, False, "")
    cpi_rec = load_series(cpi)
    print(cpi_rec.series.tail())
    print(f"unit={cpi_rec.unit!r}, freq={cpi_rec.freq!r}")

    # --- extract multiple series ---
    sought_after: ReqsDict = {
        "Monthly CPI (SA)": cpi,
        "Unemployment rate monthly (SA)":
            ReqsTuple("6202.0", "6202001", "Unemployment rate ;  Persons ;", "S", "", False, False, ""),
    }
    dataset = get_abs_data(sought_after)
    for (series_name, record) in dataset.items():
        print(f"{series_name} ({record.unit}):\n{record.series.tail()}\n")
