"""This module does three things:
A. It obtains the freshest time-series data from the 
   Australian Bureau of Statistics (ABS).
B. It allows that data to be searched for a specific series.
C. It provides a short-hand way to plot the ABS data.

In respect of getting data from the ABS, the general 
approach is to:
1. Download the "latest-release" webpage from the ABS.

2. Scan that webpage to find the link(s) to the download
   all-tables zip-file. We do this because the name of
   the file location on the ABS server changes from
   month to month, and varies beyween ABS webpages.

3. Get the URL headers for this file, amd compare freshness
   with the version in the local cache directory (if any).

4. Use either the zip-file from the cache, or download
   a zip-file from the ABS, save it to the cache,
   and use that file.

5. Open the zip-file, and extract each table to a pandas
   DataFrame with a PeriodIndex. And save the metadata
   to a pandas DataFrame. Return all of these DataFrames
   in a dictionary.

Useful information from the ABS website ...
i.   ABS Catalog numbers:
https://www.abs.gov.au/about/data-services/help/abs-time-series-directory

ii.  ABS Landing Pages:
https://www.abs.gov.au/welcome-new-abs-website#navigating-our-web-address-structure."""

# === imports
# standard library imports
import calendar
import io
import re
import zipfile
from collections import namedtuple
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Callable, Final, TypeVar, TypeAlias, cast, Sequence

# analytical imports
import pandas as pd
from pandas import Series, DataFrame
from bs4 import BeautifulSoup

# local imports
import common
from plotting import (
    recalibrate,
    seas_trend_plot,
    line_plot,
    abbreviate,
    state_colors,
    state_abbr,
    LEGEND_SET,
)
from utility import qtly_to_monthly


# === typing information
# public
# an unexpected error when capturing ABS data ...
class AbsCaptureError(Exception):
    """Raised when the data capture process goes awry."""


# abbreviations for columns in the metadata DataFrame
Metacol = namedtuple(
    "Metacol",
    [
        "did",
        "stype",
        "id",
        "start",
        "end",
        "num",
        "unit",
        "dtype",
        "freq",
        "cmonth",
        "table",
        "tdesc",
        "cat",
    ],
)


# An unpacked zipfile and metadata
AbsDict = dict[str, pd.DataFrame]


# keywords to navigate to an ABS landing page
@dataclass(frozen=True)
class AbsLandingPage:
    """Class for identifying ABS landing pages by theme,
    parent-topic and topic."""

    theme: str
    parent_topic: str
    topic: str


@dataclass
class AbsSelectInput:
    """Data used to select muktiple ABS timeseries
    from different sources within the ABS."""

    landing_page: AbsLandingPage
    table: str
    orig_sa: str
    search1: str
    search2: str
    abbr: str
    calc_growth: bool


@dataclass
class AbsSelectOutput:
    """For each series returned, include some useful metadata."""

    series: pd.Series
    cat_id: str
    table: str
    series_id: str
    unit: str
    orig_sa: str
    abbr: str


AbsSelectionDict: TypeAlias = dict[str, AbsSelectInput]
AbsMultiSeries: TypeAlias = dict[str, AbsSelectOutput]
DataT = TypeVar("DataT", Series, DataFrame)  # python 3.11+


# === Constants
SEAS_ADJ: Final[str] = "Seasonally Adjusted"
TREND: Final[str] = "Trend"
ORIG: Final[str] = "Original"

# public
META_DATA: Final[str] = "META_DATA"
metacol = Metacol(
    did="Data Item Description",
    stype="Series Type",
    id="Series ID",
    start="Series Start",
    end="Series End",
    num="No. Obs.",
    unit="Unit",
    dtype="Data Type",
    freq="Freq.",
    cmonth="Collection Month",
    table="Table",
    tdesc="Table Description",
    cat="Catalogue number",
)


# private = constants
_CACHE_DIR: Final[str] = "./ABS_CACHE/"
_CACHE_PATH: Final[Path] = Path(_CACHE_DIR)
_CACHE_PATH.mkdir(parents=True, exist_ok=True)


# === utility functions
# public
def get_fs_constants(
    abs_dict: AbsDict,
    landing_page: AbsLandingPage,
    chart_dir_suffix: str = "",
) -> tuple[str, str, str, pd.DataFrame]:
    """Get file system constants for a catalogue ID."""

    cat_id = abs_dict[META_DATA][metacol.cat].unique()[0]
    source = f"ABS {cat_id}"
    chart_dir = f"./CHARTS/{cat_id} - {landing_page.topic}{chart_dir_suffix}/"
    Path(chart_dir).mkdir(parents=True, exist_ok=True)
    return source, chart_dir, cat_id, abs_dict[META_DATA]


# public
def clear_cache() -> None:
    """Clear the cache directory of zip and xlsx files."""

    extensions = ("*.zip", "*.ZIP", "*.xlsx", "*.XLSX")
    for extension in extensions:
        for fs_object in Path(_CACHE_DIR).glob(extension):
            if fs_object.is_file():
                fs_object.unlink()


# public
def get_plot_constants(
    meta: DataFrame,
) -> tuple[pd.Timestamp, list[None | pd.Timestamp], tuple[str, str]]:
    """Get plotting constants from ABS meta data table
    - used in a loop to produce a plot of the full
      series, and a plot of the recent period."""

    recency_period = 5  # years
    recency_extra = 6  # months
    today = pd.Timestamp("today")
    date_series = meta["Series End"][meta["Series End"] <= today]
    reasonable_end = date_series.max() if len(date_series) > 0 else today
    recent = reasonable_end - pd.DateOffset(years=recency_period, months=recency_extra)
    plot_times = [None, recent]
    plot_tags = ("full", "recent")
    return recent, plot_times, plot_tags


# public
def fix_abs_title(title: str, lfooter: str) -> tuple[str, str]:
    """Simplify complex ABS series names."""

    check = [
        "Chain volume measures",  # National Accounts,
        "Chain Volume Measures",  # Business indicators,
        "Chain Volume Measure",  # Retail Trade
        "Current Prices",
        "Current Price",
        "Total (State)",
        "Total (Industry)",
        # Business Indicators
        "CORP",
        "TOTAL (SCP_SCOPE)",
    ]
    for c in check:
        if c in title:
            title = title.replace(f"{c} ;", "")
            lfooter += f"{c}. "

    for s, abbr in state_abbr.items():
        title = title.replace(s, abbr)

    title = (
        title.replace(";", "")
        .replace(" - ", " ")
        .replace("    ", " ")
        .replace("   ", " ")
        .replace("  ", " ")
        .strip()
    )
    return title, lfooter


# === Data capture from the ABS
# private
def _get_abs_page(
    page: AbsLandingPage,
) -> bytes:
    """Return the HTML for the ABS topic landing page."""

    head = "https://www.abs.gov.au/statistics/"
    tail = "/latest-release"
    url = f"{head}{page.theme}/{page.parent_topic}/{page.topic}{tail}"
    return common.request_get(url)


# private
def _prefix_url(url: str) -> str:
    """Apply ABS URL prefix to relative links."""

    prefix = "https://www.abs.gov.au"
    # remove a prefix if it already exists (just to be sure)
    url = url.replace(prefix, "")
    url = url.replace(prefix.replace("https://", "http://"), "")
    # add the prefix (back) ...
    return f"{prefix}{url}"


# public
@cache
def get_data_links(
    landing_page: AbsLandingPage,
    verbose: bool = False,
    inspect="",  # for debugging - save the landing page to disk
) -> dict[str, list[str]]:
    """Scan the ABS landing page for links to ZIP files and for
    links to Microsoft Excel files. Return the links in
    a dictionary of lists by file type ending. Ensure relative
    links are fully expanded."""

    # get relevant web-page from ABS website
    page = _get_abs_page(landing_page)

    # save the page to disk for inspection
    if inspect:
        with open(inspect, "w") as file_handle:
            file_handle.write(page.decode("utf-8"))

    # remove those pesky span tags - probably not necessary
    page = re.sub(b"<span[^>]*>", b" ", page)
    page = re.sub(b"</span>", b" ", page)
    page = re.sub(b"\\s+", b" ", page)  # tidy up white space

    # capture all links (of a particular type)
    link_types = (".xlsx", ".zip", ".xls")  # must be lower case
    soup = BeautifulSoup(page, features="lxml")
    link_dict: dict[str, list[str]] = {}
    for link in soup.findAll("a"):
        url = link.get("href")
        if url is None:
            # ignore silly cases
            continue
        for link_type in link_types:
            if url.lower().endswith(link_type):
                if link_type not in link_dict:
                    link_dict[link_type] = []
                link_dict[link_type].append(_prefix_url(url))
                break

    if verbose:
        for link_type, link_list in link_dict.items():
            summary = [x.split("/")[-1] for x in link_list]  # just the file name
            print(f"Found: {len(link_list)} items of type {link_type}: {summary}")

    return link_dict


# private
def _get_abs_zip_file(
    landing_page: AbsLandingPage,
    zip_table: int,
    verbose: bool,
    inspect: str,
) -> bytes:
    """Get the latest zip_file of all tables for
    a specified ABS catalogue identifier"""

    link_dict = get_data_links(landing_page, verbose, inspect)

    # happy case - found a .zip URL on the ABS page
    if r".zip" in link_dict and zip_table < len(link_dict[".zip"]):
        url = link_dict[".zip"][zip_table]
        return common.get_file(
            url, _CACHE_PATH, cache_name_prefix=landing_page.topic, verbose=verbose
        )

    # sad case - need to fake up a zip file
    print("A little unexpected: We need to fake up a zip file")
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for u in link_dict[".xlsx"]:
            u = _prefix_url(u)
            file_bytes = common.get_file(
                u, _CACHE_PATH, cache_name_prefix=landing_page.topic, verbose=verbose
            )
            name = Path(u).name
            zip_file.writestr(f"/{name}", file_bytes)
    zip_buf.seek(0)
    return zip_buf.read()


# private
def _get_meta_from_excel(
    excel: pd.ExcelFile, tab_num: str, tab_desc: str, cat_id: str
) -> pd.DataFrame:
    """Capture the metadata from the Index sheet of an ABS excel file.
    Returns a DataFrame specific to the current excel file.
    Returning an empty DataFrame, mneans that the meatadata could not
    be identified."""

    # Unfortunately, the header for some of the 3401.0
    #                spreadsheets starts on row 10
    starting_rows = 9, 10
    required = metacol.did, metacol.id, metacol.stype, metacol.unit
    required_set = set(required)
    for header_row in starting_rows:
        file_meta = excel.parse(
            "Index",
            header=header_row,
            parse_dates=True,
            infer_datetime_format=True,
            converters={"Unit": str},
        )
        file_meta = file_meta.iloc[1:-2]  # drop first and last 2
        file_meta = file_meta.dropna(axis="columns", how="all")

        if required_set.issubset(set(file_meta.columns)):
            break

        if header_row == starting_rows[-1]:
            print(f"Could not find metadata for {cat_id}-{tab_num}")
            return pd.DataFrame()

    # make damn sure there are no rogue white spaces
    for col in required:
        file_meta[col] = file_meta[col].str.strip()

    # standarise some units
    file_meta[metacol.unit] = (
        file_meta[metacol.unit]
        .str.replace("000 Hours", "Thousand Hours")
        .replace("$'000,000", "$ Million")
        .replace("$'000", " $ Thousand")
        .replace("000,000", "Millions")
        .replace("000", "Thousands")
    )
    file_meta[metacol.table] = tab_num.strip()
    file_meta[metacol.tdesc] = tab_desc.strip()
    file_meta[metacol.cat] = cat_id.strip()
    return file_meta


# private
def _unpack_excel_into_df(
    excel: pd.ExcelFile, meta: DataFrame, freq: str, verbose: bool
) -> DataFrame:
    """Take an ABS excel file and put all the Data sheets into a single
    pandas DataFrame and return that DataFrame."""

    data = DataFrame()
    data_sheets = [x for x in excel.sheet_names if cast(str, x).startswith("Data")]
    for sheet_name in data_sheets:
        sheet_data = excel.parse(
            sheet_name,
            header=9,
            index_col=0,
        ).dropna(how="all", axis="index")
        data.index = pd.to_datetime(data.index)

        for i in sheet_data.columns:
            if i in data.columns:
                # Remove duplicate Series IDs before merging
                del sheet_data[i]
                continue
            if verbose and sheet_data[i].isna().all():
                # Warn if data series is all NA
                problematic = meta.loc[meta["Series ID"] == i][
                    ["Table", "Data Item Description", "Series Type"]
                ]
                print(f"Warning, this data series is all NA: {i} (details below)")
                print(f"{problematic}\n\n")

        # merge data into a large dataframe
        if len(data) == 0:
            data = sheet_data
        else:
            data = pd.merge(
                left=data,
                right=sheet_data,
                how="outer",
                left_index=True,
                right_index=True,
                suffixes=("", ""),
            )
    if freq:
        if freq in ("Q", "A"):
            month = calendar.month_abbr[
                cast(pd.PeriodIndex, data.index).month.max()
            ].upper()
            freq = f"{freq}-{month}"
        if isinstance(data.index, pd.DatetimeIndex):
            data = data.to_period(freq=freq)

    return data


# regex patterns for the next function
PATTERN_SUBSUB = re.compile(r"_([0-9]+[a-zA-Z]?)_")
PATTERN_NUM_ALPHA = re.compile(r"^([0-9]+[a-zA-Z]?)_[a-zA-z_]+$")
PATTERN_FOUND = re.compile(r"^[0-9]+[a-zA-Z]?$")


# private
def _get_table_name(z_name: str, e_name: str, verbose: bool):
    """Try and get a consistent and unique naming system for the tables
    found in each zip-file. This is a bit fraught because the ABS does
    this differently for various catalog identifiers.
    Arguments:
    z_name - the file name from zip-file.
    e_name - the self reported table name from the excel spreadsheet.
    verbose - provide additional feedback on this step."""

    # first - lets look at table number from the zip-file name
    z_name = (
        z_name.split(".")[0][4:]
        .replace("55003DO0", "")
        .replace("55024", "")
        .replace("55001Table", "")
        .replace("_Table_", "")
        .replace("55001_", "")
        .lstrip("0")
    )

    if result := re.search(PATTERN_SUBSUB, z_name):
        # search looks anywhere in the string
        z_name = result.group(1)
    if result := re.match(PATTERN_NUM_ALPHA, z_name):
        # match looks from the beginning
        z_name = result.group(1)

    # second - lets see if we can get a table name from the excel meta data.
    splat = e_name.replace("-", " ").replace("_", " ").split(".")
    e_name = splat[0].split(" ")[-1].strip().lstrip("0")

    # third - let's pick the best one
    if e_name == z_name:
        r_value = e_name
    elif re.match(PATTERN_FOUND, e_name):
        r_value = e_name
    else:
        r_value = z_name

    if verbose:
        print(f"table names: {z_name=} {e_name=} --> {r_value=}")
    return r_value


# private
def _get_all_dataframes(zip_file: bytes, verbose: bool) -> AbsDict:
    """Get a DataFrame for each table in the zip-file, plus a DataFrame
    for the metadata. Return these in a dictionary
    Arguments:
     - zip_file - ABS zipfile as a bytes array - contains excel spreadsheets
     - verbose - provide additional feedback on this step.
    Returns:
     - either an empty dictionary (failure) or a dictionary containing
       a separate DataFrame for each table in the zip-file,
       plus a DataFrame called META_DATA for the metadata.
    """

    if verbose:
        print("Extracting DataFrames from the zip-file.")
    freq_dict = {"annual": "Y", "biannual": "Q", "quarter": "Q", "month": "M"}
    returnable: dict[str, DataFrame] = {}
    meta = DataFrame()

    with zipfile.ZipFile(io.BytesIO(zip_file)) as zipped:
        for count, element in enumerate(zipped.infolist()):
            # get the zipfile into pandas
            excel = pd.ExcelFile(io.BytesIO(zipped.read(element.filename)))

            # get table information
            if "Index" not in excel.sheet_names:
                print(
                    "Caution: Could not find the 'Index' "
                    f"sheet in {element.filename}. File not included"
                )
                continue

            # get table header information
            header = excel.parse("Index", nrows=8)  # ???
            cat_id = header.iat[3, 1].split(" ")[0].strip()
            table_name = _get_table_name(
                z_name=element.filename,
                e_name=header.iat[4, 1],
                verbose=verbose,
            )
            tab_desc = header.iat[4, 1].split(".", 1)[-1].strip()

            # get the metadata rows
            file_meta = _get_meta_from_excel(excel, table_name, tab_desc, cat_id)
            if len(file_meta) == 0:
                continue

            # establish freq - used for making the index a PeriodIndex
            freqlist = file_meta["Freq."].str.lower().unique()
            if not len(freqlist) == 1 or freqlist[0] not in freq_dict:
                print(f"Unrecognised data frequency {freqlist} for {tab_desc}")
                continue
            freq = freq_dict[freqlist[0]]

            # fix tabulation when ABS uses the same table numbers for data
            # This happens occasionally
            if table_name in returnable:
                tmp = f"{table_name}-{count}"
                if verbose:
                    print(f"Changing duplicate table name from {table_name} to {tmp}.")
                table_name = tmp
                file_meta[metacol.table] = table_name

            # aggregate the meta data
            meta = pd.concat([meta, file_meta])

            # add the table to the returnable dictionary
            returnable[table_name] = _unpack_excel_into_df(
                excel, file_meta, freq, verbose
            )

    returnable[META_DATA] = meta
    return returnable


# public
@cache
def get_abs_data(
    landing_page: AbsLandingPage,
    zip_table: int = 0,
    verbose: bool = False,
    inspect: str = "",  # filename for saving the webpage
) -> AbsDict:
    """For the relevant ABS page return a dictionary containing
    a meta-data Data-Frame and one or more DataFrames of actual
    data from the ABS.
    Arguments:
     - page - class ABS_topic_page - desired time_series page in
            the format:
            abs.gov.au/statistics/theme/parent-topic/topic/latest-release
     - zip_table - select the zipfile to return in order as it
            appears on the ABS webpage - default=0
            (e.g. 6291 has four possible tables,
            but most ABS pages only have one).
            Note: a negative zip_file number will cause the
            zip_file not to be recovered and for individual
            excel files to be recovered from the ABS
     - verbose - display additional web-scraping and caching information.
     - inspect - save the webpage to disk for inspection -
            inspect is the file name."""

    if verbose:
        print(f"In get_abs_data() {zip_table=} {verbose=}")
        print(f"About to get data on: {landing_page.topic.replace('-', ' ').title()} ")
    zip_file = _get_abs_zip_file(landing_page, zip_table, verbose, inspect)
    if not zip_file:
        raise AbsCaptureError("An unexpected empty zipfile.")
    dictionary = _get_all_dataframes(zip_file, verbose=verbose)
    if len(dictionary) <= 1:
        # dictionary should contain meta_data, plus one or more other dataframes
        raise AbsCaptureError("Could not extract dataframes from zipfile")
    return dictionary


# === find ABS data based on search terms
# public
def find_rows(
    meta: DataFrame,
    search_terms: dict[str, str],
    exact: bool = False,
    regex: bool = False,
    verbose: bool = False,
) -> DataFrame:
    """Extract from meta the rows that match the search_terms.
    Arguments:
     - meta - pandas DataFrame of metadata from the ABS
     - search_terms - dictionary - {search_phrase: meta_column_name}
     - exact - bool - whether to match with == or .str.contains()
     - regex - bool - for .str.contains() - use regulare expressions
     - verbose - bool - print additional information when searching.
    Returns a pandas DataFrame (subseted from meta):"""

    meta_select = meta.copy()
    if verbose:
        print(f"In find_rows() {exact=} {regex=} {verbose=}")
        print(f"In find_rows() starting with {len(meta_select)} rows in the meta_data.")

    for phrase, column in search_terms.items():
        if verbose:
            print(f"Searching {len(meta_select)}: term: {phrase} in-column: {column}")

        pick_me = (
            (meta_select[column] == phrase)
            if (exact or column == metacol.table)
            else meta_select[column].str.contains(phrase, regex=regex)
        )
        meta_select = meta_select[pick_me]
        if verbose:
            print(f"In find_rows() have found {len(meta_select)}")

    if verbose:
        print(f"Final selection is {len(meta_select)} rows.")

    if len(meta_select) == 0:
        print("Nothing selected?")

    return meta_select


# public
def find_id(
    meta: DataFrame,
    search_terms: dict[str, str],
    exact=False,
    verbose: bool = False,
    validate_unique: bool = True,
) -> tuple[str, str]:
    """Get the ABS series identifier that matches the given
    search-terms. This is a more generalised search function than
    get_identifier() below.
    Arguments:
     - meta - pandas DataFrame of metadata from the ABS
     - search_terms - dictionary - {search_phrase: meta_column_name}
     - exact - bool - whether to match with == or .str.contains()
     - verbose - bool - print additional information when searching.
     - validate_unique - bool - apply assertion test to ensure only one match
    Returns a Tuple:
     - the ABS Series Identifier - str - which ws found using the search terms
     - units - str - unit of measurement."""

    if verbose:
        print(f"In find_id() {exact=} {verbose=} {validate_unique=}")

    meta_select = find_rows(meta, search_terms, exact=exact, verbose=verbose)
    if verbose and len(meta_select) != 1:
        print(meta_select)
    if validate_unique:
        assert len(meta_select) == 1
    return meta_select["Series ID"].values[0], meta_select["Unit"].values[0]


# public
def get_identifier(
    meta: DataFrame,
    data_item_description: str,
    series_type: str,
    table: str,
    verbose: bool = False,
) -> tuple[str, str]:
    """Get the ABS series identifier that matches the given
    data_item_description, series_type, table
    Arguments:
     - meta - pandas DataFrame of metadata from the ABS
     - data_item_description - string
     - series_type - string - typically one of "Original"
                              "Seasonally Adjusted" or "Trend"
     - table - string - ABS Table number - eg. '1' or '19a'
    Returns:
     - Tuple (id, units), where:
         - id - string - identifier for an ABS data series (column name)
         - units - string - unit of measurement."""

    search = {
        table: metacol.table,
        series_type: metacol.stype,
        data_item_description: metacol.did,
    }

    return find_id(meta, search, exact=False, verbose=verbose)


# === Get ABS data using catalogue numbers and series identifiers
# public
@cache
def get_abs_directory() -> pd.Series:
    """Return a Series of ABS landing pages for all
    ABS Catalogue numbers."""

    # get ABS web page of catalogue numbers
    url = "https://www.abs.gov.au/about/data-services/help/abs-time-series-directory"
    page = common.get_file(url, _CACHE_PATH, cache_name_prefix="ABS_DIRECTORY")
    links = pd.read_html(page, extract_links='body')[1]  # second table on the page

    # extract catalogue numbers
    cats = links['Catalogue Number'].apply(pd.Series)[0]

    # extract landing pages
    root = "https://www.abs.gov.au/statistics/"
    topics = links['Topic'].apply(pd.Series)[1].apply(str).str.replace(root, "")
    topics = topics[~topics.str.contains("http")]  # remove bad links
    landings = topics.str.split("/").apply(lambda x: AbsLandingPage( *(x[0:3])) )

    # combine and return
    cats = cats.loc[landings.index]
    landings.index = cats
    landings.index.name = "Catalogue Number"
    landings.name = "Landing Page"
    return landings


# public
def get_abs_series(
    cat_id: str,
    series_ids: Sequence[str]|str,
    **kwargs: Any,  # passed to get_abs_data()
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Get selected ABS selected items.
    Arguments:
     - cat_id - string - ABS catalogue number
     - series_ids - a string or a list of strings - series identifiers
     - kwargs - additional arguments for get_abs_data()
    Returns two data frames:
     - the first is the metadata for the series_ids
     - the second is the actual data for the series_ids
    Both dataframes will be empty if the series_ids are not found."""

    # get the ABS landing page directory, check we are in the directory
    landings = get_abs_directory()
    if cat_id not in landings.index:
        print(f"Catalogue number {cat_id} not found in ABS directory.")
        return pd.DataFrame(), pd.DataFrame()

    # get the key data from the ABS landing page
    landing_page = landings[cat_id]
    abs_dict = get_abs_data(landing_page, **kwargs)
    meta = abs_dict[META_DATA]
    if isinstance(series_ids, str):
        series_ids = [series_ids]

    # get the data for each series
    r_meta, r_data = {}, {}
    for series_id in series_ids:
        if meta[metacol.id].str.contains(series_id).sum() == 0:
            print(f"Could not find series {series_id} in the metadata.")
            continue
        series_meta = meta[meta[metacol.id] == series_id].iloc[0]
        r_meta[series_id] = series_meta
        table = series_meta[metacol.table]
        data = abs_dict[table][series_id].copy()
        r_data[series_id] = data
    
    return pd.DataFrame(r_meta), pd.DataFrame(r_data)


# === simplified plotting of ABS data ...
# public
def iudts_from_row(row: pd.Series) -> tuple[str, str, str, str, str]:
    """Return a tuple comrising series_id, units, data_description,
    table_number, series_type."""
    return (
        row[metacol.id],
        row[metacol.unit],
        row[metacol.did],
        row[metacol.table],
        row[metacol.stype],
    )


def longest_common_prefex(strings: list[str]) -> str:
    """Find the longest common string prefix."""
    num_strings: int = len(strings)

    # trivial cases
    if num_strings == 0:
        return ""
    if num_strings == 1:
        return strings[0]

    # harder cases
    broken = False
    for i in range(0, len(strings[0]) + 1):
        if i == len(strings[0]):
            break
        for j in range(1, num_strings):
            if i >= len(strings[j]) or strings[0][i] != strings[j][i]:
                broken = True
                break
        if broken:
            break

    return strings[0][:i]


def _column_name_fix(r_frame: DataFrame) -> tuple[DataFrame, str, list[str]]:
    """Shorten column names."""
    columns = r_frame.columns.to_list()
    title = longest_common_prefex(columns)
    renamer = {x: x.replace(title, "") for x in columns}
    r_frame = r_frame.rename(columns=renamer)
    renamer = {x: abbreviate(x) for x in r_frame.columns}
    colours = [state_colors[x] for x in renamer.values()]
    r_frame = r_frame.rename(columns=renamer)
    return r_frame, title, colours


def plot_rows_collectively(
    abs_dict: AbsDict,
    selector: dict[str, str],
    regex=False,  # passed to find_rows()
    verbose: bool = False,  # passed to find_rows()
    **kwargs: Any,  # passed to plotting function
) -> None:
    """Produce an collective/single chart covering each row
    selected from the meta data with selector.
    Agruments:
    - abs_dict - dict[str, DataFrame] - dictionary of ABS dataframes
    - selector - dict - used with find_rows() to select rows from meta
    - regex - bool - used with selector in find_rows()
    - verbose - bool - used for feedback from find_rows()
    - **kwargs - arguments passed to plotting function."""

    frame = DataFrame()
    for _, row in find_rows(
        abs_dict[META_DATA], selector, regex=regex, verbose=verbose
    ).iterrows():
        series_id, units, did, table, _ = iudts_from_row(row)
        frame[did.replace(" ;  ", ": ").replace(" ;", "")] = abs_dict[table][series_id]
    if len(frame) == 0:
        return

    r_frame, units = recalibrate(frame, units)
    r_frame, title, colours = _column_name_fix(r_frame)

    legend = {**LEGEND_SET, "ncols": 2, **(kwargs.pop("legend", {}))}
    line_plot(
        r_frame,
        title=title,
        ylabel=units,
        legend=legend,
        color=colours,
        **kwargs,
    )
    # end plot_rows_collectively()


def plot_rows_individually(
    abs_dict: AbsDict,
    selector: dict[str, str],
    plot_function: Callable,
    regex=False,  # passed to find_rows()
    verbose: bool = False,  # passed to find_rows()
    **kwargs: Any,  # passed to plotting function
) -> None:
    """Produce an single chart for each row selected from
    the meta data with selector.
    Agruments:
    - abs_dict - dict[str, DataFrame] - dictionary of ABS dataframes
    - selector - dict - used with find_rows() to select rows from meta
    - plot_function - callable - for plotting a series of dara
    - regex - bool - used with selector in find_rows()
    - verbose - bool - used for feedback from find_rows()
    - **kwargs - arguments passed to plotting function."""

    rows = find_rows(abs_dict[META_DATA], selector, regex=regex, verbose=verbose)
    for _, row in rows.iterrows():
        series_id, units, did, table, series_type = iudts_from_row(row)
        series, units = recalibrate(abs_dict[table][series_id], units)
        series.name = f"{series_type.capitalize()} series"

        plot_function(
            series,
            title=did.replace(" ;  ", ": ").replace(" ;", ""),
            ylabel=units,
            **kwargs,
        )


def plot_rows_seas_trend(
    abs_dict: AbsDict,
    selector: dict[str, str],
    regex=False,  # passed to find_rows()
    verbose: bool = False,  # passed to find_rows()
    **kwargs: Any,  # passed to plotting function
) -> None:
    """Produce an seasonal/Trend chart for the rows selected from
    the metadata with selector.
    Agruments:
    - abs_dict - dict[str, DataFrame] - dictionary of ABS dataframes
    - selector - dict - used with find_rows() to select rows from meta
      this needs to select both a Trend and Seasonally Adjusted row, and
      must exclude the "Series Type" column
    - regex - bool - used with selector in find_rows()
    - verbose - bool - used for feedback from find_rows()
    - **kwargs - arguments passed to plotting function."""

    # sanity checks - make sure seas/trend not in the selector
    if metacol.stype in selector.values():
        print(f'Check: unexpected column "{metacol.stype}" in the selector')
        return

    # identify the plot-sets using the selector ...
    st_data = {}
    for series_type in SEAS_ADJ, TREND:
        st_data[series_type] = find_rows(
            abs_dict[META_DATA],
            {**selector, series_type: metacol.stype},
            regex=regex,
            verbose=verbose,
        )

    # check plot-sets look reasonable
    if len(st_data[SEAS_ADJ]) != len(st_data[TREND]):
        print("The number of Trend and Seasonally Adjusted rows do not match")
        return
    if (
        not st_data[SEAS_ADJ][metacol.did].is_unique
        or not st_data[TREND][metacol.did].is_unique
    ):
        print("Data item descriptions are not unique")
        return

    # plot Seaspnal + Trend charts one-by-one
    for did in st_data[TREND][metacol.did]:
        # get data series
        frame_data = {}
        for row_type in SEAS_ADJ, TREND:
            row = st_data[row_type][st_data[row_type][metacol.did] == did].iloc[0]
            r_id, r_units, _, r_table, _ = iudts_from_row(row)
            frame_data[row_type] = abs_dict[r_table][r_id]

        # put the data into a frame and plot
        # Note - assume SA and Trend units are the same, this is not checked.
        frame, r_units = recalibrate(DataFrame(frame_data), r_units)
        seas_trend_plot(
            frame,  # cast(DataFrame, frame),
            title=did.replace(" ;  ", ": ").replace(" ;", ""),  # NEEDS THINKING
            ylabel=r_units,
            **kwargs,
        )


# === Select multiple series from different ABS datasets
# public - select an individual series
def get_single_series(
    selector: AbsSelectInput, verbose: bool = False
) -> AbsSelectOutput:
    """Return an ABS series for the specified selector."""

    if verbose:
        print(f"seeking: {selector.landing_page.topic}")

    # get the ABS data
    data_dict = get_abs_data(selector.landing_page, verbose=verbose)
    _, _, cat_id, meta = get_fs_constants(data_dict, selector.landing_page)
    data = data_dict[selector.table]

    # get the specific series we want to plot
    search_terms = {
        selector.table: metacol.table,
        {"SA": SEAS_ADJ, "Orig": ORIG}[selector.orig_sa]: metacol.stype,
        selector.search1: metacol.did,
        selector.search2: metacol.did,
    }
    if verbose:
        print(f"Search terms:\n{search_terms}")

    series_id, unit = find_id(meta, search_terms, verbose=verbose)
    series = data[series_id].copy(deep=True)
    if selector.calc_growth:
        periods = 4 if cast(pd.PeriodIndex, series.index).freqstr[0] == "Q" else 12
        series = (series / series.shift(periods) - 1) * 100.0

    return AbsSelectOutput(
        series=series,
        cat_id=cat_id,
        table=selector.table,
        series_id=series_id,
        unit=unit,
        orig_sa=selector.orig_sa,
        abbr=selector.abbr,
    )


# public - select multiple series
def get_multi_series(
    selection_dict: AbsSelectionDict, verbose: bool = False
) -> AbsMultiSeries:
    """Return a dictionary of Series data from the ABS,
    One series for each item in the selection_dict dictionary."""

    pool = {}
    for name, selector in selection_dict.items():
        if verbose:
            print("-----------------")
        pool[name] = get_single_series(selector, verbose)
    return pool


# public - convert an AbsMultiSeries into a DataFrame
def df_from_ams(ams: AbsMultiSeries) -> pd.DataFrame:
    """Get a dataframe from the ABS Multi Series item."""

    frame_dict = {key: val.series for key, val in ams.items()}
    for name, series in frame_dict.items():
        if cast(pd.PeriodIndex, series.index).freqstr[0] == "Q":
            series_m = qtly_to_monthly(series, interpolate=False)
            frame_dict[name] = series_m
    return pd.DataFrame(frame_dict)


# public
def rename_cols_with_stype(df: pd.DataFrame, ams: AbsMultiSeries) -> pd.DataFrame:
    """Add series type to the column names of a DataFrame."""

    names = {}
    for col in df.columns:
        orig_sa = ams[col].orig_sa  # assume happy case only
        orig_sa = f" ({orig_sa})" if orig_sa else orig_sa
        names[col] = f"{col}{orig_sa}"
    return df.rename(columns=names)
