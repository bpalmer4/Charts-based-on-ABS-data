"""Get data from the Australian Bureau of Statistics (ABS).

Our general approach here is to:

1. Download the "latest-release" webpage from the ABS.

2. Parse that webpage to find the link to the download
   all-tables zip-file. We do this because the name of
   the file location on the ABS server changes from
   month to month.

3. Get the URL headers for this file, amd compare freshness
   with the version in the cache directory (if any).

4. Use either the zip-file from the cache, or download
   the zip-file from the ABS andsave it to the cache,
   and use that.

5. Open the zip-file, and save each table to a pandas
   DataFrame with a PeriodIndex. And save the metadata
   to a pandas DataFrame. Return all of the DataFrames
   in a dictionary."""

# standard library imports
import calendar
import io
import re
import zipfile
from collections import namedtuple
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Callable, Final, TypeVar, cast

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

# --- Some useful constants

# typing information
# public
AbsDict = dict[str, pd.DataFrame]

# private
_DataT = TypeVar("_DataT", Series, DataFrame)  # python 3.11+
SEAS_ADJ: Final[str] = "Seasonally Adjusted"
TREND: Final[str] = "Trend"


# to get to an ABS landing page ...
# public
@dataclass(frozen=True)
class AbsLandingPage:
    """Class for selecting ABS data files to download."""
    theme: str
    parent_topic: str
    topic: str


# columns in the meta data DataFrame
# private
_META_DATA: Final[str] = "META_DATA"
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

# public
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


# --- utility functions


# public
def get_fs_constants(
    abs_dict: AbsDict,
    landing_page: AbsLandingPage,
) -> tuple[str, str, str, pd.DataFrame]:
    """Get file system constants for a catalogue ID."""

    cat_id = abs_dict[_META_DATA][metacol.cat].unique()[0]
    source = f"ABS {cat_id}"
    chart_dir = f"./CHARTS/{cat_id} - {landing_page.topic}/"
    Path(chart_dir).mkdir(parents=True, exist_ok=True)
    return source, chart_dir, cat_id, abs_dict[_META_DATA]


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

    recency_period = 6  # years
    recency_extra = 3  # months
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


# --- Data fetch from the ABS
# private
def _get_abs_page(page: AbsLandingPage):
    """Return the HTML for the ABS topic landing page."""

    head = "https://www.abs.gov.au/statistics/"
    tail = "/latest-release"
    url = f"{head}{page.theme}/{page.parent_topic}/{page.topic}{tail}"
    return common.request_get(url)


# private
def _get_url_iteration(soup: BeautifulSoup, search_terms: list[str]) -> list[str]:
    """Search through webpage (in BS4 format) for search terms
    within hyperlimk-anchors.  Return a list of matching link URLs."""

    url_list: list[str] = []
    for term in search_terms:
        text = re.compile(term, re.IGNORECASE)
        found = soup.findAll("a", text=text)
        if not found or len(found) == 0:
            continue
        for element in found:
            result = re.search(r'href="([^ ]+)"', str(element.prettify))
            if result is not None:
                url = result.group(1)
                url_list.append(url)
    return url_list


# private
def _get_urls(page: bytes, table: int, verbose: bool) -> str | list[str]:
    """Scrape a webpage for the ZIP file from the ABS page.
    If the ZIP file cannot be located, scrape a list of
    URLs for the individual excel files."""

    # remove those pesky span tags
    page = re.sub(b"<span[^>]*>", b" ", page)
    page = re.sub(b"</span>", b" ", page)
    page = re.sub(b"\\s+", b" ", page)  # tidy up white space

    # get a single all-table URL from the web page
    soup = BeautifulSoup(page, features="lxml")
    search_terms = ["Download All", "Download ZIP"]
    url_list = _get_url_iteration(soup, search_terms)
    if verbose:
        print("Length of URL list: ", len(url_list))
        print(f"Selecting {table} from list: {url_list}")
    if isinstance(url_list, list) and len(url_list) > table:
        print("Found URL for a ZIP file on ABS web page")
        url = url_list[table]
        if verbose:
            print(f"-1--> {url}")
        return url  # of type str

    # get a list of individual table URLs
    print("Did not find the URL for a ZIP file")
    search_terms = ["download.xlsx"]
    url_list = _get_url_iteration(soup, search_terms)
    if not url_list or not isinstance(url_list, list):
        print("Could not fimd individual urls")
        raise common.HttpError("No URLs found on web-page for downloading data")

    print("URL list of excel files identified")
    return url_list  # of type list


# private
def _prefix_url(url: str) -> str:
    """Apply ABS URL prefix to relative links."""
    prefix = "http://www.abs.gov.au"
    url = url.replace(prefix, "")
    return f"{prefix}{url}"


# private
def _get_abs_zip_file(landing_page: AbsLandingPage, table: int, verbose: bool) -> bytes:
    """Get the latest zip_file of all tables for
    a specified ABS catalogue identifier"""

    # get relevant web-page from ABS website
    text_page = _get_abs_page(landing_page)

    # extract web address
    url = _get_urls(text_page, table, verbose)

    # get direct from ABS and cache for future use
    if isinstance(url, list):
        # url is a list of individual spreedsheets
        # We need to fake up a zip file from these spreadsheets ...
        print("The ABS is being difficult, we need to fake up a zip file")
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for u in url:
                u = _prefix_url(u)
                file_bytes = common.get_file(u, _CACHE_PATH)
                name = Path(u).name
                zip_file.writestr(f"/{name}", file_bytes)
        zip_buf.seek(0)
        returnable = zip_buf.read()
    else:
        # url is for a single zip file ...
        url = _prefix_url(url)
        returnable = common.get_file(url, _CACHE_PATH)

    return returnable


# private
def _get_meta(
    excel: pd.ExcelFile, tab_num: str, tab_desc: str, cat_id: str
) -> pd.DataFrame:
    """Capture the metadata from the Index sheet of an ABS excel file.
    Returns a DataFrame specific to the current excel file."""

    file_meta = excel.parse(
        "Index",
        header=9,
        parse_dates=True,
        infer_datetime_format=True,
        converters={"Unit": str},
    )
    file_meta = file_meta.iloc[1:-2]  # drop first and last 2
    file_meta = file_meta.dropna(axis="columns", how="all")
    file_meta["Unit"] = (
        file_meta["Unit"]
        .str.replace("000 Hours", "Thousand Hours")
        .replace("000,000", "Millions")
        .replace("000", "Thousands")
    )
    file_meta[metacol.table] = tab_num.strip()
    file_meta[metacol.tdesc] = tab_desc
    file_meta[metacol.cat] = cat_id
    return file_meta


# private
def _get_data(
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
            parse_dates=True,
            infer_datetime_format=True,
        )

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
                print(f"Warning: no data for {i}\n{problematic}\n\n")

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
                cast(pd.DatetimeIndex, data.index).month.max()
            ].upper()
            freq = f"{freq}-{month}"
        data = data.to_period(freq=freq)
    return data


# private
def _get_dataframes(zip_file: bytes, verbose: bool) -> AbsDict:
    """Get a DataFrame for each table in the zip-file,
    plus an overall DataFrame for the metadata.
    Return these in a dictionary
    Arguments:
     - zip_file - bytes array of ABS zip file of excel spreadsheets
     - verbose - provide additional feedback on this step.
    Returns:
     - either an empty dictionary (failure) or a dictionary containing
       a separate DataFrame for each table in the zip-file,
       plus a DataFrame called 'META' for the metadata.
    """

    freq_dict = {"annual": "Y", "quarter": "Q", "month": "M"}

    print("Extracting DataFrames from the zip-file ...")
    returnable: dict[str, DataFrame] = {}
    meta = DataFrame()
    with zipfile.ZipFile(io.BytesIO(zip_file)) as zipped:
        for element in zipped.infolist():
            # We get a new pandas DataFrame for every excel file.

            # get the zipfile into pandas
            excel = pd.ExcelFile(io.BytesIO(zipped.read(element.filename)))

            # get table information
            if "Index" not in excel.sheet_names:
                print(
                    'Caution: Could not find the "Index" '
                    f"sheet in {element.filename}"
                )
                continue
            file_meta = excel.parse("Index", nrows=8)
            cat_id = file_meta.iat[3, 1].split(" ")[0].strip()
            table = file_meta.iat[4, 1]
            splat = table.split(".")
            tab_num = splat[0].split(" ")[-1].strip()
            tab_desc = ".".join(splat[1:]).strip()

            # get the metadata
            file_meta = _get_meta(excel, tab_num, tab_desc, cat_id)

            # establish freq - used for making the index a PeriodIndex
            freq = file_meta["Freq."].str.lower().unique().tolist()
            freq = (
                freq_dict[freq[0]] if len(freq) == 1 and freq[0] in freq_dict else None
            )
            if freq is None:
                print(f"Unrecognised data frequency for {table}")

            # fix tabulation when ABS uses the same table numbers for Qrtly and Mthly data
            # which it does, for example, in the experimental household spending indicator
            if tab_num in returnable:
                tab_num += freq
                file_meta["Table"] = tab_num

            # aggregate the meta data
            meta = pd.concat([meta, file_meta])

            # add the table to the returnable dictionary
            returnable[tab_num] = _get_data(excel, file_meta, freq, verbose)

    returnable[_META_DATA] = meta
    return returnable


# public
@cache
def get_abs_data(
    landing_page: AbsLandingPage, table: int = 0, verbose: bool = False
) -> dict[str, DataFrame]:
    """For the relevant ABS page return a dictionary containing
    a meta-data Data-Frame and one or more DataFrames of actual
    data from the ABS.
    Arguments:
     - page - class ABS_topic_page - desired time_series page in
              the format:
              abs.gov.au/statistics/theme/parent-topic/topic/latest-release
     - table - select the zipfile to return in order as it
               appears on the ABS webpage - default=0
               (e.g. 6291 has four possible tables,
               but most ABS pages only have one).
     - verbose - display detailed web-scraping and caching information"""

    zip_file = _get_abs_zip_file(landing_page, table, verbose)
    if not zip_file:
        raise TypeError("An unexpected empty zipfile.")
    dictionary = _get_dataframes(zip_file, verbose)
    if len(dictionary) <= 1:
        # dictionary should contain meta_data, plus one or more other dataframes
        raise TypeError("Could not extract dataframes from zipfile")
    return dictionary


# --- identify the specific data series from the meta data DataFrame
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

    for phrase, column in search_terms.items():
        if verbose:
            print(
                f"Searching {len(meta_select)}: " f"term: {phrase} in-column: {column}"
            )
        pick_me = (
            (meta_select[column] == phrase)
            if (exact or column == metacol.table)
            else meta_select[column].str.contains(phrase, regex=regex)
        )
        meta_select = meta_select[pick_me]

    if verbose:
        print(len(meta_select))

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

    meta_select = find_rows(meta, search_terms, exact, verbose)
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

    return find_id(meta, search, exact=True, verbose=verbose)


# --- simplified plotting of ABS data ...
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
    abs_dict: dict[str, DataFrame],
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
        abs_dict[_META_DATA], selector, regex=regex, verbose=verbose
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
    abs_dict: dict[str, DataFrame],
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

    rows = find_rows(abs_dict[_META_DATA], selector, regex=regex, verbose=verbose)
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
    abs_dict: dict[str, DataFrame],
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
            abs_dict[_META_DATA],
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
            title=did.replace(" ;  ", ": ").replace(" ;", ""),
            ylabel=r_units,
            **kwargs,
        )
