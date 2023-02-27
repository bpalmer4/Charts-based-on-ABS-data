"""Get data from the Australian Bureau of Statistics (ABS).

Our general approach here is to:

1. Download the "latest-release" webpage from the ABS
   for known ABS catalogue numbers.

2. Parse that webpage to find the link to the download
   all-tables zip-file. We do this because the name of
   the file location on the ABS server changes from
   month to month.

3. Check to see whether I have cached that file previously,
   if not, download and save the zip-file to the cache.

4. Open the zip-file, and save each table to a pandas
   DataFrame with a PeriodIndex. And save the metadata
   to a pandas DataFrame. Return all of the DataFrames
   in a dictionary."""


# standard library imports
import io
import re
from pathlib import Path
import zipfile

# analytical imports
import pandas as pd
from bs4 import BeautifulSoup
import arrow

# local imports
import common


# --- ABS catalgue map - these are the possible downloads we know about
ABS_data_map: dict[str, dict[str, str]] = {
    "3101": {
        "Name": "National, State and Territory " "Estimated Resident Population",
        "URL": "https://www.abs.gov.au/statistics/"
        "people/population/national-state-"
        "and-territory-population/latest-release",
    },
    "5206": {
        "Name": "Australian National Accounts: "
        "National Income Expenditure and Product",
        "URL": "https://www.abs.gov.au/statistics/"
        "economy/national-accounts/australian-national-accounts-"
        "national-income-expenditure-and-product/latest-release",
    },
    "5232": {
        "Name": "Australian National Accounts: Finance and Wealth",
        "URL": "https://www.abs.gov.au/statistics/economy/"
        "national-accounts/australian-national-accounts"
        "-finance-and-wealth/latest-release",
    },
    "5676": {
        "Name": "Business Indicators, Australia",
        "URL": "https://www.abs.gov.au/statistics/economy/"
        "business-indicators/business-indicators-australia/"
        "latest-release",
    },
    "6150": {
        "Name": "Labour Account, Australia",
        "URL": "https://www.abs.gov.au/statistics/"
        "labour/labour-accounts/"
        "labour-account-australia/latest-release",
    },
    "6202": {
        "Name": "Labour Force, Australia",
        "URL": "https://www.abs.gov.au/statistics/"
        "labour/employment-and-unemployment/"
        "labour-force-australia/latest-release",
    },
    "6291": {
        "Name": "Labour Force, Australia, Detailed",
        "URL": "https://www.abs.gov.au/statistics/"
        "labour/employment-and-unemployment/"
        "labour-force-australia-detailed/latest-release",
    },
    "6354": {
        "Name": "Job Vacancies, Australia",
        "URL": "https://www.abs.gov.au/statistics/"
        "labour/employment-and-unemployment/"
        "job-vacancies-australia/latest-release",
    },
    "6345": {
        "Name": "Wage Price Index, Australia",
        "URL": "https://www.abs.gov.au/statistics/"
        "economy/price-indexes-and-inflation/"
        "wage-price-index-australia/latest-release",
    },
    "6401": {
        "Name": "Consumer Price Index, Australia",
        "URL": "https://www.abs.gov.au/statistics/economy/"
        "price-indexes-and-inflation/"
        "consumer-price-index-australia/latest-release",
    },
    "6427": {
        "Name": "Producer Price Indexes, Australia",
        "URL": "https://www.abs.gov.au/statistics/"
        "economy/price-indexes-and-inflation/"
        "producer-price-indexes-australia/latest-release",
    },
    "6484": {
        "Name": "Monthly CPI Indicator, Australia",
        "URL": "https://www.abs.gov.au/statistics/"
        "economy/price-indexes-and-inflation/"
        "monthly-consumer-price-index-indicator/latest-release",
    },
    "8501": {
        "Name": "Retail Trade, Australia",
        "URL": "https://www.abs.gov.au/statistics/industry/"
        "retail-and-wholesale-trade/retail-trade-australia/"
        "latest-release",
    },
    "8731": {
        "Name": "Building Approvals, Australia",
        "URL": "https://www.abs.gov.au/statistics/industry/"
        "building-and-construction/"
        "building-approvals-australia/latest-release",
    },
    "8752": {
        "Name": "Building Activity, Australia",
        "URL": "https://www.abs.gov.au/statistics/industry/"
        "building-and-construction/"
        "building-activity-australia/latest-release",
    },
}

# --- initialisation


# private
def _check_abs_data_map(data_map: dict[str, dict[str, str]]) -> None:
    """Check the integrity of the ABS_data_map."""

    for data in data_map.values():
        assert "Name" in data
        assert "URL" in data


_check_abs_data_map(ABS_data_map)


# private
def _establish_cache_directory() -> str:
    """Establish the ABS cache directory in the file system."""

    cache_dir = "./ABS_CACHE/"
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return cache_dir


_CACHE_DIR = _establish_cache_directory()


# --- utility functions
# public
_META_DATA = "META_DATA"


def get_fs_constants(catalogue_id: str) -> tuple[str, str, str]:
    """Get file system constants for a catalogue ID."""

    assert catalogue_id in ABS_data_map  # sanity check
    source = f"ABS {catalogue_id}"
    chart_dir = f"./CHARTS/{catalogue_id}/"
    Path(chart_dir).mkdir(parents=True, exist_ok=True)
    return source, chart_dir, _META_DATA


# public
def clear_cache() -> None:
    """Clear the cache directory of zip and xlsx files."""

    extensions = ("*.zip", "*.ZIP", "*.xlsx", "*.XLSX")
    for extension in extensions:
        for fs_object in Path(_CACHE_DIR).glob(extension):
            if fs_object.is_file():
                fs_object.unlink()


# public
def get_ABS_catalogue_IDs() -> dict[str, str]:
    """Return a dictionary of known ABS catalogue identifiers."""

    response = {}
    for identifer, data in ABS_data_map.items():
        response[identifer] = data["Name"]
    return response


# public
def get_plot_constants(meta: pd.DataFrame) -> tuple[pd.Timestamp, list, list]:
    """Get plotting constants from ABS meta data table
    - used in a loop to produce a plot of the full
      series, and a plot of the recent period."""

    recency_period = 5  # years
    recency_extra = 3  # months
    today = pd.Timestamp("today")
    reasonable_end = meta["Series End"][meta["Series End"] <= today]
    reasonable_end = reasonable_end.max() if len(reasonable_end) > 0 else today
    recent = reasonable_end - pd.DateOffset(years=recency_period, months=recency_extra)
    plot_times = [None, recent]
    plot_tags = ("full", "recent")
    return recent, plot_times, plot_tags


# --- Data fetch from the ABS
# private
def _get_abs_webpage(catalogue_id: str) -> bytes | None:
    """Get the ABS web page for latest data in respect
    of a specified ABS catalogue identifier."""

    if catalogue_id not in ABS_data_map:
        print(f"Catalogue identifier not recognised: {catalogue_id}")
        return None
    url = ABS_data_map[catalogue_id]["URL"]
    return common.request_get(url)


# private
def _get_cache_contents(file: Path) -> bytes | None:
    """Get cache_contents for a particular file.
    Erase the cache file if it is stale.
    Return None if cache file not found or is stale."""

    if not file.is_file():
        return None  # no such zip-file

    # sometimes the ABS does not sufficiently differentiate
    # file names over time, so we use the concept of staleness
    # to ensure we have fresh files, without overburdening the
    # ABS servers.
    stale = 1  # only use cache files less than stale days old
    fresh_time = arrow.now().shift(days=-stale)
    file_time = arrow.get(file.stat().st_mtime)
    if file_time > fresh_time:
        print("Retrieving zip-file from cache ...")
        zip_file = file.read_bytes()
        return zip_file  # zip-file acquired

    print("Cache looks stale: Removing old cache version")
    file.unlink()
    return None  # zip-file is old and stale


# private
def _get_url_iteration(soup, search_terms):
    url_list = []
    for term in search_terms:
        text = re.compile(term, re.IGNORECASE)
        found = soup.findAll("a", text=text)
        if not found or len(found) == 0:
            continue
        for element in found:
            url = re.search(r'href="([^ ]+)"', str(element.prettify)).group(1)
            url_list.append(url)
    return url_list


# private
def _get_urls(page: bytes, table: int, verbose: bool) -> None | str | list[str]:
    """Scrape a URL for the ZIP file from the ABS page.
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
        if verbose:
            print(f"-2--> {url_list}")
        return None
    print("URL list of excel files identified")
    if verbose:
        print(f"-3--> {url_list}")
    return url_list  # of type list


# private
def _get_zip_from_cache(
    url: str | list[str], prefix: str, verbose: bool
) -> tuple[Path, bytes | None]:
    """Get a zip file from the cache if it is there and not stale."""

    stem = (
        f"{url[0] if isinstance(url, list) else url}".replace(prefix, "")
        .replace("/", "_")
        .replace(".xlsx", ".zip")
        .replace(".XLSX", ".zip")
    )
    cache_name = f"{_CACHE_DIR}{stem}"
    if verbose:
        print(f"Cache file name: {cache_name}")
    zip_file = _get_cache_contents(cache_path := Path(cache_name))
    return cache_path, zip_file


# private
def _get_xlsx_from_abs(
    url_list: list, prefix: str, cache_path: Path, verbose: bool
) -> bytes | None:
    """Get each of individual .xlsx files and put in a
    zip-file. Return that zip-file as bytes"""

    # get the individual xl-table data from ABS
    xl_dict = {}
    for url in url_list:
        url = url.replace(prefix, "")
        url = f"{prefix}{url}"
        name = Path(url).name
        xl_dict[name] = common.request_get(url)
    if verbose:
        print(f"Captured: {xl_dict.keys()}")

    # build a cache file ...
    with zipfile.ZipFile(cache_path, "w", zipfile.ZIP_DEFLATED) as zfile:
        for name, contents in xl_dict.items():
            if not contents:
                print(f"Something odd happened when zipping {name}")
                continue
            zfile.writestr(
                f"/{name}",
                contents,
            )

    # return the zip-file
    zip_file = _get_cache_contents(cache_path)
    if zip_file is None:
        print("Unexpected error: the written zip-file is not there?")
    else:
        if verbose:
            print(f"Zipfile is {len(zip_file):,} bytes long.")
    return zip_file


# private
def _get_zip_from_abs(
    url: str, prefix: str, cache_path: Path, verbose: bool
) -> bytes | None:
    """Get zip-file from the ABS and place into the cache."""

    # get zip-file from ABS website
    url = prefix + url
    print("We need to download this file from the ABS ...")
    if verbose:
        print(url)
    zip_file = common.request_get(url)
    if zip_file is None:
        return None

    # cache for next time and return
    print("Saving ABS download to cache.")
    cache_path.open(mode="w", buffering=-1, encoding=None, errors=None, newline=None)
    cache_path.write_bytes(zip_file)
    return zip_file


# private
def _get_abs_zip_file(catalogue_id: str, table: int, verbose: bool) -> bytes | None:
    """Get the latest zip_file of all tables for
    a specified ABS catalogue identifier"""

    # get relevant web-page from ABS website
    page = _get_abs_webpage(catalogue_id)
    if not page:
        print(f"Failed to retrieve ABS web page for {catalogue_id}")
        return None

    # extract web address
    url = _get_urls(page, table, verbose)
    if not url:
        print("No URL found for data")
        return None

    # get from cache:
    prefix = "https://www.abs.gov.au"
    cache_path, zip_file = _get_zip_from_cache(url, prefix, verbose)
    if zip_file:
        return zip_file

    # get direct from ABS and cache for future use
    returnable = None
    if isinstance(url, list):
        returnable = _get_xlsx_from_abs(url, prefix, cache_path, verbose)
    else:
        returnable = _get_zip_from_abs(url, prefix, cache_path, verbose)
    return returnable


# private
def _get_meta(
    excel: pd.ExcelFile, tab_num: str, tab_desc: str, meta: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Capture the metadata from the Index sheet of an ABS excel file.
    Returns a tuple of two DataFrames. The first Dataframe (meta) is the
    cumulative meta data for all excel files in an ABS zip download.
    The second dataframe is the meta data specific to the current
    excel file."""

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
    file_meta["Table"] = tab_num.strip()
    file_meta["Table Description"] = tab_desc
    if meta is None:
        meta = file_meta
    else:
        meta = pd.concat([meta, file_meta])
    return meta, file_meta


# private
def _get_data(
    excel: pd.ExcelFile, meta: pd.DataFrame, freq: str, verbose: bool
) -> pd.DataFrame:
    """Take an ABS excel file and put all the Data sheets into a single
    pandas DataFrame and return that DataFrame."""

    months = {
        1: "JAN",
        2: "FEB",
        3: "MAR",
        4: "APR",
        5: "MAY",
        6: "JUN",
        7: "JUL",
        8: "AUG",
        9: "SEP",
        10: "OCT",
        11: "NOV",
        12: "DEC",
    }
    data = pd.DataFrame()
    data_sheets = [x for x in excel.sheet_names if x.startswith("Data")]
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
            month = months[data.index.month.max()]
            freq = f"{freq}-{month}"
        data = data.to_period(freq=freq)
    return data


# private
def _get_dataframes(zip_file: bytes, verbose: bool) -> None | dict[str, pd.DataFrame]:
    """Get a DataFrame for each table in the zip-file,
    plus an overall DataFrame for the metadata.
    Return these in a dictionary
    Arguments:
     - zip_file - bytes array of ABS zip file of excel spreadsheets
     - verbose - provide additional feedback on this step.
    Returns:
     - either None (failure) or a dictionary containing a
       separate DataFrame for each table in the zip-file,
       plus a DataFrame called 'META' for the metadata.
    """

    freq_dict = {"annual": "Y", "quarter": "Q", "month": "M"}

    print("Extracting DataFrames from the zip-file ...")
    returnable = {}
    meta = pd.DataFrame()
    with zipfile.ZipFile(io.BytesIO(zip_file)) as zipped:
        for element in zipped.infolist():
            # We get a new pandas DataFrame for every excel file.

            # get the zipfile into pandas
            excel = pd.ExcelFile(zipped.read(element.filename))

            # get table information
            if "Index" not in excel.sheet_names:
                print(
                    'Caution: Could not find the "Index" '
                    f"sheet in {element.filename}"
                )
                continue
            file_meta = excel.parse("Index", nrows=8)
            table = file_meta.iat[4, 1]
            splat = table.split(".")
            tab_num = splat[0].split(" ")[-1].strip()
            tab_desc = ".".join(splat[1:]).strip()

            # get the metadata
            meta, file_meta = _get_meta(excel, tab_num, tab_desc, meta)

            # establish freq - used for making the index a PeriodIndex
            freq = file_meta["Freq."].str.lower().unique()
            freq = (
                freq_dict[freq[0]] if len(freq) == 1 and freq[0] in freq_dict else None
            )
            if freq is None:
                print(f"Unrecognised data frequency for {table}")

            # get the actual data
            returnable[tab_num] = _get_data(excel, meta, freq, verbose)

    returnable[_META_DATA] = meta
    return returnable


# public
def get_ABS_meta_and_data(
    catalogue_id: str, table: int = 0, verbose: bool = False
) -> None | dict[str, pd.DataFrame]:
    """For the relevant catalogue-ID return a dictionary containing
    a meta-data Data-Frame and one or more DataFrames of actual
    data from the ABS.
    Arguments:
     - catalogue_id - string - ABS catalogue number for the
                      desired dataset.
     - table - select the zipfile to return in order as it
               appears on the ABS webpage - default=0
               (e.g. 6291 has four possible tables,
               but most ABS pages only have one).
     - verbose - display detailed web-scraping and caching information"""

    zip_file = _get_abs_zip_file(catalogue_id, table, verbose)
    if zip_file is None:
        return None
    return _get_dataframes(zip_file, verbose)


# --- identify the specific data series from the meta data DataFrame


# public
def find_id(
    meta: pd.DataFrame,
    search_terms: dict[str, str],
    exact=False,
    verbose: bool = False,
    validate_unique: bool = True,
) -> tuple[str]:
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

    meta_select = meta.copy()
    for phrase, column in search_terms.items():
        if verbose:
            print(
                f"Searching {len(meta_select)}: " f"term: {phrase} in-column: {column}"
            )
        if exact or column == "Table":  # always match the table column exactly
            meta_select = meta_select[meta_select[column] == phrase]
        else:
            meta_select = meta_select[
                meta_select[column].str.contains(phrase, regex=False)
            ]
    if verbose:
        print(len(meta_select))
    if verbose and len(meta_select) != 1:
        print(meta_select)
    if validate_unique:
        assert len(meta_select) == 1
    return meta_select["Series ID"].values[0], meta_select["Unit"].values[0]


# public
def get_identifier(
    meta: pd.DataFrame,
    data_item_description: str,
    series_type: str,
    table: str,
    verbose: bool = False,
) -> tuple[str]:
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
        table: "Table",
        series_type: "Series Type",
        data_item_description: "Data Item Description",
    }

    return find_id(meta, search, exact=True, verbose=verbose)