# abs_common.py

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.units as munits
import statsmodels.formula.api as smf

import requests
from bs4 import BeautifulSoup
import re
from pathlib import Path
import zipfile as zip
import io
import arrow

from typing import Tuple, Optional, Dict, List, Union
from operator import mul, truediv

# --- utility functions
META_DATA = 'META_DATA'

def get_fs_constants(catalogue_id):
    source = f'ABS {catalogue_id}'
    CHART_DIR = f"./CHARTS/{catalogue_id}/"
    Path(CHART_DIR).mkdir(parents=True, exist_ok=True)
    return source, CHART_DIR, META_DATA


def get_plot_constants(meta):
    """Get plotting constants"""
    
    RECENCY_PERIOD = 6 # years
    RECENCY_EXTRA = 3 # months
    RECENT = (
        meta['Series End'].max() 
        - pd.DateOffset(years=RECENCY_PERIOD, 
                        months=RECENCY_EXTRA)
    )
    plot_times = [None, RECENT]
    plot_tags = ('full', 'recent')
    return RECENT, plot_times, plot_tags


# --- Data fetch from the ABS

"""
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
   in a dictionary.
"""

# -- Establish an ABS cache directory
CACHE_DIR = "./ABS_CACHE/"
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

# -- URLs for getting key data sets from the ABS
ABS_data_map = {
    # map of catalogue identifiers to names and URLs

    "3101": {
        "Name": "National, State and Territory "
                "Estimated Resident Population",
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
        "Name": "Australian National Accounts: "
                "Finance and Wealth",
        "URL": "https://www.abs.gov.au/statistics/economy/"
               "national-accounts/australian-national-accounts"
               "-finance-and-wealth/latest-release",
    },
    
    "6202": {
        "Name": "Labour Force Australia",
        "URL": "https://www.abs.gov.au/statistics/"
               "labour/employment-and-unemployment/"
               "labour-force-australia/latest-release",
    },
    
    "6291": {
        "Name" : "Labour Force, Australia, Detailed",
        "URL":   "https://www.abs.gov.au/statistics/"
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

    "LAA": {
        "Name": "Labour Account Australia",
        "URL": "https://www.abs.gov.au/statistics/"
               "labour/employment-and-unemployment/"
               "labour-account-australia/latest-release",
    },
}

# Quick sanity check on the above data map
for data in ABS_data_map.values():
    assert "Name" in data
    assert "URL" in data

    
# public
def clear_cache():
    """Clear the cache directory of zip and xlsx files."""
    
    EXTENSIONS= ("*.zip", "*.ZIP", "*.xlsx", "*.XLSX")
    for extension in EXTENSIONS:
        for fs_object in Path(CACHE_DIR).glob(extension):
            if fs_object.is_file():
                fs_object.unlink() 

    
# public
def get_ABS_catalogue_IDs() -> Dict[str, str]:
    """Return a dictionary of known ABS catalogue identifiers."""
    
    response = {}
    for id, data in ABS_data_map.items():
        response[id] = data["Name"]
    return response


# private
def _get_url_contents(url:str) -> Optional[bytes]:
    """Get contents at URL, return None if it is not gettable"""
    
    gotten = requests.get(url, allow_redirects=True)
    if gotten.status_code != 200:
        print(f'Could not get ({url}), error code: {code}')
        return None    
    return gotten.content


# private
def _get_ABS_webpage(catalogue_id:str) -> Optional[bytes]:
    """Get the ABS web page for latest data in respect 
       of a specified ABS catalogue identifier."""

    if catalogue_id not in ABS_data_map:
        print(f"Catalogue identifier not recognised: {catalogue_id}")
        return None
    url = ABS_data_map[catalogue_id]["URL"]
    return _get_url_contents(url)


# private
def _get_cache_contents(file:Path) -> Optional[bytes]:
    """Get cache_contents for a particular file.
       Erase the cache file if it is stale.
       Return None if cache file not found or is stale."""
    
    if not file.is_file():
        return None # no such zip-file
    
    # sometimes the ABS does not sufficiently differentiate 
    # file names over time, so we use the concept of staleness
    # to ensure we have fresh files, without overburdening the
    # ABS servers.
    STALE = 1 # only use cache files less than STALE days old
    fresh_time = arrow.now().shift(days=-STALE)
    file_time = arrow.get(file.stat().st_mtime)
    if file_time > fresh_time:
        print('Retrieving zip-file from cache ...')
        zip_file = file.read_bytes()
        return zip_file # zip-file acquired

    print('Cache looks stale: Removing old cache version')
    file.unlink()
    return None # zip-file is old and stale


# private
def _get_url_iteration(soup, search_terms):
    
    url_list = []
    for term in search_terms:
        text=re.compile(term, re.IGNORECASE)
        found = soup.findAll('a', text=text)
        if not found or len(found) == 0:
            continue
        for element in found:
            url = re.search(r'href="([^ ]+)"', 
                            str(element.prettify)).group(1)
            url_list.append(url)
    return url_list


# private
def _get_urls(page:bytes, table:int, verbose:bool
    ) -> Optional[Union[str, List[str]]]:
    """Scrape a URL for the ZIP file from the ABS page.
       If the ZIP file cannot be located, scrape a list of
       URLs for the individual excel files."""
    
    # remove those pesky span tags
    page = re.sub(b'<span[^>]*>', b' ', page)
    page = re.sub(b'</span>', b' ', page)
    page = re.sub(b'\s+', b' ', page) # tidy up white space
    
    # get a single all-table URL from the web page
    soup = BeautifulSoup(page, features="lxml")
    search_terms = ['Download All', 'Download ZIP']
    url_list = _get_url_iteration(soup, search_terms)
    if verbose: print(f'Selecting {table} from list: {url_list}')
    if isinstance(url_list, list) and len(url_list) > table:
        print('Found URL for a ZIP file on ABS web page')
        url = url_list[table]
        if verbose: print(f'-1--> {url}')
        return url # of type str

    # get a list of individual table URLs
    print('Did not find the URL for a ZIP file')
    search_terms = ['download.xlsx']
    url_list = _get_url_iteration(soup, search_terms)
    if not url_list or not isinstance(url_list, list):
        print('Could not fimd individual urls')
        if verbose: print(f'-2--> {url_list}')
        return None
    print('URL list of excel files identified')
    if verbose: print(f'-3--> {url_list}')
    return url_list # of type list


# private
def _get_zip_from_cache(url:Union[str, List[str]], 
                        PREFIX:str, verbose:bool
                       ) -> Tuple[str, Path, Optional[bytes]]:
    """Get a zip file from the cache if it is there and not stale."""
    
    stem = (
        f'{url[0] if isinstance(url, list) else url}'
        .replace(PREFIX, '')
        .replace('/', '_')
        .replace('.xlsx', '.zip')
        .replace('.XLSX', '.zip')
    )
    cache_name = f'{CACHE_DIR}{stem}'
    if verbose: print(f'Cache file name: {cache_name}')
    zip_file = _get_cache_contents(cache_path := Path(cache_name))
    return cache_name, cache_path, zip_file


# private
def _get_xlsx_from_ABS(url_list:List, PREFIX:str, 
               cache_path:Path, verbose:bool
              ) -> Optional[bytes]:
    """Get each of individual .xlsx files and put in a
       zip-file. Return that zip-file as bytes"""
    
    # get the individual xl-table data from ABS
    xl_dict = {}
    for url in url_list:
        url = url.replace(PREFIX, '')
        url = f'{PREFIX}{url}' 
        name = Path(url).name
        xl_dict[name] = _get_url_contents(url)
    if verbose: print(f'Captured: {xl_dict.keys()}')

    # build a cache file ...
    with zip.ZipFile(cache_path, 'w', zip.ZIP_DEFLATED) as zf:
        for name, contents in xl_dict.items():
            if not contents:
                print(f'Something odd happened when zipping {name}')
                continue
            zf.writestr(f'/{name}', contents, )
    
    # return the zip-file
    zipfile = _get_cache_contents(cache_path)
    if zipfile is None:
        print('Unexpected error: the written zip-file is not there?')
    else:
        if verbose:
            print(f'Zipfile is {len(zipfile):,} bytes long.')
    return zipfile


# private
def _get_zip_from_ABS(url:str, PREFIX:str, 
                      cache_path:Path, verbose:bool
                     ) -> Optional[bytes]:
    """Get zip-file from the ABS and place into the cache."""

    # get zip-file from ABS website
    url = PREFIX + url
    print('We need to download this file from the ABS ...')
    if verbose: print(url)
    zip_file = _get_url_contents(url)
    if zip_file is None: 
        return None

    # cache for next time and return
    print(f'Saving ABS download to cache.')
    cache_path.open(mode='w', buffering=-1, encoding=None, 
              errors=None, newline=None)
    cache_path.write_bytes(zip_file)
    return zip_file


# private
def _get_ABS_zip_file(catalogue_id:str, table:int, verbose:bool) -> Optional[bytes]:
    """Get the latest zip_file of all tables for
       a specified ABS catalogue identifier"""
    
    # get relevant web-page from ABS website
    page = _get_ABS_webpage(catalogue_id)
    if not page:
        print(f'Failed to retrieve ABS web page for {catalogue_id}')
        return None
    
    # extract web address
    url = _get_urls(page, table, verbose)
    if not url:
        print('No URL found for data')
        return None
    
    # get from cache:
    PREFIX = "https://www.abs.gov.au"
    cache_name, cache_path, zip_file = (
        _get_zip_from_cache(url, PREFIX, verbose)
    )
    if zip_file:
        return zip_file
    
    # get direct from ABS and cache for future use
    return (
        _get_xlsx_from_ABS(url, PREFIX, cache_path, verbose) 
        if isinstance(url, list) 
        else _get_zip_from_ABS(url, PREFIX, cache_path, verbose)
    )


# private
def _get_dataframes(zip_file:bytes, verbose:bool
                   ) -> Optional[Dict[str, pd.DataFrame]]:
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
    
    freq_dict = {'annual':'Y', 'quarter':'Q', 'month':'M'}
    months = {1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR', 5: 'MAY', 6: 'JUN',
              7: 'JUL', 8: 'AUG', 9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC'}
    
    print('Extracting DataFrames from the zip-file ...')
    returnable = {}
    zipped = zip.ZipFile(io.BytesIO(zip_file))
    zipped_elements = zipped.infolist()

    meta = pd.DataFrame()
    for ze in zipped_elements:
        # a new DataFrame for each table
        data = pd.DataFrame()
    
        # get the zipfile into pandas
        xl_file = zipped.read(ze.filename)
        xl = pd.ExcelFile(xl_file)

        # get table information
        if 'Index' not in xl.sheet_names:
            print(f'Caution: Could not find the "Index" sheet in {ze.filename}')
            continue
        sheet_meta = xl.parse('Index', nrows=8)
        table = sheet_meta.iat[4,1]
        splat = table.split('.')
        tab_num = splat[0].split(' ')[-1].strip()
        tab_desc = '.'.join(splat[1:]).strip()
        
        # get the metadata
        sheet_meta = xl.parse('Index', header=9, parse_dates=True, 
                              infer_datetime_format=True, converters={'Unit':str})
        sheet_meta = sheet_meta.iloc[1:-2] # drop first and last 2
        sheet_meta = sheet_meta.dropna(axis='columns', how='all')
        sheet_meta['Unit'] = (
            sheet_meta['Unit'].str
            .replace('000 Hours', 'Thousand Hours')
            .replace('000,000', 'Millions')
            .replace('000', 'Thousands')
        )
        sheet_meta['Table'] = tab_num.strip()
        sheet_meta['Table Description'] = tab_desc
        if meta is None:
            meta = sheet_meta
        else:
            meta = pd.concat([meta, sheet_meta])
        # establish freq - used for making the index a PeriodIndex
        freq = sheet_meta['Freq.'].str.lower().unique()
        freq = freq_dict[freq[0]] if len(freq) == 1 and freq[0] in freq_dict else None
        if freq is None:
            print(f'Unrecognised data frequency for {table}')
    
        # get the actual data
        data_sheets = [x for x in xl.sheet_names if x.startswith('Data')]
        for sheet_name in data_sheets:
            sheet_data = xl.parse(sheet_name, header=9, index_col=0, 
                                  parse_dates=True, infer_datetime_format=True,)
        
            for i in sheet_data.columns:
                if i in data.columns:
                    # Remove duplicate Series IDs before merging
                    del sheet_data[i]
                    continue
                if verbose and sheet_data[i].isna().all():
                    # Warn if data series is all NA
                    problematic = meta.loc[meta["Series ID"] == i][
                        ['Table', 'Data Item Description', 'Series Type']]
                    print(f'Warning: no data for {i}\n{problematic}\n\n')

            # merge data into a large dataframe
            if len(data) == 0:
                data = sheet_data
            else:
                data = pd.merge(left=data, right=sheet_data, 
                                how='outer', left_index=True, 
                                right_index=True, suffixes=('', ''))
        if freq:
            if freq == 'Q' or freq == 'A':
                month = months[data.index.month.max()]
                freq = f'{freq}-{month}'
            data = data.to_period(freq=freq)
        returnable[tab_num] = data
    
    returnable[META_DATA] = meta
    return returnable


# public
def get_ABS_meta_and_data(catalogue_id:str, table:int=0, verbose=False
                         ) -> Optional[Dict[str, pd.DataFrame]]:
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

    zip_file = _get_ABS_zip_file(catalogue_id, table, verbose)
    if zip_file is None:
        return None
    return _get_dataframes(zip_file, verbose)


# --- identify the data series from the meta data DataFrame

def find_id(meta:pd.DataFrame, search_terms: Dict[str, str], 
            exact=False,
            verbose:bool=False, validate_unique:bool=True) -> Tuple[str]:
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
    
    m = meta.copy()
    for phrase, column in search_terms.items():
        if verbose: 
            print(f'Searching {len(m)}: term: {phrase} in-column: {column}')
        if exact or column == 'Table': # always match table exactly
            m = m[m[column] == phrase]
        else:
            m = m[m[column].str.contains(phrase, regex=False)]
    if verbose: 
        print(len(m))
    if verbose and len(m) != 1:
        display(m)
    if validate_unique:
        assert len(m) == 1
    return m['Series ID'].values[0], m['Unit'].values[0]


def get_identifier(meta, data_item_description, series_type, table):
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
        table: 'Table',
        series_type: 'Series Type',
        data_item_description: 'Data Item Description'
    }
    
    return find_id(meta, search, exact=True)
    

# --- data recalibration

keywords = {'Number':0, 'Thousand':3, 'Million':6, 'Billion':9, 'Trillion':12, 'Quadrillion':15}
r_keywords = {v: k for k, v in keywords.items()}

def find_calibration(series: pd.Series, units:str) -> Optional[str]:
    contains = []
    found = None
    for keyword in keywords:
        if keyword in units or keyword.lower() in units: 
            found = keyword
            break
    return found


def dont_recalibrate(series: pd.Series, units:str, verbose:bool=False) -> bool:
    if series.max() < 0:
        if verbose: 
            print('Negative max numbers will not be adjusted')
        return True
    if not pd.api.types.is_numeric_dtype(series):
        if verbose: 
            print(f'Series not numeric {series.dtype}')
        return True
    if find_calibration(series, units) is None:
        if verbose: 
            print(f'Units not calibrated {units}')
        return True
    if series.max() <= 1000 and series.max() >= 1:
        if verbose: 
            print('No adjustments needed')
        return True
    return False


def recalibrate_series(series: pd.Series, units:str) -> Tuple[pd.Series, str]:
    if dont_recalibrate(series, units):
        #print('Not recallibrated')
        return series, units
    
    def _recalibrate(factor, step, operator):
        if factor + step in r_keywords:
            replacement = r_keywords[factor + step]
            nonlocal units, series # bit ugly
            units = units.replace(text, replacement)
            units = units.replace(text.lower(), replacement)
            series = operator(series, 1000)
            return True
        return False

    again = True
    while again:
        text = find_calibration(series, units)
        factor = keywords[text]
        
        if series.max() > 1000:
            if _recalibrate(factor, 3, truediv):
                continue
                
        if series.max() < 1:
            if _recalibrate(factor, -3, mul):
                continue
          
        again = False
    return series, units
    

def recalibrate_value(value: float, units: str):
    input = pd.Series([value])
    output, units = recalibrate_series(input, units)
    return output[0], units
    

# --- plotting

def clear_chart_dir(chart_dir):
    """Remove all .png files from the chart_dir."""
    
    for fs_object in Path(chart_dir).glob("*.png"):
        if fs_object.is_file():
            fs_object.unlink()


# local imports
from finalise_plot import finalise_plot

def plot_growth2(annual, periodic, title, from_, tag, 
                 chart_dir, **kwargs):

    # put our two series into a datadrame
    frame = pd.DataFrame([annual.copy(), periodic.copy()], 
                         index=['Annual', 'Periodic']).T
    
    # get period description
    if kwargs['ppy'] == 4: # ppy = periods_per_year
        period = 'Quarterly'
        freq = "Q"
        adjustment = 46 # days from start to centre
    elif kwargs['ppy'] == 12:
        period = 'Monthly'
        freq="M"
        adjustment = 15 
    else:
        period = 'Unknown'
        freq="D"
        
    # set index to the middle of the period for selection
    if from_:
        frame = frame[frame.index >= from_]
    frame.index = pd.PeriodIndex(frame.index, freq=freq)
    frame = frame.to_timestamp(how='start')
    frame.index = frame.index + pd.Timedelta(days=adjustment)
    
    # plot
    fig, ax = plt.subplots()
    ax.plot(frame[frame.columns[0]].index, 
            frame[frame.columns[0]].values,
            lw=1, color='#0000dd',)
    ax.bar(frame[frame.columns[1]].index, 
           frame[frame.columns[1]].values, 
           color="#dd0000",
           width=(0.8 * adjustment * 2))
    ax.margins(x=0, y=0.025) # further adjusted in finalise_plot()

    ax.legend(['Annual growth', f'{period} growth'], loc='best')

    locator = mdates.AutoDateLocator(minticks=4, maxticks=13)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    finalise_plot(ax, title, 'Per cent', f'growth-{tag}', chart_dir, **kwargs)

    
def plot_growth(series, title, from_, tag, chart_dir, **kwargs):
    assert('ppy' in kwargs) # ppy = periods_per_year
    annual = series.pct_change(periods=kwargs['ppy']) * 100.0
    periodic = series.pct_change(periods=1) * 100.0
    plot_growth2(annual, periodic, title, from_, tag, chart_dir, **kwargs)

    
def plot_Qgrowth(series, title, from_, tag, chart_dir, **kwargs):
    kwargs['ppy'] = 4
    plot_growth(series, title, from_, tag, chart_dir, **kwargs)
    
    
def plot_Mgrowth(series, title, from_, tag, chart_dir, **kwargs):
    kwargs['ppy'] = 12
    plot_growth(series, title, from_, tag, chart_dir, **kwargs)


# --- project and plot a short-run counter-factual series

def get_projection(original: pd.Series, to_period: pd.Period) -> pd.Series:
    """Projection based on data from the start of a series 
       to the to_period (inclusive). Returns projection over the whole
       period of the original series."""
    
    y = original[original.index <= to_period]
    x = np.arange(len(y))
    regress_data = pd.DataFrame({
        'y': y.values,
        'x': x,
    })
    model = smf.ols(formula='y ~ x', data=regress_data).fit()
    #print(model.summary())
    #print(model.params)

    xx = np.arange(len(original))
    projection = (
        pd.Series(xx * model.params['x'] + model.params['Intercept'],
                  index = original.index)
    )
    
    return projection


def plot_covid_recovery(series:pd.Series, *args, **kwargs) -> None:
    """Plots a series with a PeriodIndex. 
       Arguments
        - series to be plotted
        - ^args and ^^kwargs - same as for finalise_plot()."""
    
    # sanity checks
    if not isinstance(series, pd.Series):
        raise TypeError('The series argument must be a pandas Series')
    if not isinstance(series.index, pd.PeriodIndex):
        raise TypeError('The series must have a pandas PeriodIndex')
    
    # plot COVID counterfactural   
    freq = series.index.freq
    #print(f'--- {freq} ---')
    if freq == 'M':
        # assume last unaffected month is January 2020
        LIN_REGRESS = pd.Period('2020-01-01', freq=freq)
        PRE_COVID = pd.Period('2017-01-01', freq=freq)
    else:
        # assume last unaffected quarter ends in December 2019
        LIN_REGRESS = pd.Period('2019-12-31', freq=freq)    
        PRE_COVID = pd.Period('2016-12-31', freq=freq)
    recent = series[series.index >= PRE_COVID]
    projection = get_projection(recent, LIN_REGRESS)

    ax = recent.plot(lw=2, c="dodgerblue", label=series.name)
    ax = projection.plot(lw=2, c="darkorange", ls='--', label='Pre-COVID projection')
    ax.legend(loc='best')
    
    # augment left-footer
    lfooter = '' if 'lfooter' not in kwargs else kwargs['lfooter']
    lfooter += f'Projection on data from {PRE_COVID} to {LIN_REGRESS}. '
    kwargs['lfooter'] = lfooter
    
    finalise_plot(ax, *args, **kwargs)