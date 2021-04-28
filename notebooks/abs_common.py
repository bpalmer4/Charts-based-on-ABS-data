# abs_common.py

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.units as munits

import requests
from bs4 import BeautifulSoup
import re
import pathlib
import zipfile
import io


# --- utility functions
META_DATA = 'META_DATA'

def get_plot_constants(meta):
    """Get plotting constants"""
    
    RECENCY_PERIOD = 3 # years
    RECENT = (
        meta['Series End'].max() 
        - pd.DateOffset(years=RECENCY_PERIOD)
    )
    plot_times = [None, RECENT]
    plot_tags = ('full', 'recent')
    return RECENT, plot_times, plot_tags


def get_fs_constants(catalogue_id):
    source = f'ABS {catalogue_id}'
    CHART_DIR = f"./charts/{catalogue_id}/"
    pathlib.Path(CHART_DIR).mkdir(parents=True, exist_ok=True)
    return source, CHART_DIR, META_DATA

# --- Data fetch from ABS

"""
Our general approach here is to:

1. Download the "latest-release" webpage from the ABS
   for known ABS catalogue nuymbers. 

2. Parse that webpage to find the link to the download 
   all-tables zip-file. We do this because the name of
   the file location on the ABS server changes from 
   month to month. 
   
3. Check to see whether I have cached that file previously,
   if not, download and save the zip-file to the cache.
   
4. Open the zip-file, and save each table to a pandas 
   DataFrame. And save the metadata to a pandas DataFrame.
   Return all of the DataFrames in a dictionary.
"""

# -- Establish an ABS cache directory
CACHE_DIR = "./ABS_CACHE/"
pathlib.Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

# -- URLs for getting key data sets from the ABS
locations = {

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
    
    "6401": {
        "Name": "Consumer Price Index, Australia",
        "URL": "https://www.abs.gov.au/statistics/economy/"
               "price-indexes-and-inflation/"
               "consumer-price-index-australia/latest-release",
    },
    
    "8501": {
        "Name": "Retail Trade, Australia",
        "URL": "https://www.abs.gov.au/statistics/industry/"
               "retail-and-wholesale-trade/retail-trade-australia/"
               "latest-release",
    },

    "LAA": {
        "Name": "Labour Account Australia",
        "URL": "https://www.abs.gov.au/statistics/"
               "labour/employment-and-unemployment/"
               "labour-account-australia/latest-release",
    },

}


def get_ABS_catalogue_IDs():
    """Return a dictionary of known ABS catalogue identifiers"""
    
    response = {}
    for location, data in locations.items():
        response[location] = data["Name"]
    return response


def get_ABS_webpage(catalogue_id):
    """Get the ABS web page for latest data for a
       specified ABS catalogue identifier.

       Arguments:
       catalogue_id - string - catalogue identifier 
       (eg. 6202 for Labour Force Australia)
    """

    # get URL
    if catalogue_id not in locations:
        print(f"Catalogue identifier not recognised: {catalogue_id}")
        return None
    
    url = locations[catalogue_id]["URL"]
    requested = requests.get(url, allow_redirects=True)
    code = requested.status_code
    if code != 200:
        print(f'Could not get web page ({url}), error code: {code}')
        return None
    
    # capture and return the ABS webpage
    page = requested.content
    return page


def get_ABS_zipfile(catalogue_id):
    """Get the latest zip_file of all tables for
       a specified ABS catalogue identifier"""
    
    # get latest from ABS website
    page = get_ABS_webpage(catalogue_id)
    if page is None or len(page) == 0:
        print(f'Failed to retrieve ABS web page for {catalogue_id}')
        return None
    
    # remove those pesky span tags
    page = re.sub(b'<span[^>]*>', b' ', page)
    page = re.sub(b'</span>', b' ', page)
    page = re.sub(b'\s+', b' ', page) # tidy up white space
    
    # extract web address
    soup = BeautifulSoup(page, features="lxml")
    found = soup.findAll('a', text=re.compile(r'Download all', re.IGNORECASE))
    if len(found) == 0:
        found = soup.findAll('a', text=re.compile(r'Download zip', re.IGNORECASE))
        if len(found) == 0:
            return None
    found = found[0]
    url = re.search(r'href="([^ ]+)"', str(found.prettify)).group(1)
    
    # note: ABS uses full URL addresses sometimes, and sometimes not
    PREFIX = "https://www.abs.gov.au"
    if PREFIX in url:
        url = url.replace(PREFIX, '')
    cache_name = CACHE_DIR + url.replace('/', '-')
    
    # check local cache for web address
    if (file := pathlib.Path(cache_name)).is_file():
        print(f'Retrieving zip-file from cache {cache_name}')
        zip_file = file.read_bytes()
        return zip_file

    # get zip-file from ABS website
    url = PREFIX + url
    print('We need to download this file')
    gotten = requests.get(url, allow_redirects=True)
    if gotten.status_code != 200:
        print(f'Could not get web page ({url}), error code: {code}')
        return None    
    zip_file = gotten.content # bytes

    # cache for next time
    print(f'Saving to cache: {cache_name}')
    file.open(mode='w', buffering=-1, encoding=None, errors=None, newline=None)
    file.write_bytes(zip_file)
    return zip_file


def get_dataframes(zip_file, warning=False):
    """Get a DataFrame for each table in the zip-file, 
       plus an overall DataFrame for the metadata. 
       Return these in a dictionary
       Arguments:
        - zip_file - bytes array of ABS zip file of excel spreadsheets
        - warning - warn when dataframes are empty. 
       Returns:
        - either None (failure) or a dictionary containing a 
          separate DataFrame for each table in the zip-file,
          plus a DataFrame called 'META' for the metadata. 
    """
    
    returnable = {}
    zipped = zipfile.ZipFile(io.BytesIO(zip_file))
    zipped_elements = zipped.infolist()

    meta = pd.DataFrame()
    for ze in zipped_elements:
        # a new DataFrame for each table
        data = pd.DataFrame()
    
        # get the zipfile into pandas
        zfile = zipped.read(ze.filename)
        xl = pd.ExcelFile(zfile)

        # get table information
        sheet_meta = xl.parse('Index', nrows=8)
        table = sheet_meta.iat[4,1]
        splat = table.split('.')
        tab_num = splat[0].split(' ')[-1].strip()
        tab_desc = '.'.join(splat[1:]).strip()
        
        # get the metadata
        sheet_meta = xl.parse('Index', header=9, parse_dates=True, 
                              infer_datetime_format=True,)
        sheet_meta = sheet_meta.iloc[1:-2] # drop first and last 2
        sheet_meta = sheet_meta.dropna(axis='columns', how='all')
        sheet_meta['Table'] = tab_num.strip()
        sheet_meta['Table Description'] = tab_desc
        meta = meta.append(sheet_meta, ignore_index=True)
    
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
                if warning and sheet_data[i].isna().all():
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
                
        returnable[tab_num] = data
    
    # some data-type clean-ups 
    meta['Series End'] = pd.to_datetime(
        meta['Series End'],
        format='%Y-%m-%d'
    )
    meta['Series Start'] = pd.to_datetime(
        meta['Series Start'],
        format='%Y-%m-%d'
    )
    returnable[META_DATA] = meta
    return returnable


def get_ABS_meta_and_data(catalogue_id):
    """Get two pandas DataFrames, the first containing the ABS metadata,
       the second contraining the complete set of actual data from the ABS
       Arguments:
        - catalogue_id - string - ABS catalogue number for the desired dataset.
       Returns:
        - either None (failure) or a dictionary containing a 
          separate DataFrame for each table in the zip-file,
          plus a DataFrame called 'META' for the metadata."""

    zip_file = get_ABS_zipfile(catalogue_id)
    if zip_file is None:
        return None
    return get_dataframes(zip_file)


# --- plotting

# local imports
from finalise_plot import finalise_plot

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
            - units - string - unit of measurement
    """
    
    # make selection
    selected = meta[
        (meta['Data Item Description'] == data_item_description) &
        (meta['Series Type'] == series_type) &
        (meta['Table'] == table)
    ]
    
    # warn if something looks odd
    if len(selected) != 1:
        print(f'Warning: {len(selected)} items selected in \n' 
              '\tget_identifier(data_item_description='
              f'"{data_item_description}", \n' 
              f'\t\tseries_type="{series_type}", \n' 
              f'\t\ttable="{table}")')
        
    # return results
    id = selected['Series ID'].iloc[0]
    units = selected['Unit'].iloc[0]
    return id, units


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