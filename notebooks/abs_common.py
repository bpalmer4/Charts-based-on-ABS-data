# abs_common.py

# --- initialisation

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.units as munits

import requests
import hashlib
import zipfile
import io

from pathlib import Path
from datetime import date

# local imports
from finalise_plot import finalise_plot


# --- useful constants

# nominal GDP can be a useful denominator
DATA_DIR = '../Data'
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
NOMINAL_GDP_CSV = f'{DATA_DIR}/nominal_gdp.csv'


# --- utility functions

def get_fs_constants(cat_num):
    """Get file system constants"""
    
    cache_dir = f'../cache/{cat_num}'
    chart_dir = f'../charts/{cat_num}'
    directories = [cache_dir, chart_dir]
    for d in directories:
        Path(d).mkdir(parents=True, exist_ok=True)
    directories.append(f'Source: ABS {cat_num} table')
    return(directories)


def get_plot_constants(data):
    """Get plotting constants"""
    
    RECENCY_PERIOD = 3 # years
    RECENT = data.index.max() - pd.DateOffset(years=RECENCY_PERIOD)
    plot_times = [None, RECENT]
    plot_frames = (data.copy(), data[data.index >= RECENT].copy())
    plot_tags = ('full', 'recent')
    return RECENT, plot_times, plot_frames, plot_tags


# --- data retrieval 

def previous_month(month, year):
    """Find the previous (month, year) combination 
        for a given (month, year) combination
       Arguments:
       - month - integer between 1 and 12
       - year - integer
       Returns:
       - Tuple (month, year) - same form as arguments
    """
    
    month = month - 1
    if month == 0:
        month = 12
        year -= 1
    return month, year


def download_and_cache(URL, CACHE_DIR):
    """If the URL has not been cached, download it and cache it:
       Arguments:
       - URL - string - url for file
       - CACHE - string - directory name where cached file is placed
       Returns:
       - None - if URL does not exist
       - a bytes array of the cached zip file of excel spreadsheets
    """

    code = requests.head(URL).status_code
    if code == 200:
        # if the URL exists, see if we have previously downloaded and cached.
        # if not previously downloaded, we will doenload now and cache.
        cache = (f'{CACHE_DIR}/{hashlib.sha384(URL.encode()).hexdigest()}.cache')
        if (file := Path(cache)).is_file():
            print('File has been cached already')
            zip_file = file.read_bytes()
        else:
            print('We need to cache this file')
            zip_file = requests.get(URL, allow_redirects=True).content # bytes
            file.open(mode='w', buffering=-1, encoding=None, errors=None, newline=None)
            file.write_bytes(zip_file)

        # return success
        return zip_file # as a bytes buffer
    
    # return failure
    return None


def get_ABS_zip_file(url_template, CACHE_DIR):
    """Using an url_template, get the ABS zipped file as a bytes array
       if not cached from the ABS, otherwise from the cache file.
       Arguments:
       - url_template - string - with MONTH-YEAR as a token that will be 
                                 substituted with a series of months/years
                                 walking backwards from the current to get 
                                 the most recent data.
       - CACHE_DIR - string - directory name for the cache directory
       Returns:
       - a bytes array of the cached zip file of excel spreadsheets
    """

    # function constants
    MAX_TRIES = 15 # maximum number of previous months to try
    MONTHS = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
              'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    
    # we will start with the previous month and work backwards
    # assume ABS data is at least one month old. 
    month, year = previous_month(date.today().month, date.today().year)
    for count in range(MAX_TRIES+1):
    
        # exit with error if the counter has gone too far backwards
        if count >= MAX_TRIES:
            sys.exit('Could not find and download the URL')
        
        # let's see if the URL exists
        month_year = MONTHS[month - 1] + '-' + str(year)
        URL = url_template.replace('MONTH-YEAR', month_year)
        if (zip_file := download_and_cache(URL, CACHE_DIR)) is not None:
            print(f'File for {month_year} of ' +
                  f'size {np.round((len(zip_file) / 1024**2), 1)} MB')
            break
        
        # let's try a month earlier
        month, year = previous_month(month, year)
    
    # if we are here, then we have a cached file we can unzip
    return zip_file


# let's build a single dataframe for all the data we have collected
def get_dataframes(zip_file, warning=False):
    """Get the dataframe for zipfile of excel spreadsheets
        Arguments:
        - zip_file - bytes array of ABS zip file of excel spreadsheets
        Returns:
        - tuple (meta, data), where
            - meta - is a pandas DataFrame of metadata
            - data - is a pandas DataFrame of the actual data
    """
    
    zipped = zipfile.ZipFile(io.BytesIO(zip_file))
    zipped_elements = zipped.infolist()

    meta = pd.DataFrame()
    data = pd.DataFrame()
    for ze in zipped_elements:
    
        # get the zipfile into pandas
        zfile = zipped.read(ze.filename)
        xl = pd.ExcelFile(zfile)

        # get table information
        sheet_meta = xl.parse('Index', nrows=8)
        table = sheet_meta.iat[4,1]
        splat = table.split('.')
        tab_num = splat[0].split(' ')[-1].strip()
        tab_desc = '.'.join(splat[1:]).strip()
        #print(tab_num, tab_desc) 
        
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
                data = pd.merge(left=data, right=sheet_data, how='outer',
                                left_index=True, right_index=True, suffixes=('', ''))
    
    return meta, data


def get_ABS_meta_and_data(url_template, cache_dir):
    """Get two pandas DataFrames, the first containing the ABS metadata,
       the second contraining the complete set of actual data from the ABS
       Arguments:
        - url_template - string - with MONTH-YEAR as a token that will be 
                                  substituted with a series of months/years
                                  walking backwards from the current to get 
                                  the most recent data.
        - cache_dir - string - directory name for the cache directory
        Returns:
        - tuple (meta, data), where
            - meta - is a pandas DataFrame of metadata
            - data - is a pandas DataFrame of the actual data
    """

    zip_file = get_ABS_zip_file(url_template, cache_dir)
    meta, data = get_dataframes(zip_file)
    return meta, data


# --- plotting

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
              f'\tget_identifier(data_item_description="{data_item_description}", \n' 
              f'\t\tseries_type="{series_type}", \n' 
              f'\t\ttable="{table}")')
        
    # return results
    id = selected['Series ID'].iloc[0]
    units = selected['Unit'].iloc[0]
    return id, units


def plot_growth2(annual, periodic, title, from_, tag, chart_dir, **kwargs):

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