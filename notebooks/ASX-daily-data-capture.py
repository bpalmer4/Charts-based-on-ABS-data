"""Daily ASX rate scraper - based on Matt Cowgill's R scraper at 
   https://github.com/MattCowgill/cash-rate-scraper.git
   crontab: 29 19 * * 1-5 /Users/bryanpalmer/ABS/notebooks/ASX-daily-capture.sh"""

# imports
from pathlib import Path
from typing import Final

import json
import pandas as pd
import common

# nameing conventions
CASH_RATE: Final = 'cash_rate'
SCRAPE_DATE: Final = 'scrape_date'
DATE = 'date'
FILE_STEM = 'scraped_cash_rate_'


def get_asx_data() -> pd.DataFrame:
    """Capture the latest ASX rate tracker data from the ASX website 
    and return it as a pandas DataFrame."""
    
    url = (
        "https://asx.api.markitdigital.com/asx-research/1.0/derivatives/"
        "interest-rate/IB/futures?days=1&height=179&width=179"       
    )
    raw_jason = common.request_get(url)
    df = pd.DataFrame(json.loads(raw_jason)['data']['items'])
    df[CASH_RATE] = (100 - df.pricePreviousSettlement).round(3)
    df[SCRAPE_DATE] = pd.to_datetime('today').normalize().date()
    df.index = pd.PeriodIndex(df['dateExpiry'], freq='M')
    df.index.name = DATE  # consistency with Matt Cowgill's R scraper
    return df[[CASH_RATE, SCRAPE_DATE]]


def save_asx_data(df: pd.DataFrame) -> None:
    """Save the ASX rate tracker data to a CSV file."""

    directory = "./ASX_DAILY_DATA/"
    Path(directory).mkdir(parents=True, exist_ok=True)
    filename = f"{directory}{FILE_STEM}{df[SCRAPE_DATE].max()}.csv"
    get_asx_data().to_csv(path_or_buf=filename)


def main() -> None:
    """The main function to capture and save the ASX rate tracker data."""

    df = get_asx_data()
    save_asx_data(df)


main()