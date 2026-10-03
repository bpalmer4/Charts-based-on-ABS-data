# Type stubs for the parts of yfinance this project uses (yfinance ships no type information).
# Add a declaration here when the package starts using another part of yfinance.

import pandas as pd

def download(
    tickers: str | list[str],
    start: str | None = None,
    end: str | None = None,
    *,
    auto_adjust: bool = ...,
    progress: bool = ...,
) -> pd.DataFrame | None: ...

class Ticker:
    def __init__(self, ticker: str) -> None: ...
    def history(self, period: str = ...) -> pd.DataFrame: ...
