"""Charts of daily closing prices from Yahoo Finance: one ticker, or several on one chart."""

import mgplot as mg
import pandas as pd

from au_econ.sources import yahoo

SOURCE_YAHOO = "Source: Yahoo Finance"
LEGEND = {"loc": "best", "fontsize": "x-small"}


def last_day(data: pd.DataFrame | pd.Series) -> str:
    """Return the last date with any data, e.g. "2-Oct-2026"."""
    last = data.dropna(how="all").index[-1]
    if not isinstance(last, pd.Period):
        raise TypeError("Expected a PeriodIndex")
    return last.strftime("%-d-%b-%Y")


def fetch_closes(tickers: list[str], start: str) -> dict[str, pd.Series]:
    """Fetch each ticker's daily close from start; a ticker with no data is reported and left out."""
    closes = {}
    for ticker in tickers:
        try:
            closes[ticker] = yahoo.get_close(ticker, start)
        except ValueError as error:
            print(f"{ticker}: {error}")
    return closes


def single_chart(series: pd.Series, *, name: str, ylabel: str, is_futures: bool = True) -> None:
    """Chart one ticker's close: a front-month futures price, or a daily close."""
    data_to = last_day(series)
    title, footer = (
        (f"{name} Futures Price", f"Front-month futures. Data to {data_to}.")
        if is_futures
        else (name, f"Daily close. Data to {data_to}.")
    )
    mg.line_plot_finalise(
        series,
        title=title,
        ylabel=ylabel,
        xlabel=None,
        annotate=True,
        lfooter=footer,
        rfooter=SOURCE_YAHOO,
    )


def single_charts(
    closes: dict[str, pd.Series], tickers: list[tuple[str, str, str]], *, is_futures: bool = True
) -> None:
    """Chart each (ticker, name, ylabel) separately; a ticker without data is skipped."""
    for ticker, name, ylabel in tickers:
        if ticker not in closes:
            print(f"{name} ({ticker}): no data, skipping")
            continue
        series = closes[ticker]
        print(f"{name}: {series.index[0]} to {series.index[-1]}  min={series.min():.2f}  max={series.max():.2f}")
        single_chart(series, name=name, ylabel=ylabel, is_futures=is_futures)


def frame_chart(frame: pd.DataFrame, *, title: str, ylabel: str, lfooter: str) -> None:
    """Chart several tickers' closes together."""
    mg.line_plot_finalise(
        frame,
        title=title,
        ylabel=ylabel,
        xlabel=None,
        legend=LEGEND,
        annotate=True,
        axvline=None,
        lfooter=lfooter,
        rfooter=SOURCE_YAHOO,
    )


def summarise(frame: pd.DataFrame, label: str) -> None:
    """Print the date range and min/max of each column."""
    valid = frame.dropna(how="all")
    print(f"{label}: {valid.index[0]} to {valid.index[-1]}")
    for column in frame.columns:
        series = frame[column].dropna()
        if len(series):
            print(f"  {column:22s}  n={len(series):3d}  min={series.min():.2f}  max={series.max():.2f}")
