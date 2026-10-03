"""Yahoo Finance: daily closing values (prices, yields, index levels), through yfinance.

Also forward curves: the latest close of each dated NYMEX contract for a futures root.
"""

import logging

import pandas as pd
import yfinance as yf

MONTH_CODES = "FGHJKMNQUVXZ"  # CME contract-month codes, January to December
CURVE_PROBE_SLACK = 6  # extra months probed, to cover expired fronts
RECENT_HISTORY = "5d"  # enough history to find a contract's latest close


def get_close(ticker: str, start: str) -> pd.Series:
    """Return a ticker's daily close from start (YYYY-MM-DD), with a daily PeriodIndex."""
    raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if raw is None or len(raw) == 0:
        raise ValueError(f"Yahoo Finance returned no data for {ticker}")
    close = raw["Close"].squeeze()
    if not isinstance(close, pd.Series) or not isinstance(close.index, pd.DatetimeIndex):
        raise TypeError(f"Yahoo Finance {ticker}: expected one dated column of closes")
    close = close.dropna()
    close.index = pd.DatetimeIndex(close.index).to_period("D")
    return close


def _contract_months(n_months: int) -> list[pd.Period]:
    """Candidate contract months, from the current month forward.

    Not every candidate is listed: a root's front contract expires before its delivery
    month (Brent settles about two months ahead of it), and Yahoo's monthly strip ends
    early for some roots (Brent: about 16 months, then only June and December). Probing
    from the current month with slack lets get_forward_curve() skip expired fronts.
    """
    start = pd.Period(pd.Timestamp.today(), freq="M")
    return [start + i for i in range(n_months + CURVE_PROBE_SLACK)]


def _contract_symbol(root: str, month: pd.Period) -> str:
    """Return a dated NYMEX contract's Yahoo symbol, e.g. CLX26.NYM."""
    return f"{root}{MONTH_CODES[month.month - 1]}{month.year % 100:02d}.NYM"


def get_forward_curve(root: str, n_months: int) -> pd.DataFrame:
    """Return the latest close ("price") and trade date ("date") of up to n_months consecutive live contracts.

    Unlisted months before the first live contract are expired fronts and are skipped;
    the first unlisted month after it ends the curve, so a sparse tail is never joined
    across a gap. yfinance logs every unlisted probe as an error; those are expected, so
    its logger is quietened while probing.
    """
    records: dict[pd.Period, dict[str, float | pd.Timestamp]] = {}
    yf_logger = logging.getLogger("yfinance")
    level = yf_logger.level
    yf_logger.setLevel(logging.CRITICAL)
    try:
        for month in _contract_months(n_months):
            if len(records) >= n_months:
                break
            try:
                history: pd.DataFrame | None = yf.Ticker(_contract_symbol(root, month)).history(
                    period=RECENT_HISTORY
                )
            except KeyError, ValueError, OSError:
                history = None
            if history is None or len(history) == 0:
                if records:
                    break
                continue
            last = history.index[-1]
            if not isinstance(last, pd.Timestamp):
                raise TypeError(f"Yahoo {root}: expected a dated history")
            records[month] = {
                "price": float(history["Close"].iloc[-1]),
                "date": last.tz_localize(None).normalize(),
            }
    finally:
        yf_logger.setLevel(level)
    return pd.DataFrame.from_dict(records, orient="index").sort_index()
