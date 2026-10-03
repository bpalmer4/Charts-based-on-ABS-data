"""ASX share price indices: All Ordinaries, S&P/ASX 200 and Small Ordinaries, from Yahoo Finance."""

# --- dependencies
import pandas as pd

from au_econ.charting.daily_prices import fetch_closes, frame_chart, last_day, single_charts, summarise

# --- module contract
RELEASE = ("asx",)
TOPICS = ()
TITLE = "ASX Indices"

# --- constants
LONG_START = "2024-03-20"  # about two years
BROAD = {"^AORD": "All Ordinaries", "^AXJO": "S&P/ASX 200"}
SMALL = [("^AXSO", "S&P/ASX Small Ordinaries", "Index")]


# --- data
def fetch() -> dict[str, pd.Series]:
    """Fetch each index's daily close from LONG_START, keyed by ticker."""
    return fetch_closes([*BROAD, *(ticker for ticker, _, _ in SMALL)], LONG_START)


# --- charts
def broad_indices(data: dict[str, pd.Series]) -> None:
    """All Ordinaries and S&P/ASX 200 on one chart."""
    frame = pd.DataFrame({label: data[ticker] for ticker, label in BROAD.items()})
    summarise(frame, "ASX")
    frame_chart(
        frame,
        title="ASX Indices: All Ordinaries and S&P/ASX 200",
        ylabel="Index",
        lfooter=f"Daily close. Data to {last_day(frame)}.",
    )


def small_ordinaries(data: dict[str, pd.Series]) -> None:
    """S&P/ASX Small Ordinaries."""
    single_charts(data, SMALL, is_futures=False)


# --- table of contents, in run order
CHARTS = (
    (broad_indices, ()),
    (small_ordinaries, ()),
)
