"""AFSA personal insolvency statistics: people entering a new bankruptcy or other personal insolvency.

Monthly data is released about a month after the reference month; quarterly about two
months after the quarter.
"""

# --- dependencies
from dataclasses import dataclass

import pandas as pd
from mgplot import line_plot_finalise

from au_econ.sources import afsa

# --- module contract
RELEASE = ("afsa",)
TOPICS = ("insolvency",)
TITLE = "Personal Insolvencies"

# --- constants
SOURCE = "AFSA: personal insolvency statistics"
CAUTION = (
    f"Australia{chr(0x2019)}s bankruptcy laws were changed in 2020, "  # AFSA's typographic apostrophe
    "which may affect the comparability of data before and after this date."
)
WANTED = ["Total bankruptcies", "Total personal insolvencies"]
SITUATION = "Number of people entering a new personal insolvency"
KEY_COLUMN = "Type of personal insolvency administration"
WIDTHS = [1, 2]
TOTAL = "Total"


@dataclass(frozen=True)
class Frequency:
    """How one AFSA file labels its national row, period and business column, and its other filters."""

    state: str
    period: str
    business: str
    freq: str
    extra_filters: tuple[tuple[str, str], ...] = ()


MONTHLY = Frequency(
    state="Australia",
    period="Month",
    business="In a business or company",
    freq="M",
    extra_filters=(("Industry of Employment", TOTAL),),
)
QUARTERLY = Frequency(state=TOTAL, period="Quarter", business="Business or company involvement", freq="Q")


@dataclass(frozen=True)
class InsolvencyData:
    """National counts of people entering a new personal insolvency, by type: monthly and quarterly."""

    monthly: pd.DataFrame
    quarterly: pd.DataFrame


# --- data
def _national(frame: pd.DataFrame, frequency: Frequency) -> pd.DataFrame:
    """National counts of the wanted insolvency types, one column each, on a PeriodIndex."""
    mask = (
        (frame["State"] == frequency.state) & (frame[frequency.business] == TOTAL) & frame[KEY_COLUMN].isin(WANTED)
    )
    for column, value in frequency.extra_filters:
        mask &= frame[column] == value
    rows = frame[mask]
    if rows.duplicated([frequency.period, KEY_COLUMN]).any():
        raise ValueError(f"AFSA {frequency.period} data has more than one row per period and type")
    national = pd.DataFrame(
        {kind: rows[rows[KEY_COLUMN] == kind].set_index(frequency.period)[SITUATION] for kind in WANTED}
    )
    if national.empty:
        raise ValueError(f"AFSA {frequency.period} data has no national rows")
    # AFSA writes two-digit years (e.g. "Jan-24"); expand them to four digits
    national.index = pd.PeriodIndex(
        ["-20".join(str(label).split("-")) for label in national.index], freq=frequency.freq
    )
    return national.sort_index()


def fetch() -> InsolvencyData:
    """Fetch the monthly and quarterly files; keep the national totals."""
    return InsolvencyData(
        monthly=_national(afsa.get_csv(afsa.MONTHLY), MONTHLY),
        quarterly=_national(afsa.get_csv(afsa.QUARTERLY), QUARTERLY),
    )


# --- helpers
def _insolvency_chart(frame: pd.DataFrame, *, title: str, ylabel: str, cadence: str) -> None:
    """Plot total bankruptcies against total personal insolvencies."""
    last = frame.index[-1]
    if not isinstance(last, pd.Period):
        raise TypeError("Expected a PeriodIndex")
    data_to = last.strftime("%b %Y") if last.freqstr.startswith("M") else str(last)
    line_plot_finalise(
        frame[WANTED],
        title=title,
        ylabel=ylabel,
        width=WIDTHS,
        rfooter=SOURCE,
        lfooter=f"Australia. {cadence}. {SITUATION}. Data to {data_to}.",
        rheader=CAUTION,
    )


# --- charts
def monthly_insolvencies(data: InsolvencyData) -> None:
    """Monthly bankruptcies and personal insolvencies."""
    _insolvency_chart(
        data.monthly, title="Monthly Personal Insolvencies", ylabel="Number of people / month", cadence="Monthly"
    )


def quarterly_insolvencies(data: InsolvencyData) -> None:
    """Quarterly bankruptcies and personal insolvencies."""
    _insolvency_chart(
        data.quarterly,
        title="Quarterly Personal Insolvencies",
        ylabel="Number of people / quarter",
        cadence="Quarterly",
    )


# --- table of contents, in run order
CHARTS = (
    (monthly_insolvencies, ()),
    (quarterly_insolvencies, ()),
)
