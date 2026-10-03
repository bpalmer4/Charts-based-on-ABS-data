"""Commodity prices in US dollars, from FRED: energy, metals, agriculture and two producer price indexes."""

# --- dependencies
import pandas as pd
from mgplot import line_plot_finalise

from au_econ.sources.fred import get_series

# --- module contract
RELEASE = ("fred-commodities",)
TOPICS = ("commodities",)
TITLE = "Commodity Prices"

# --- constants
START = "2010-01-01"
SOURCE = "FRED"
MIN_OBSERVATIONS = 10
SERIES = {  # chart title: FRED series ID; all prices in USD
    "Coal - Australia ($/metric ton)": "PCOALAUUSDM",
    "LNG - Japan ($/mmbtu)": "PNGASJPUSDM",
    "Natural Gas - Henry Hub Spot ($/mmbtu)": "DHHNGSP",
    "Natural Gas - Europe ($/mmbtu)": "PNGASEUUSDM",
    "Crude Oil - WTI Spot ($/barrel)": "DCOILWTICO",
    "Crude Oil - Brent Spot ($/barrel)": "DCOILBRENTEU",
    "Copper ($/metric ton)": "PCOPPUSDM",
    "Aluminum ($/metric ton)": "PALUMUSDM",
    "Iron Ore ($/metric ton)": "PIORECRUSDM",
    "Nickel ($/metric ton)": "PNICKUSDM",
    "Steel Mill Products PPI (Index)": "WPU101",
    "Wheat ($/metric ton)": "PWHEAMTUSDM",
    "Corn/Maize ($/metric ton)": "PMAIZMTUSDM",
    "Soybeans ($/metric ton)": "PSOYBUSDM",
    "Lumber PPI (Index)": "PCU321114321114",
    "Cotton ($/kg)": "PCOTTINDUSDM",
}

# pandas inferred frequencies, mapped to PeriodIndex frequencies
INFERRED_FREQUENCIES = {
    "M": ("MS", "M", "ME"),
    "Q": ("QS", "Q", "QE"),
    "A": ("AS", "A", "AE", "Y", "YS", "YE"),
    "D": ("B", "D"),
}
# when pandas cannot infer: the largest median gap (days) for each frequency, in order
MEDIAN_GAP_LIMITS = (("D", 7), ("M", 35), ("Q", 100))


# --- data
def _frequency(dates: pd.DatetimeIndex) -> str:
    """Return the PeriodIndex frequency of a series' dates (daily, monthly, quarterly or annual)."""
    inferred = pd.infer_freq(dates)
    for freq, aliases in INFERRED_FREQUENCIES.items():
        if inferred in aliases:
            return freq
    if inferred is not None:
        return inferred
    median_gap = pd.Series(dates).diff().dt.days.dropna().median()
    return next((freq for freq, limit in MEDIAN_GAP_LIMITS if median_gap <= limit), "A")


def fetch() -> dict[str, pd.Series]:
    """Return each commodity's prices, keyed by chart title, with a PeriodIndex at its native frequency."""
    prices: dict[str, pd.Series] = {}
    for title, series_id in SERIES.items():
        series = get_series(series_id, START)
        if len(series) < MIN_OBSERVATIONS:
            raise ValueError(f"FRED {series_id}: only {len(series)} observations from {START}")
        if not isinstance(series.index, pd.DatetimeIndex):
            raise TypeError(f"FRED {series_id}: expected a DatetimeIndex")
        series.index = pd.PeriodIndex(series.index, freq=_frequency(series.index))
        prices[title] = series.rename(title)
    return prices


# --- charts
def prices(data: dict[str, pd.Series]) -> None:
    """One chart per commodity."""
    for title, series in data.items():
        line_plot_finalise(
            series,
            title=title,
            ylabel="Price (USD)",
            xlabel=None,
            rfooter=SOURCE,
            lfooter=f"{len(series)} observations. Latest: {series.index[-1]}",
        )


# --- table of contents, in run order
CHARTS = ((prices, ()),)
