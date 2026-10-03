"""World Bank commodity prices (Pink Sheet): every commodity in US dollars, and in Australian dollars."""

# --- dependencies
from dataclasses import dataclass

import pandas as pd
from mgplot import line_plot_finalise, multi_start

from au_econ.series.rates import get_aud_usd_monthly_average
from au_econ.sources import worldbank

# --- module contract
RELEASE = ("wb-commodities",)
TOPICS = ("commodities",)
TITLE = "Commodity Prices (Pink Sheet)"

# --- constants
SOURCE = "World Bank: Pink Sheet"
AUD_SOURCE = f"{SOURCE}; RBA: F11, F11.1"
METHOD = "** = method changes."
METHOD_MARK = "**"  # in a commodity label: its methodology changed
INDEX_MARK = "index"  # in a commodity label: a price index, with no AUD version
RECENT_MONTHS = 125  # about ten years
plot_times = (0, -RECENT_MONTHS)


@dataclass(frozen=True)
class PinkSheet:
    """Monthly prices (one column per commodity), their units, and US dollars per Australian dollar.

    The exchange rate is a monthly average of daily rates from averaged_from, end-of-month before.
    """

    prices: pd.DataFrame
    units: pd.Series
    aud_usd: pd.Series
    averaged_from: pd.Period


# --- data
def fetch() -> PinkSheet:
    """Fetch the World Bank workbook and the RBA exchange rates once; every chart function receives them."""
    aud_usd, averaged_from = get_aud_usd_monthly_average()
    prices, units = worldbank.get_commodity_prices()
    return PinkSheet(prices=prices, units=units, aud_usd=aud_usd, averaged_from=averaged_from)


# --- helpers
def _lfooter(commodity: str, prices: pd.Series, averaged_from: pd.Period | None = None) -> str:
    """Left footer: monthly averages and the latest month; for AUD, the exchange rate; any methodology note."""
    last = prices.index[-1]
    if not isinstance(last, pd.Period):
        raise TypeError(f"{commodity}: expected a monthly PeriodIndex")
    notes = [f"Monthly averages, data to {last.strftime('%b %Y')}."]
    if averaged_from is not None:
        notes.append(f"AUD at avg RBA rate (month-end pre-{averaged_from.year}).")
    if METHOD_MARK in commodity:
        notes.append(METHOD)
    return " ".join(notes)


# --- charts
def prices_usd(data: PinkSheet) -> None:
    """Each commodity's price in US dollars: full history and the last RECENT_MONTHS months."""
    for commodity in data.prices.columns:
        print(commodity)
        usd = data.prices[commodity].dropna().astype(float)
        multi_start(
            usd,
            function=line_plot_finalise,
            starts=plot_times,
            title=f"{commodity} (USD)",
            y0=True,
            ylabel=data.units[commodity],
            lfooter=_lfooter(commodity, usd),
            rfooter=SOURCE,
            annotate=True,
        )


def prices_aud(data: PinkSheet) -> None:
    """Each commodity's price (not the indexes) in Australian dollars, at the RBA's AUD/USD rate."""
    for commodity in data.prices.columns:
        if INDEX_MARK in str(commodity).lower():
            continue
        aud = (data.prices[commodity].dropna().astype(float) / data.aud_usd).dropna()
        multi_start(
            aud,
            function=line_plot_finalise,
            starts=plot_times,
            title=f"{commodity} (AUD)",
            ylabel=data.units[commodity].replace("US$", "AU$"),
            y0=True,
            lfooter=_lfooter(commodity, aud, data.averaged_from),
            rfooter=AUD_SOURCE,
            annotate=True,
        )


# --- table of contents, in run order
CHARTS = (
    (prices_usd, ()),
    (prices_aud, ()),
)
