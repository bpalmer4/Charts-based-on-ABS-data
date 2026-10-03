"""Australian Petroleum Statistics (DCCEEW): fuel security cover, production, refining, trade, LNG and fuel prices.

Released monthly, about six to seven weeks after the reference month.
"""

# --- dependencies
from dataclasses import dataclass
from functools import partial

import pandas as pd
from mgplot import line_plot_finalise

from au_econ.sources import dcceew

# --- module contract
RELEASE = ("petroleum",)
TOPICS = ("commodities",)
TITLE = "Australian Petroleum Statistics"

# --- constants
SOURCE = "DCCEEW: APS"  # Australian Petroleum Statistics
LFOOTER = "Australia. "
MONTHLY_LFOOTER = LFOOTER + "Monthly."
IEA_LFOOTER = (
    LFOOTER
    + "Monthly. Crude-oil-equivalent aggregate. "
    + "Cover based on prior year ave daily net imports, updated each April."
)
DNI_LFOOTER = LFOOTER + "Average daily net imports for previous calendar year. Updated each April."
COVER_LFOOTER = LFOOTER + "Monthly. Stocks / rolling 12-month ave daily consumption."
PRICE_LFOOTER = LFOOTER + "Quarterly average retail prices, sales-weighted by state."
LEGEND = {"loc": "best", "fontsize": "small"}
ML_MONTH = "Megalitres / Month"
CPL = "Cents per litre"
KEY_PRODUCTS = [
    "Crude oil and refinery feedstocks (days)",
    "Automotive gasoline (days)",
    "Diesel oil (days)",
    "Aviation turbine fuel (days)",
]
PETROL, DIESEL = "Regular unleaded petrol (91 RON)", "Automotive diesel"


@dataclass(frozen=True)
class PetroleumData:
    """The workbook's monthly sheets used here, and its quarterly fuel prices."""

    production: pd.DataFrame
    sales: pd.DataFrame
    imports: pd.DataFrame
    exports: pd.DataFrame
    refinery: pd.DataFrame
    consumption_cover: pd.DataFrame
    iea_cover: pd.DataFrame
    fuel_prices: pd.DataFrame


# --- data
def fetch() -> PetroleumData:
    """Fetch the latest workbook once; every chart function receives its sheets."""
    workbook = dcceew.get_workbook()
    sheet = partial(dcceew.monthly_sheet, workbook)
    data = PetroleumData(
        production=sheet("Petroleum production"),
        sales=sheet("Sales of products"),
        imports=sheet("Imports volume"),
        exports=sheet("Exports volume"),
        refinery=sheet("Refinery production"),
        consumption_cover=sheet("Consumption cover"),
        iea_cover=sheet("IEA days net import cover"),
        fuel_prices=dcceew.fuel_price_sheet(workbook),
    )
    print(f"Data from {data.production.index[0]} to {data.production.index[-1]}")
    print(f"Fuel prices from {data.fuel_prices.index[0]} to {data.fuel_prices.index[-1]}")
    return data


# --- helpers
def _data_to(data: pd.Series | pd.DataFrame) -> str:
    """Name the last period with data in what is plotted (sheets end at different points)."""
    valid = data.dropna(how="all") if isinstance(data, pd.DataFrame) else data.dropna()
    last = valid.index[-1]
    if not isinstance(last, pd.Period):
        raise TypeError("expected a PeriodIndex")
    return f"Data to {last.strftime('%b %Y') if last.freqstr.startswith('M') else last}."


def _line[T: (pd.Series, pd.DataFrame)](
    data: T, *, title: str, ylabel: str, lfooter: str, legend: bool = False
) -> None:
    """Draw a line chart with the module's footers, the left one ending "Data to ...".

    Without a legend, a single series gets none, as mgplot's default would give it.
    """
    line_plot_finalise(
        data,
        title=title,
        ylabel=ylabel,
        legend=LEGEND if legend else False,
        rfooter=SOURCE,
        lfooter=f"{lfooter} {_data_to(data)}",
    )


# --- charts
def import_cover(data: PetroleumData) -> None:
    """IEA days of net import cover, and the daily net imports behind it."""
    _line(
        data.iea_cover["IEA days of net import coverage"],
        title="IEA Days of Net Import Coverage",
        ylabel="Days",
        lfooter=IEA_LFOOTER,
    )
    _line(
        data.iea_cover["Daily Net Imports (kT/day)"],
        title="Daily Net Imports",
        ylabel="kT / day",
        lfooter=DNI_LFOOTER,
    )


def consumption_cover(data: PetroleumData) -> None:
    """Days of consumption cover: the key products together, then one by one."""
    key_products = data.consumption_cover[KEY_PRODUCTS].rename(columns=lambda x: x.replace(" (days)", ""))
    _line(
        key_products,
        title="Consumption Cover by Product",
        ylabel="Days of supply",
        lfooter=COVER_LFOOTER,
        legend=True,
    )
    for product in key_products.columns:
        _line(
            key_products[product],
            title=f"Consumption Cover: {product}",
            ylabel="Days of supply",
            lfooter=COVER_LFOOTER,
        )


def trade_and_sales(data: PetroleumData) -> None:
    """Total oil imports and exports against domestic sales."""
    trade = pd.DataFrame(
        {
            "Total imports": data.imports["Total oil imports (ML)"],
            "Total exports": data.exports["Total oil exports (inc. ships' and aircraft stores) (ML)"],
            "Domestic sales": data.sales["Total (ML)"],
        }
    )
    _line(
        trade,
        title="Petroleum: Imports, Exports and Domestic Sales",
        ylabel=ML_MONTH,
        lfooter=MONTHLY_LFOOTER + " Exports exclude LNG. Exports are mainly crude oil & condensate.",
        legend=True,
    )


def production_and_refining(data: PetroleumData) -> None:
    """Domestic crude production, refinery throughput (refinery closures) and the indigenous crude share."""
    _line(
        data.production["Crude oil & condensate (ML)"],
        title="Domestic Crude Oil & Condensate Production",
        ylabel=ML_MONTH,
        lfooter=MONTHLY_LFOOTER,
    )
    _line(
        data.refinery["Total input (ML)"],
        title="Refinery Total Input",
        ylabel=ML_MONTH,
        lfooter=MONTHLY_LFOOTER,
    )
    _line(
        data.refinery["Percentage indigenous: Total input (%)"],
        title="Refinery Input: Indigenous Crude Share",
        ylabel="Per cent",
        lfooter=MONTHLY_LFOOTER,
    )


def import_dependence(data: PetroleumData) -> None:
    """Domestic crude against imported feedstock and refined product; refined products' share of imports."""
    sources = pd.DataFrame(
        {
            "Domestic crude oil & condensate": data.production["Crude oil & condensate (ML)"],
            "Imported crude & feedstock": data.imports["Crude oil & other refinery feedstocks (ML)"],
            "Imported refined products": data.imports["Total refined petroleum products (ML)"],
        }
    )
    _line(
        sources,
        title="Domestic Production vs Imports",
        ylabel=ML_MONTH,
        lfooter=MONTHLY_LFOOTER,
        legend=True,
    )
    share = data.imports["Total refined petroleum products (ML)"] / data.imports["Total oil imports (ML)"] * 100
    _line(
        share,
        title="Refined Product Imports as Share of Total Oil Imports",
        ylabel="Per cent",
        lfooter=MONTHLY_LFOOTER,
    )


def lng_exports(data: PetroleumData) -> None:
    """LNG exports, from the production sheet (Mm3) and the exports sheet (ML)."""
    by_gas = pd.to_numeric(data.production["LNG exports (Mm3)"], errors="coerce").dropna()
    _line(by_gas, title="LNG Exports", ylabel="Mm³ / Month", lfooter=MONTHLY_LFOOTER)
    by_liquid = pd.to_numeric(data.exports["LNG (ML)"], errors="coerce").dropna()
    _line(by_liquid, title="LNG Exports (Volume)", ylabel=ML_MONTH, lfooter=MONTHLY_LFOOTER)


def retail_fuel_prices(data: PetroleumData) -> None:
    """Quarterly retail fuel prices: every product, then petrol against diesel."""
    prices = data.fuel_prices.rename(columns=lambda x: x.replace(" (cpl)", ""))
    _line(
        prices,
        title="Australian Retail Fuel Prices",
        ylabel=CPL,
        lfooter=PRICE_LFOOTER,
        legend=True,
    )
    _line(
        prices[[PETROL, DIESEL]],
        title="Petrol vs Diesel Prices",
        ylabel=CPL,
        lfooter=PRICE_LFOOTER,
        legend=True,
    )


# --- table of contents, in run order
CHARTS = (
    (import_cover, ()),
    (consumption_cover, ()),
    (trade_and_sales, ()),
    (production_and_refining, ()),
    (import_dependence, ()),
    (lng_exports, ()),
    (retail_fuel_prices, ()),
)
