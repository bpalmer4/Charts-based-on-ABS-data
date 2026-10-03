"""Quarterly Update of Australia's National Greenhouse Gas Inventory (DCCEEW): emissions and emissions per head.

Released quarterly, about five months after the reference quarter. The workbook's Figure 1
also carries a preliminary estimate for the quarter after the one it reports.
"""

# --- dependencies
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mgplot import line_plot_finalise, seastrend_plot_finalise

from au_econ.charting.footers import SERIES_TYPE_NOTES
from au_econ.series.population import ERP_CATALOGUE, get_erp
from au_econ.sources import dcceew

if TYPE_CHECKING:
    import pandas as pd

# --- module contract
RELEASE = ("nggi",)
TOPICS = ("environment",)
TITLE = "Greenhouse Gas Emissions"

# --- constants
EMISSIONS_SHEET = "Figure 1"  # million tonnes of carbon dioxide equivalent (Mt CO2-e)
ACTUAL, SEASONALLY_ADJUSTED, TREND = (
    "Actual emissions",
    "Seasonally adjusted and weather normalised",
    "Trend",
)
ERP_PROJECTION_QUARTERS = 2  # ERP lags the inventory; extend it at its latest growth rate
TONNES_PER_MEGATONNE = 1_000_000
LFOOTER = "Australia. "
SOURCE = "DCCEEW: NGGI"  # National Greenhouse Gas Inventory
EMISSIONS_TITLE = "Australia's Greenhouse Gas Emissions"
EMISSIONS_YLABEL = "Million tonnes $CO_{{2}}$-e / Quarter"


@dataclass(frozen=True)
class EmissionsData:
    """Quarterly emissions (Mt CO2-e), the quarter the update reports, and national ERP (persons)."""

    emissions: pd.DataFrame
    reported_quarter: pd.Period
    population: pd.Series


# --- data
def fetch() -> EmissionsData:
    """Fetch the latest inventory workbook and national ERP."""
    workbook, quarter = dcceew.get_inventory_workbook()
    emissions = dcceew.inventory_sheet(workbook, EMISSIONS_SHEET)
    missing = {ACTUAL, SEASONALLY_ADJUSTED, TREND} - set(emissions.columns)
    if emissions.empty or missing:
        raise ValueError(f"Greenhouse gas inventory {EMISSIONS_SHEET}: empty or missing {sorted(missing)}")
    population, _ = get_erp(ERP_PROJECTION_QUARTERS)
    return EmissionsData(emissions=emissions, reported_quarter=quarter, population=population)


# --- helpers
def _data_to(series: pd.Series, data: EmissionsData) -> str:
    """Name the series' last quarter, marked preliminary when it is past the quarter the update reports."""
    last = series.dropna().index[-1]
    return f"Data to {last}{' (preliminary)' if last > data.reported_quarter else ''}."


# --- charts
def emissions_level(data: EmissionsData) -> None:
    """Total quarterly emissions, original series."""
    line_plot_finalise(
        data.emissions[ACTUAL],
        tag="lineplot",
        annotate=True,
        title=EMISSIONS_TITLE,
        ylabel=EMISSIONS_YLABEL,
        lfooter=f"{LFOOTER}{SERIES_TYPE_NOTES['Original']} {_data_to(data.emissions[ACTUAL], data)}",
        rfooter=SOURCE,
        legend=True,
    )


def emissions_seastrend(data: EmissionsData) -> None:
    """Quarterly emissions: seasonally adjusted against trend."""
    seastrend_plot_finalise(
        data.emissions[[SEASONALLY_ADJUSTED, TREND]],
        tag="seastrend",
        title=EMISSIONS_TITLE,
        ylabel=EMISSIONS_YLABEL,
        lfooter=f"{LFOOTER}{_data_to(data.emissions[TREND], data)}",
        rfooter=SOURCE,
        legend=True,
    )


def emissions_per_capita(data: EmissionsData) -> None:
    """Trend emissions per head of population, in tonnes per quarter."""
    per_capita = data.emissions[TREND] * TONNES_PER_MEGATONNE / data.population
    per_capita.name = "Trend per Capita"
    line_plot_finalise(
        per_capita,
        tag="lineplot",
        annotate=True,
        title=f"{EMISSIONS_TITLE} per Capita",
        ylabel="Tonnes $CO_{2}$-e / Quarter",
        lfooter=(
            f"{LFOOTER}{SERIES_TYPE_NOTES['Trend']} ERP projected {ERP_PROJECTION_QUARTERS} quarters. "
            f"{_data_to(per_capita, data)}"
        ),
        rfooter=f"{SOURCE}; ABS: {ERP_CATALOGUE}",
        legend=True,
    )


# --- table of contents, in run order
CHARTS = (
    (emissions_level, ()),
    (emissions_seastrend, ()),
    (emissions_per_capita, ()),
)
