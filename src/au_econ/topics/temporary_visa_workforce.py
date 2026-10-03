"""Temporary visa holders in the Australian workforce, by visa class; net overseas migration by visa pathway.

The working stock is an estimate: Home Affairs' quarterly visa stock times the employment
rate of each visa class at the 2021 Census (ABS ACTEID). NOM is from ABS 3407.0, Table 4.1.
"""

# --- dependencies
import io
import re
from dataclasses import dataclass

import pandas as pd
from mgplot import bar_plot_finalise, line_plot_finalise

from au_econ.sources import homeaffairs
from au_econ.sources.abs import get_data_cube, latest_data_cube_url

# --- module contract
RELEASE = ("visa-workforce",)
TOPICS = ("migration",)
TITLE = "Temporary Visa Workforce"

# --- constants
SOURCE_HA = "Home Affairs: BP0019; ABS: ACTEID 2021"
SOURCE_ABS = "ABS: 3407.0"
THOUSAND = 1_000

# ACTEID 2021 employment rates of visa holders aged 15+, by visa class: applied to the
# current visa stock as a proxy for the share working
EMPLOYMENT_RATE = {
    "Bridging": 0.674,  # other-temporary rate
    "Crew and Transit": 0.0,  # no work rights
    "Other Temporary": 0.674,
    "Special Category": 0.679,  # NZ SCV (444)
    "Student": 0.636,
    "Temporary Graduate": 0.674,
    "Temporary Protection": 0.674,
    "Temporary Resident (Other Employment)": 0.843,  # temporary-skilled rate
    "Temporary Resident (Skilled Employment)": 0.843,
    "Visitor": 0.0,  # no work rights
    "Working Holiday Maker": 0.857,
}
LABEL = {  # visa class: chart label, in stacking order
    "Special Category": "NZ SCV (444)",
    "Student": "Student (500)",
    "Working Holiday Maker": "Working Holiday (417/462)",
    "Temporary Resident (Skilled Employment)": "Skilled (482/SID/186)",
    "Bridging": "Bridging",
    "Temporary Graduate": "Graduate (485)",
    "Temporary Resident (Other Employment)": "Other Employment",
    "Other Temporary": "Other Temporary",
    "Temporary Protection": "Temporary Protection",
}
STACK_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]
NOM_COLORS = ["#1a6fbf", "#a6cee3", "#ff7f0e", "#d62728", "#2ca02c", "#7f7f7f", "#9467bd"]
STACK_LEGEND = {"loc": "upper left", "fontsize": "x-small", "ncol": 2}
BAR_WIDTH = 0.9

# ABS 3407.0 Table 4.1: NOM by visa pathway, financial years from 2004-05
MIGRATION_RELEASE = "https://www.abs.gov.au/statistics/people/population/overseas-migration/latest-release"
MIGRATION_CUBE = "34070DO004"  # the data cube holding Table 4.1
MIGRATION_SHEET = "Table 4.1"
# Table 4.1 is read by its labels, not row numbers, so a new release's layout cannot be misread
HEADER = "Direction"  # first cell of the row holding the financial years
YEAR = r"^\d{4}-\d{2}"  # a financial-year column heading, e.g. 2004-05
ARRIVALS, DEPARTURES = "Overseas migrant arrivals", "Overseas migrant departures"  # block headings
GROUP_COLUMN, DETAIL_COLUMN = 1, 2  # the most detailed label is in DETAIL_COLUMN, else GROUP_COLUMN
FOOTNOTE = r"\([a-z]\)"  # footnote markers such as (h), renumbered between releases
PATHWAYS = {  # chart label: Table 4.1 rows summed
    "Permanent: Skilled": ["Skilled (permanent)"],
    "Permanent: Family/Hum/Other": ["Family", "Special eligibility & humanitarian", "Other (permanent)"],
    "Temporary: Student": ["Student"],
    "Temporary: Skilled": ["Skilled (temporary)"],
    "Temporary: Working holiday": ["Working holiday"],
    "Temporary: Visitors/Other": ["Visitors", "Other (temporary)"],
    "NZ citizens (444)": ["New Zealand citizens (subclass 444)"],
}


@dataclass(frozen=True)
class VisaData:
    """Visa stock and estimated working stock (quarters by visa class), and NOM by pathway (financial years)."""

    visa_stock: pd.DataFrame
    working: pd.DataFrame
    nom: pd.DataFrame
    stock_date: pd.Timestamp  # the latest visa stock's actual date (it can fall mid-quarter)


# --- data
def _row_starting(raw: pd.DataFrame, text: str, after: int = -1) -> int:
    """Return the position of the one row, below row `after`, whose first cell starts with text."""
    rows = [i for i, cell in enumerate(raw.iloc[:, 0]) if i > after and str(cell).startswith(text)]
    if len(rows) != 1:
        raise ValueError(f"ABS 3407.0 {MIGRATION_SHEET}: expected one row starting {text!r}, found {len(rows)}")
    return rows[0]


def _migration_block(
    raw: pd.DataFrame, heading: str, header_row: int, year_columns: list[int], years: list[str]
) -> pd.DataFrame:
    """Extract a block below the header row (heading to the next blank row), keyed by its most detailed label."""
    start = _row_starting(raw, heading, after=header_row)
    blank = [i for i in range(start + 1, len(raw)) if raw.iloc[i].isna().all()]
    block = raw.iloc[start : blank[0] if blank else len(raw)]
    labels = block.iloc[:, DETAIL_COLUMN].where(block.iloc[:, DETAIL_COLUMN].notna(), block.iloc[:, GROUP_COLUMN])
    labels = labels.str.replace(FOOTNOTE, "", regex=True).str.strip()
    keep = labels.notna() & ~labels.astype(str).str.startswith("Total")
    values = block.iloc[:, year_columns].apply(pd.to_numeric, errors="coerce")
    frame = pd.DataFrame(values.to_numpy(), index=labels.to_numpy(), columns=years)
    return frame.loc[keep.to_numpy()].dropna(how="all")


def _nom_by_pathway() -> pd.DataFrame:
    """Net overseas migration (arrivals less departures) by visa pathway, financial years."""
    url = latest_data_cube_url(MIGRATION_RELEASE, MIGRATION_CUBE)
    print(f"Overseas migration workbook: {url.rsplit('/', 1)[-1]}")
    raw = pd.read_excel(io.BytesIO(get_data_cube(url)), sheet_name=MIGRATION_SHEET, header=None)
    header_row = _row_starting(raw, HEADER)
    header = raw.iloc[header_row].astype(str).str.replace(r"\(.*\)", "", regex=True).str.strip()
    year_columns = [i for i, cell in enumerate(header) if isinstance(cell, str) and re.match(YEAR, cell)]
    years = [header.iloc[i] for i in year_columns]
    arrivals = _migration_block(raw, ARRIVALS, header_row, year_columns, years)
    departures = _migration_block(raw, DEPARTURES, header_row, year_columns, years)
    common = arrivals.index.intersection(departures.index)
    nom = arrivals.loc[common] - departures.loc[common]
    pathways = pd.DataFrame(index=nom.columns)
    for label, rows in PATHWAYS.items():
        pathways[label] = nom.loc[rows].sum() if len(rows) > 1 else nom.loc[rows[0]]
    return pathways


def fetch() -> VisaData:
    """Fetch the visa stock and NOM once; estimate the working stock; report the latest quarter."""
    stock, stock_date = homeaffairs.get_visa_stock()
    working = stock.copy()
    for visa_class in working.columns:
        working[visa_class] = working[visa_class] * EMPLOYMENT_RATE.get(visa_class, 0.0)
    latest = stock.index[-1]
    print(f"Visa stock at {latest}:")
    print(stock.iloc[-1].sort_values(ascending=False).map(lambda x: f"{x:>12,.0f}"))
    print(f"\nEstimated working stock at {latest}:")
    print(working.iloc[-1].sort_values(ascending=False).map(lambda x: f"{x:>12,.0f}"))
    print(f"\nTotal visa stock:    {stock.iloc[-1].sum():>14,.0f}")
    print(f"Total working stock: {working.iloc[-1].sum():>14,.0f}")
    return VisaData(visa_stock=stock, working=working, nom=_nom_by_pathway(), stock_date=stock_date)


def _stock_data_to(data: VisaData) -> str:
    """Footer note giving the latest visa stock's actual date."""
    return f"Data to {data.stock_date.strftime('%-d %b %Y')}."


def _incomplete_quarter(data: VisaData) -> str:
    """Header note when the latest stock falls before its quarter's end (its bar is labelled by quarter)."""
    if data.stock_date.is_quarter_end:
        return ""  # an empty header draws nothing
    return f"Final quarter incomplete: stock at {data.stock_date.strftime('%-d %b %Y')}."


# --- charts
def workforce_by_visa(data: VisaData) -> None:
    """Estimated temporary visa holders in the workforce, stacked by visa class."""
    bar_plot_finalise(
        data.working[list(LABEL)].rename(columns=LABEL) / THOUSAND,
        stacked=True,
        annotate=False,
        width=BAR_WIDTH,
        color=STACK_COLORS,
        title="Temporary visa holders in the Australian workforce, by visa class",
        ylabel="Number ('000s)",
        rfooter=SOURCE_HA,
        lfooter=f"Australia. Visa stock x ABS ACTEID 2021 employment rate by class. {_stock_data_to(data)}",
        rheader=_incomplete_quarter(data),
        legend=STACK_LEGEND,
    )


def workforce_total(data: VisaData) -> None:
    """Estimated temporary visa holders in the workforce, in total."""
    line_plot_finalise(
        (data.working.sum(axis=1) / THOUSAND).rename("Total"),
        annotate=True,
        rounding=0,
        title="Estimated temporary visa holders in the Australian workforce: total",
        ylabel="Number ('000s)",
        rfooter=SOURCE_HA,
        lfooter=f"Australia. Visa stock x ACTEID 2021 employment rate by class. {_stock_data_to(data)}",
        rheader=_incomplete_quarter(data),
        legend=False,
    )


def nom_by_visa(data: VisaData) -> None:
    """Net overseas migration, stacked by visa pathway."""
    bar_plot_finalise(
        (data.nom / THOUSAND).rename_axis("Year"),
        stacked=True,
        annotate=False,
        width=BAR_WIDTH,
        label_rotation=45,
        color=NOM_COLORS,
        title="Australia: Net overseas migration by visa pathway",
        ylabel="Persons ('000s)",
        rfooter=SOURCE_ABS,
        lfooter=(
            "Australia. Financial years. 12/16-month rule basis. NOM = arrivals - departures. "
            f"Data to {data.nom.index[-1]}."
        ),
        legend=STACK_LEGEND,
        y0=True,
    )


# --- table of contents, in run order
CHARTS = (
    (workforce_by_visa, ()),
    (workforce_total, ()),
    (nom_by_visa, ()),
)
