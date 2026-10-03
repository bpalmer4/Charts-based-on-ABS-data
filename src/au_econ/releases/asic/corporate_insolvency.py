"""ASIC corporate insolvency statistics: first-time external administrations, by state and industry sector.

Released monthly; the latest month or two are provisional, so EXCLUDE_LATEST months are
dropped from every chart. The current series is joined onto the 2022 history workbooks,
the current data winning where they overlap.
"""

# --- dependencies
import io
import textwrap
from dataclasses import dataclass

import pandas as pd
from mgplot import (
    abbreviate_state,
    bar_plot,
    bar_plot_finalise,
    finalise_plot,
    get_color,
    line_plot_finalise,
    multi_start,
    state_names,
)

from au_econ.analysis.decompose import decompose
from au_econ.charting.footers import SERIES_TYPE_NOTES
from au_econ.sources import asic

# --- module contract
RELEASE = ("asic",)
TOPICS = ("insolvency",)
TITLE = "Corporate Insolvencies"

# --- constants
EXCLUDE_LATEST = 1  # months of provisional data to exclude
SOURCE = "ASIC: series 1, 1A, 2"  # the insolvency statistics workbooks used
LFOOTER = "Australia. First-time external administration or controller appointment. "
PROVISIONAL_LHEADER = "Last month of (provisional) data excluded" if EXCLUDE_LATEST else ""
YLABEL = "First-time Insolvencies/Month"
RECENT_START = pd.Period("2019-01", freq="M")
plot_times = 0, RECENT_START
AUSTRALIA = "Australia"
STATES = [*state_names, AUSTRALIA]
TREND_COLUMNS = ["Trend", "Seasonally Adjusted"]
TREND_WIDTHS = (2.5, 1)
MONTHS_IN_YEAR = 12
BASE_START, BASE_END = 2015, 2019  # the pre-pandemic years growth is measured against
SECTOR_CUTOFF = 150  # insolvencies a year in the base period, for the sector growth chart
ORIGINAL_LFOOTER = f"{LFOOTER}{SERIES_TYPE_NOTES['Original']} "
SECTOR_TITLE_WIDTH = 60
SECTOR_BY_STATE_START = 2019

# sheet layout
CURRENT_STATE_SECTOR_SHEET, CURRENT_STATE_SHEET = "1.4.2", "1.3"
CURRENT_SKIP_ROWS = 10
HISTORY_STATE_SHEET, HISTORY_STATE_SKIP_ROWS = "1.2", 4
HISTORY_SECTOR_SHEET, HISTORY_SECTOR_SKIP_ROWS = "1A.1.2", 5
MIN_VALUES_PER_ROW = 8  # history rows with fewer are headings or notes
CURRENT_LEADING_COLUMNS = 4  # period and descriptive columns ahead of the sector counts
PERIOD_MONTH, PERIOD_YEAR = "Period (Month)", "Period (Calendar Year)"
SHEET_1_3_MONTH, SHEET_1_3_YEAR = "Period(Month)", "Period(Calendar Year)"
PLACE = "Principal Place Of Business (State Or Territory)"
PERIOD_AND_REGION = "Period and Region"
SECOND_HALF_MONTHS = ("July", "August", "September", "October", "November", "December")
EN_DASH = chr(0x2013)  # ASIC joins "Fis" to each sub-sector with an en dash
FIS_COLUMNS = [  # the 2022 history splits Financial and Insurance Services into sub-sectors
    "Financial and Insurance Services",
    *(
        f"Fis{EN_DASH}{sub_sector}"
        for sub_sector in (
            "Credit Provider",
            "Deposit Taking Institutions",
            "Insurance",
            "Managed Investments",
            "Other Financial Services",
            "Superannuation",
        )
    ),
]
FIS_POSITION = 9  # where the amalgamated column goes, matching the current workbook
HISTORY_SECTOR_RENAMES = {
    "Other (Business and Personal) Services": "Other Services",
    "Information Media and Tele- Communications": "Information Media and Telecommunications",
    "Total": AUSTRALIA,
}
SECTORS = [
    "Accommodation and Food Services",
    "Administrative and Support Services",
    "Agriculture, Forestry and Fishing",
    "Arts and Recreation Services",
    "Construction",
    "Education and Training",
    "Electricity, Gas, Water and Waste Services",
    "Financial and Insurance Services",
    "Health Care and Social Assistance",
    "Information Media and Telecommunications",
    "Manufacturing",
    "Mining",
    "Other Services",
    "Professional, Scientific and Technical Services",
    "Public Administration and Safety",
    "Rental, Hiring and Real Estate Services",
    "Retail Trade",
    "Transport, Postal and Warehousing",
    "Wholesale Trade",
    "Unknown",
    AUSTRALIA,
]
TREND_SECTOR = "Construction"
SECTORS_OF_INTEREST = [
    "Accommodation and Food Services",
    "Construction",
    "Rental, Hiring and Real Estate Services",
    "Financial and Insurance Services",
    "Manufacturing",
    "Retail Trade",
    "Other Services",
    AUSTRALIA,
]


@dataclass(frozen=True)
class InsolvencyData:
    """Monthly first-time insolvencies: by state (history and current), and by (month, state) and sector."""

    state_history: pd.DataFrame
    state_current: pd.DataFrame
    sector_current: pd.DataFrame
    sector_combined: pd.DataFrame


# --- data
def _fix_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Standardise ASIC column names: title case, no line breaks, "and" for "&", single spaces."""
    renames = {}
    for column in frame.columns:
        name = str(column).title().replace("\n", "").replace("&", " and ").replace("And", "and")
        renames[column] = name.replace("  ", " ")
    return frame.rename(columns=renames)


def _month_index(years: pd.Series, months: pd.Series) -> pd.PeriodIndex:
    """Return a monthly PeriodIndex from calendar-year and month-name columns."""
    return pd.PeriodIndex(years.astype(int).astype(str) + "-" + months, freq="M")


def _state_current(workbook: bytes) -> pd.DataFrame:
    """Monthly state totals from the sector-by-state detail (sheet 1.4.2), checked against sheet 1.3.

    From mid 2025, sheet 1.3 collapses each September quarter into one row, so the monthly
    totals come from 1.4.2, which keeps every month; they must agree with 1.3 wherever
    1.3 still has monthly rows.
    """
    raw = _fix_columns(
        pd.read_excel(io.BytesIO(workbook), sheet_name=CURRENT_STATE_SECTOR_SHEET, skiprows=CURRENT_SKIP_ROWS)
    )
    raw = raw.loc[
        raw[PERIOD_MONTH].notna() & ~raw[PERIOD_MONTH].astype(str).str.contains("Total") & raw[PLACE].notna()
    ]
    raw = raw.assign(month=_month_index(raw[PERIOD_YEAR], raw[PERIOD_MONTH]))
    states = raw.pivot_table(index="month", columns=PLACE, values="Total", aggfunc="sum")
    states.columns.name = None
    states = states.fillna(0)  # a state-month absent from the detail means no insolvencies
    states[AUSTRALIA] = states.sum(axis=1)

    check = _fix_columns(
        pd.read_excel(io.BytesIO(workbook), sheet_name=CURRENT_STATE_SHEET, skiprows=CURRENT_SKIP_ROWS)
    ).dropna(how="all", axis=1)
    check = check.loc[check[SHEET_1_3_MONTH].notna()]
    check.index = _month_index(check[SHEET_1_3_YEAR], check[SHEET_1_3_MONTH])
    check = check.rename(columns={"Total": AUSTRALIA})
    common = states.index.intersection(check.index)
    for column in states.columns:
        if not (states.loc[common, column] == check.loc[common, column]).all():
            raise ValueError(
                f"ASIC sheets {CURRENT_STATE_SECTOR_SHEET} and {CURRENT_STATE_SHEET} disagree for {column}"
            )
    return states


def _state_history(workbook: bytes) -> pd.DataFrame:
    """Monthly state totals from the 2022 series 1 workbook, whose months sit under financial-year headings."""
    frame = _fix_columns(
        pd.read_excel(io.BytesIO(workbook), sheet_name=HISTORY_STATE_SHEET, skiprows=HISTORY_STATE_SKIP_ROWS)
    )
    frame = frame.dropna(axis=0, how="all").iloc[1:-1]  # drop the heading and the closing note
    frame["Fin Year"] = frame.loc[frame[AUSTRALIA].isna(), "Period"]
    frame["Fin Year"] = frame["Fin Year"].ffill()
    frame = frame.dropna(thresh=MIN_VALUES_PER_ROW, axis=0)
    first_year, second_year = frame["Fin Year"].str.split("-").str[0], frame["Fin Year"].str.split("-").str[1]
    frame["Year"] = first_year.where(frame["Period"].isin(SECOND_HALF_MONTHS), other=second_year)
    frame.index = pd.PeriodIndex(frame["Year"] + "-" + frame["Period"], freq="M")
    return frame


def _sector_history(workbook: bytes) -> pd.DataFrame:
    """Monthly insolvencies by (month, state) and sector from the 2022 series 1A workbook."""
    frame = _fix_columns(
        pd.read_excel(io.BytesIO(workbook), sheet_name=HISTORY_SECTOR_SHEET, skiprows=HISTORY_SECTOR_SKIP_ROWS)
    )
    fis = frame[FIS_COLUMNS].sum(axis=1, skipna=True)
    frame = frame.drop(columns=FIS_COLUMNS)
    columns = list(frame.columns)
    columns.insert(FIS_POSITION, FIS_COLUMNS[0])
    frame = frame.assign(**{FIS_COLUMNS[0]: fis})[columns].rename(columns=HISTORY_SECTOR_RENAMES)

    label = frame[PERIOD_AND_REGION]
    keep = (
        label.notna()
        & ~label.astype(str).str.contains("MONTHLY")
        & ~label.astype(str).str.contains("Total for")
        & ~label.astype(str).str.contains("Australian Securities & Investments Commission")
        & ~label.astype(str).str.contains(r"\d{4}-\d{4}")
    )
    frame = frame.loc[keep]
    months = frame[PERIOD_AND_REGION].where(frame[PERIOD_AND_REGION].str.contains(r"[A-Za-z]*\s\d{4}")).ffill()
    month = pd.PeriodIndex(months.str.split(" ").str[::-1].str.join("-"), freq="M")
    state = frame[PERIOD_AND_REGION].replace({name: abbreviate_state(name) for name in state_names})
    frame.index = pd.MultiIndex.from_arrays([month, state], names=["month", "State"])
    return frame.drop(columns=[PERIOD_AND_REGION]).dropna(thresh=MIN_VALUES_PER_ROW)


def _sector_current(workbook: bytes) -> pd.DataFrame:
    """Monthly insolvencies by (month, state) and sector from the current workbook (sheet 1.4.2)."""
    frame = _fix_columns(
        pd.read_excel(io.BytesIO(workbook), sheet_name=CURRENT_STATE_SECTOR_SHEET, skiprows=CURRENT_SKIP_ROWS)
    ).dropna(how="all", axis=1)
    frame = frame.loc[frame[PLACE].notna()]
    month = _month_index(frame[PERIOD_YEAR], frame[PERIOD_MONTH])
    state = frame[PLACE].replace({name: abbreviate_state(name) for name in state_names})
    frame = frame.drop(columns=frame.columns[:CURRENT_LEADING_COLUMNS]).rename(columns={"Total": AUSTRALIA})
    frame.index = pd.MultiIndex.from_arrays([month, state], names=["month", "State"])
    return frame.drop(columns=[PLACE])


def _combine[T: (pd.Series, pd.DataFrame)](history: T, current: T) -> T:
    """Join history and current data, the current winning where they overlap."""
    combined = pd.concat([history, current], axis=0)
    return combined[~combined.index.duplicated(keep="last")].sort_index()


def fetch() -> InsolvencyData:
    """Fetch the current and 2022 history workbooks; parse states and sectors."""
    current = asic.get_current_workbook()
    state_current = _state_current(current)
    sector_current = _sector_current(current)
    state_history = _state_history(asic.get_history_workbook(asic.HISTORY_SERIES_1))
    sector_history = _sector_history(asic.get_history_workbook(asic.HISTORY_SERIES_1A))
    for name, frame in (("state", state_current), ("sector", sector_current)):
        if frame.empty:
            raise ValueError(f"ASIC current {name} data is empty")
    return InsolvencyData(
        state_history=state_history,
        state_current=state_current,
        sector_current=sector_current,
        sector_combined=_combine(sector_history, sector_current),
    )


# --- helpers
def _drop_provisional[T: (pd.Series, pd.DataFrame)](data: T) -> T:
    """Drop the provisional months at the end."""
    return data.iloc[:-EXCLUDE_LATEST] if EXCLUDE_LATEST else data


def _when(index: pd.Index) -> str:
    """Return the last period of an index as e.g. "Aug-2026"."""
    last = index[-1]
    if not isinstance(last, pd.Period):
        raise TypeError("Expected a PeriodIndex")
    return last.strftime("%b-%Y")


def _data_to(index: pd.Index) -> str:
    """Return the footer note for the last period of an index, e.g. "Data to Aug 2026."."""
    return f"Data to {_when(index).replace('-', ' ')}."


def _years(index: pd.Index) -> pd.Index:
    """Return the calendar year of each period in a PeriodIndex."""
    if not isinstance(index, pd.PeriodIndex):
        raise TypeError("Expected a PeriodIndex")
    return index.year


def _by_state(frame: pd.DataFrame, sector: str) -> pd.DataFrame:
    """One sector's insolvencies as months by states."""
    if frame.index.duplicated().any():
        raise ValueError("ASIC sector data has duplicate (month, state) rows")
    table = (
        frame[[sector]]
        .reset_index()
        .pivot_table(index="month", columns="State", values=sector, aggfunc="first", dropna=False)
    )
    table.index.name, table.columns.name = None, None
    return table


def _sector_total(data: InsolvencyData, sector: str) -> pd.Series:
    """One sector's national monthly insolvencies, provisional months dropped."""
    return _drop_provisional(_by_state(data.sector_combined, sector).sum(axis=1))


def _growth(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Growth (%) of the latest 12 months over the base-period annual average; and that average."""
    end = frame.index[-1]
    latest = frame.loc[end - (MONTHS_IN_YEAR - 1) : end].sum()
    years = _years(frame.index)
    base = frame.loc[(years >= BASE_START) & (years <= BASE_END)].sum() / (BASE_END - BASE_START + 1)
    return ((latest / base) - 1) * 100, base


def _trend_chart(series: pd.Series, title: str) -> None:
    """Trend and seasonally adjusted, over the full history and from RECENT_START."""
    model = "multiplicative" if (series > 0).all() else "additive"  # multiplicative fails on zeros
    decomp = decompose(series, model=model, arima_extend=True)  # symmetric end weights; padding clipped
    multi_start(
        decomp[TREND_COLUMNS],
        function=line_plot_finalise,
        starts=plot_times,
        title=title,
        ylabel=YLABEL,
        width=TREND_WIDTHS,
        rfooter=SOURCE,
        lheader=PROVISIONAL_LHEADER,
        lfooter=LFOOTER + f"{model.capitalize()} decomposition. {_data_to(decomp.index)}",
        annotate=[True, False],
    )


# --- charts
def trend_by_state(data: InsolvencyData) -> None:
    """Trend and seasonally adjusted insolvencies: Australia, then each state."""
    for column in [AUSTRALIA, *state_names]:
        series = _drop_provisional(_combine(data.state_history[column], data.state_current[column]))
        _trend_chart(series, f"{TITLE}: {column}")


def _states_original(data: InsolvencyData) -> pd.DataFrame:
    """Original monthly insolvencies for each state and Australia, provisional months dropped."""
    return _drop_provisional(_combine(data.state_history[STATES], data.state_current[STATES]))


def original_by_state(data: InsolvencyData) -> None:
    """Original monthly insolvencies, each state and Australia."""
    combined = _states_original(data)
    data_to = _data_to(combined.index)
    for column in combined:
        line_plot_finalise(
            combined[column],
            title=f"{TITLE}: {column}",
            ylabel=YLABEL,
            width=1,
            rfooter=SOURCE,
            lheader=PROVISIONAL_LHEADER,
            lfooter=ORIGINAL_LFOOTER + data_to,
            tag="states",
            annotate=True,
        )


def growth_by_state(data: InsolvencyData) -> None:
    """Insolvencies in the latest 12 months against the base-period average, by state."""
    combined = _states_original(data)
    growth, _ = _growth(combined)
    growth = growth.sort_values()
    bar_plot_finalise(
        growth,
        horizontal=True,
        color=[get_color(state) for state in growth.index],
        annotate=True,
        rounding=1,
        title=f"{TITLE} in the 12 months\nto {_when(combined.index)} by State over {BASE_START}-{BASE_END} Ave",
        xlabel="Growth (%)",
        rfooter=SOURCE,
        lheader=PROVISIONAL_LHEADER,
        lfooter=LFOOTER,
    )


def original_by_sector(data: InsolvencyData) -> None:
    """Original monthly insolvencies, each industry sector and Australia."""
    for sector in SECTORS:
        series = _sector_total(data, sector)
        line_plot_finalise(
            series,
            title="\n".join(textwrap.wrap(f"{TITLE}: {sector}", SECTOR_TITLE_WIDTH)),
            ylabel=YLABEL,
            width=1,
            rfooter=SOURCE,
            lheader=PROVISIONAL_LHEADER,
            lfooter=ORIGINAL_LFOOTER + _data_to(series.index),
            annotate=True,
        )


def growth_by_sector(data: InsolvencyData) -> None:
    """Insolvencies in the latest 12 months against the base-period average, by larger industry sector."""
    sectors = pd.DataFrame({sector: _sector_total(data, sector) for sector in SECTORS})
    growth, base = _growth(sectors)
    growth = growth[base > SECTOR_CUTOFF].drop(AUSTRALIA)
    ax = bar_plot(growth.sort_values(), horizontal=True, annotate=True, rounding=1)
    ax.tick_params(axis="both", which="major", labelsize="x-small")
    finalise_plot(
        ax,
        title=(
            f"{TITLE} in the 12 months\nto {_when(sectors.index)} by Sector over "
            f"{BASE_START}-{str(BASE_END)[2:]} Ave"
        ),
        xlabel="Growth (%)",
        rfooter=SOURCE,
        rheader=(
            f"Industry sectors with more than {SECTOR_CUTOFF}/year insolvencies "
            f"in the {BASE_START}-{BASE_END} period."
        ),
        lheader=PROVISIONAL_LHEADER,
        lfooter=LFOOTER,
    )


def sector_trend(data: InsolvencyData) -> None:
    """Trend and seasonally adjusted insolvencies for one national industry sector."""
    _trend_chart(_sector_total(data, TREND_SECTOR), f"{TITLE}: {TREND_SECTOR}")


def sector_by_state(data: InsolvencyData) -> None:
    """Monthly insolvencies by state, stacked, for the sectors of interest, from SECTOR_BY_STATE_START."""
    for sector in SECTORS_OF_INTEREST:
        frame = _by_state(data.sector_current, sector)
        frame = _drop_provisional(frame[_years(frame.index) >= SECTOR_BY_STATE_START])
        bar_plot_finalise(
            frame,
            stacked=True,
            color=[get_color(state) for state in frame.columns],
            title="\n".join(textwrap.wrap(f"Monthly Insolvencies by State: {sector}", SECTOR_TITLE_WIDTH)),
            ylabel="First-time Insolvencies",
            legend={"loc": "upper left", "fontsize": "x-small", "ncols": 2},
            rfooter=SOURCE,
            lheader=PROVISIONAL_LHEADER,
            lfooter=LFOOTER + _data_to(frame.index),
        )


# --- table of contents, in run order
CHARTS = (
    (trend_by_state, ()),
    (original_by_state, ()),
    (growth_by_state, ()),
    (original_by_sector, ()),
    (growth_by_sector, ()),
    (sector_trend, ()),
    (sector_by_state, ()),
)
