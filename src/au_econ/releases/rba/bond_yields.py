"""RBA Australian Government bond yields: the yield curve, curve inversions, the term spread and long-run yields.

Tables: F2 (daily capital market yields, from 2013) and its history Z:F2-Daily-2013;
F2.1 (monthly) and Z:F2-Monthly-2013; F1 and Z:F1-Daily-2010 (the interbank overnight
cash rate). Nominal GDP comes from ABS 5206.0. The current F2 table holds a stray text
entry (a lone space in the 3-year yield on 31 May 2013), so its yields are converted to
numbers before use.
"""

# --- dependencies
from dataclasses import dataclass

import mgplot as mg
import pandas as pd
import readabs as ra
from readabs import metacol as mc

from au_econ.series.gdp import get_gdp
from au_econ.sources import rba
from au_econ.sources.abs import fetch_release

# --- module contract
RELEASE = ("rba-bonds",)
TOPICS = ("rba",)
TITLE = "Australian Bond Yields"

# --- constants
SOURCE = "Source: RBA"
YIELDS_TITLE = "Capital Market Yields - Australian Government Bonds"
RECENT_DAYS = -100  # the yield chart's window; the full-history version shared its file name and was overwritten
inversion_starts = 0, -150
INVERSION_PAIRS = ((2, 3), (2, 5), (2, 10))  # (shorter, longer) bond tenors in years
BOND_TITLE_WORDS = "Australian|Commonwealth"
BOND_TITLE_PREFIXES = ("Australian Government ", "Commonwealth Government ")
YIELD_WIDTH = 1.5

# RBA series IDs
RBA_SERIES = {
    "10-year bond yield, daily": "FCMYGBAG10D",
    "10-year bond yield, monthly": "FCMYGBAG10",
    "Interbank overnight cash rate, daily": "FIRMMCRID",
}
TEN_YEAR_DAILY = RBA_SERIES["10-year bond yield, daily"]
TEN_YEAR_MONTHLY = RBA_SERIES["10-year bond yield, monthly"]
OVERNIGHT_DAILY = RBA_SERIES["Interbank overnight cash rate, daily"]

# nominal GDP growth against the bond yield
CAGR_YEARS = 10
QUARTERS_PER_YEAR = 4
PERCENT = 100
RBA_BOND_TITLE = "Australian Government 10 year bond"
GDP_CATALOGUE = "5206.0"
GDP_TABLE = "5206001_Key_Aggregates"
GDP_PER_CAPITA_DID = "GDP per capita: Current prices ;"
TENDER_START = "1982Q3"  # first Treasury bond tender (Aug-1982); before it, yields were set on tap


@dataclass(frozen=True)
class BondData:
    """Each RBA table used, as (data, metadata); quarterly nominal GDP and GDP per capita (original)."""

    tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]]
    gdp: pd.Series
    gdp_per_capita: pd.Series


# --- data
def _gdp_per_capita_original() -> pd.Series:
    """Nominal GDP per capita (Original, from 1959Q3); the SA series starts only in 1973Q3."""
    release = fetch_release(GDP_CATALOGUE, single_excel_only=GDP_TABLE, verbose=False)
    table, series_id, _ = ra.find_abs_id(
        release.meta, {GDP_PER_CAPITA_DID: mc.did, "Original": mc.stype}, verbose=False
    )
    series = release.data[table][series_id].dropna()
    if series.empty:
        raise ValueError(f"No data for {GDP_PER_CAPITA_DID!r}")
    return series


def fetch() -> BondData:
    """Fetch the bond and overnight-rate tables, nominal GDP and nominal GDP per capita."""
    names = ("F2", "Z:F2-Daily-2013", "F2.1", "Z:F2-Monthly-2013", "F1", "Z:F1-Daily-2010")
    gdp, _ = get_gdp("CP", "SA")
    return BondData(
        tables={name: rba.get_table(name) for name in names},
        gdp=gdp,
        gdp_per_capita=_gdp_per_capita_original(),
    )


# --- helpers
def _daily_yields(data: BondData) -> pd.DataFrame:
    """F2 government bond yields, one column per tenor (e.g. "10 year bond"), as numbers."""
    frame, meta = data.tables["F2"]
    bonds = meta[meta.Title.str.contains(BOND_TITLE_WORDS) & meta.Title.str.contains("year")]
    labels = bonds.Title
    for prefix in BOND_TITLE_PREFIXES:
        labels = labels.str.replace(prefix, "")
    yields = frame[labels.index].apply(pd.to_numeric, errors="coerce")
    yields.columns = pd.Index(labels)
    return yields


def _series(data: BondData, table: str, series_id: str) -> pd.Series:
    """One series from an RBA table, as numbers."""
    frame, _ = data.tables[table]
    return pd.to_numeric(frame[series_id], errors="coerce").dropna().astype(float).rename(series_id)


def _spliced(data: BondData, *sources: tuple[str, str]) -> pd.Series:
    """Splice RBA table segments, highest priority (most recent) first."""
    series, _ = ra.splice([_series(data, table, series_id) for table, series_id in sources], rebase=False)
    return series


def _cagr(level: pd.Series) -> pd.Series:
    """Compound annual growth (%) over the trailing CAGR_YEARS of a quarterly level."""
    return ((level / level.shift(CAGR_YEARS * QUARTERS_PER_YEAR)) ** (1 / CAGR_YEARS) - 1) * PERCENT


def _monthly_ten_year(data: BondData) -> pd.Series:
    """Monthly 10-year yield: F2.1 extended back by its pre-2013 history, the current table winning on overlap."""
    current, meta = data.tables["F2.1"]
    match = meta[meta["Title"] == RBA_BOND_TITLE]
    if len(match) != 1:
        raise ValueError(f"RBA F2.1 holds {len(match)} series titled {RBA_BOND_TITLE!r}")
    series_id = str(match["Series ID"].iloc[0])
    history, _ = data.tables["Z:F2-Monthly-2013"]
    if series_id not in history.columns:
        raise ValueError(f"RBA Z:F2-Monthly-2013 has no {series_id} column")
    joined = pd.concat(
        [pd.to_numeric(part, errors="coerce").dropna() for part in (history[series_id], current[series_id])]
    )
    joined = joined[~joined.index.duplicated(keep="last")].sort_index()
    if joined.empty:
        raise ValueError("RBA returned no 10-year bond yields")
    return joined


# --- charts
def yield_curve(data: BondData) -> None:
    """Chart daily government bond yields at each tenor, over the last 100 trading days."""
    yields = _daily_yields(data)
    print(f"Last date: {data.tables['F2'][0].index[-1]}")
    mg.line_plot_finalise(
        yields,
        plot_from=RECENT_DAYS,
        tag="F2-Daily",
        width=YIELD_WIDTH,
        drawstyle="steps-post",
        title=YIELDS_TITLE,
        ylabel="Per cent per annum",
        rfooter=f"{SOURCE} F2 Daily",
        lfooter=f"Australian Government Bonds. Data up to {yields.index[-1]}. ",
        pre_tag="f2-",
        annotate=True,
    )


def yield_inversions(data: BondData) -> None:
    """Chart how far each longer yield sat below the 2-year yield (zero when the curve was not inverted)."""
    yields = _daily_yields(data)
    last = data.tables["F2"][0].index[-1]
    for short, long in INVERSION_PAIRS:
        gap = yields[f"{long} year bond"] - yields[f"{short} year bond"]
        inversion = -gap.where(gap < 0, other=0)
        mg.multi_start(
            inversion,
            function=mg.line_plot_finalise,
            starts=inversion_starts,
            title=f"Capital Market Yield Inversions [({long}-year - {short}-year) * -1]",
            ylabel="% points difference",
            rfooter=f"{SOURCE} F2 Daily",
            lfooter=f"Australian Government Bonds. Data up to {last}. ",
            pre_tag="f2-",
        )


def long_run_ten_year(data: BondData) -> None:
    """Chart the daily 10-year yield since 1995, joining the pre-2013 history to the current table."""
    history, _ = data.tables["Z:F2-Daily-2013"]
    current, _ = data.tables["F2"]
    combined = pd.concat([history, current], axis=0)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()  # history is blank on its overlap
    ten_year = pd.to_numeric(combined[TEN_YEAR_DAILY], errors="coerce")
    mg.line_plot_finalise(
        ten_year,
        width=1,
        annotate=True,
        rounding=2,
        title="10 Year Australian Government Bond Yields",
        ylabel="Per cent per annum",
        rfooter=f"{SOURCE} F2 Daily",
        lfooter=f"Australian Government Bonds. Data up to {combined.index[-1]}. ",
    )


def term_spread(data: BondData) -> None:
    """Chart the 10-year yield less the overnight cash rate: monthly to Dec-1994, daily thereafter.

    The cash rate is averaged over each month in the monthly era: before 1990 the interbank
    overnight rate was the unofficial cash market, and month-end values diverge from the
    month mean by 1.2 points on average (28 points at worst in the 1980s).
    """
    ten_daily = _spliced(data, ("F2", TEN_YEAR_DAILY), ("Z:F2-Daily-2013", TEN_YEAR_DAILY))
    cash_daily = _spliced(data, ("F1", OVERNIGHT_DAILY), ("Z:F1-Daily-2010", OVERNIGHT_DAILY))
    ten_monthly = _series(data, "Z:F2-Monthly-2013", TEN_YEAR_MONTHLY)
    cash_index = cash_daily.index
    if not isinstance(cash_index, pd.PeriodIndex):
        raise TypeError("Expected a daily PeriodIndex")
    cash_monthly = cash_daily.groupby(cash_index.asfreq("M")).mean()
    daily = (ten_daily - cash_daily.reindex(ten_daily.index, method="ffill")).dropna()
    monthly = (ten_monthly - cash_monthly).dropna()
    spread, _ = ra.splice([daily, monthly], rebase=False, name="Term spread")
    mg.line_plot_finalise(
        spread,
        width=1,
        annotate=True,
        rounding=2,
        y0=True,
        title="Term Spread: 10 Year Bond less Overnight Cash Rate",
        ylabel="Percentage points",
        rfooter=f"{SOURCE} F1, F2",
        lfooter="Australia. Monthly to Dec-1994, daily thereafter. ",
        pre_tag="f1-f2-",
    )


def gdp_growth_vs_bond_yield(data: BondData) -> None:
    """Chart rolling 10-year nominal GDP growth (total and per capita) against the 10-year bond yield."""
    monthly = _monthly_ten_year(data)
    monthly_index = monthly.index
    if not isinstance(monthly_index, pd.PeriodIndex):
        raise TypeError("Expected a monthly PeriodIndex")
    bond = monthly.groupby(monthly_index.asfreq("Q")).mean()
    frame = pd.DataFrame(
        {
            f"Nominal GDP: rolling {CAGR_YEARS}-year CAGR": _cagr(data.gdp),
            f"Nominal GDP per capita: rolling {CAGR_YEARS}-year CAGR": _cagr(data.gdp_per_capita),
            "10-year Australian Government bond yield": bond,
        }
    ).dropna()
    frame = frame[frame.index >= pd.Period(TENDER_START, freq="Q")]
    mg.line_plot_finalise(
        frame,
        title="Nominal GDP growth vs the 10-year bond yield",
        ylabel="Per cent per year",
        legend=True,
        rfooter="Source: ABS 5206.0, RBA F2.1",
        lfooter=(
            "Australia. GDP current prices, seasonally adjusted; per capita original. "
            "Since bond tenders began in Aug-1982. "
        ),
        y0=True,
    )


# --- table of contents, in run order
CHARTS = (
    (yield_curve, ()),
    (yield_inversions, ()),
    (long_run_ten_year, ()),
    (term_spread, ()),
    (gdp_growth_vs_bond_yield, ()),
)
