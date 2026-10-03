"""Quarterly real GDP growth across the OECD: Australia in context, contractions, recessions, per capita."""

# --- dependencies
from dataclasses import dataclass

import pandas as pd
from mgplot import bar_plot, bar_plot_finalise, finalise_plot, line_plot

from au_econ.sources import oecd

# --- module contract
RELEASE = ("oecd-gdp",)
TOPICS = ("international",)
TITLE = "International GDP Growth"

# --- constants
AUSTRALIA = "Australia"
COUNTRIES = {  # country: OECD reference area; the 38 OECD members
    AUSTRALIA: "AUS",
    "Austria": "AUT",
    "Belgium": "BEL",
    "Canada": "CAN",
    "Chile": "CHL",
    "Colombia": "COL",
    "Costa Rica": "CRI",
    "Czech Republic": "CZE",
    "Denmark": "DNK",
    "Estonia": "EST",
    "Finland": "FIN",
    "France": "FRA",
    "Germany": "DEU",
    "Greece": "GRC",
    "Hungary": "HUN",
    "Iceland": "ISL",
    "Ireland": "IRL",
    "Israel": "ISR",
    "Italy": "ITA",
    "Japan": "JPN",
    "South Korea": "KOR",
    "Latvia": "LVA",
    "Lithuania": "LTU",
    "Luxembourg": "LUX",
    "Mexico": "MEX",
    "Netherlands": "NLD",
    "New Zealand": "NZL",
    "Norway": "NOR",
    "Poland": "POL",
    "Portugal": "PRT",
    "Slovakia": "SVK",
    "Slovenia": "SVN",
    "Spain": "ESP",
    "Sweden": "SWE",
    "Switzerland": "CHE",
    "Turkey": "TUR",
    "United Kingdom": "GBR",
    "United States": "USA",
}
# Dataflows are left unversioned, so the API serves the latest version.
# Keys have 13 dimensions: FREQ.ADJUSTMENT.REF_AREA.SECTOR.COUNTERPART_SECTOR.TRANSACTION.
# INSTR_ASSET.ACTIVITY.EXPENDITURE.UNIT_MEASURE.PRICE_BASE.TRANSFORMATION.TABLE_IDENTIFIER
GDP_FLOW = "OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA_EXPENDITURE_NATIO_CURR,"
# seasonally adjusted real GDP, national currency: chain-linked volumes (L), or constant
# prices (Q) where a country publishes no chain-linked series (Mexico)
GDP_KEY = "Q.Y.{areas}...B1GQ.....L+Q.."
PER_CAPITA_FLOW = "OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA_EXPENDITURE_CAPITA,"
PER_CAPITA_KEY = "Q.Y.{areas}........LR.."  # seasonally adjusted, chain-linked volumes
POPULATION_FLOW = "OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA_POP_EMPNC,"
POPULATION_KEY = "Q.Y.{areas}...POP......."  # seasonally adjusted total population
PER_CAPITA_EXCLUDED = {"Turkey": "OECD publishes its GDP per capita annually only"}
START = pd.Period("1999Q4", freq="Q")  # growth from 2000Q1
SOURCE = "OECD: QNA"
NATIONS = "OECD member countries."  # starts each GDP-growth chart's left footer
SEAS_ADJ = "Seas adj."  # ends it
PER_CAPITA_SOURCE = "OECD: QNA GDP per capita, population"
PER_CAPITA_LFOOTER = "Seas adj. * Population projected at its last-4-quarter average growth."
REPORT_COUNTRIES = 10  # countries listed in the data-recency report
REPORT_NAMES = 5  # names listed per line of the latest-period report

MEAN_MEDIAN = 0.80  # share of countries reporting before the mean and median are drawn
MIN_FOR_MEAN = 3  # countries needed before the mean and median are drawn
RECENT_FROM_YEAR = 2022
LAST_N_QUARTERS = 17
COUNT_START = pd.Period("2000Q1", freq="Q")
POST_COVID_START = pd.Period("2022Q1", freq="Q")
BASES = (POST_COVID_START - 1, pd.Period("2019Q4", freq="Q"))  # cumulative charts: post- and pre-COVID
TICK_LABEL_SIZE = "x-small"
VERTICAL = 90  # degrees: country names on the x-axis
PROJECTION_WINDOW = 4  # quarters of published population growth averaged for a projection
MAX_PROJECTION = 3  # quarters; a country further behind is dropped from the per-capita charts


@dataclass(frozen=True)
class GdpData:
    """Q/Q real GDP growth (per cent), and per-capita GDP and population levels from the base quarter."""

    growth: pd.DataFrame
    per_capita: pd.DataFrame
    population: pd.DataFrame


# --- data
def _oecd_table(dataflow: str, key: str, start: pd.Period, countries: dict[str, str]) -> pd.DataFrame:
    """Fetch one OECD measure from start, one column per country label; raise if a country is missing."""
    areas = "+".join(countries.values())
    rows = oecd.get_data(dataflow, key.format(areas=areas), start=str(start).replace("Q", "-Q"))
    duplicated = rows[rows.duplicated(["REF_AREA", "TIME_PERIOD"])]["REF_AREA"].unique()
    if len(duplicated):  # pivot_table would average them
        raise ValueError(f"OECD {dataflow}: more than one series for {sorted(duplicated)}")
    frame = rows.pivot_table(index="TIME_PERIOD", columns="REF_AREA", values="OBS_VALUE")
    frame.index = pd.PeriodIndex(frame.index, freq="Q")
    missing = set(countries.values()) - set(frame.columns)
    if missing:
        raise ValueError(f"OECD {dataflow}: no data for {sorted(missing)}")
    return frame.rename(columns={code: country for country, code in countries.items()})


def _report_missing(frame: pd.DataFrame) -> None:
    """Print coverage of the latest quarter and how recent each country's data is."""
    print("📊 DATA OVERVIEW")
    print(f"   Countries: {len(frame.columns)}")
    print(f"   Quarters: {len(frame)}")
    print(f"   Date range: {frame.index[0]} to {frame.index[-1]}")
    final_row = frame.iloc[-1]
    missing = frame.columns[final_row.isna()].tolist()
    if missing:
        available = frame.columns[final_row.notna()].tolist()
        more = "..." if len(missing) > REPORT_NAMES else ""
        print(f"\n📅 LATEST PERIOD: {final_row.name}")
        print(f"   Missing data: {len(missing)}/{len(frame.columns)} countries")
        print(f"   Countries missing data: {', '.join(missing[:REPORT_NAMES])}{more}")
        if available:
            more = "..." if len(available) > REPORT_NAMES else ""
            print(f"   Countries with data: {', '.join(available[:REPORT_NAMES])}{more}")
    else:
        print(f"\n✅ Complete data for latest period: {final_row.name}")
    print("\n📈 MOST RECENT DATA BY COUNTRY:")
    last_dates = frame.apply(lambda column: column.last_valid_index()).sort_values(ascending=False)
    for country in last_dates.head(REPORT_COUNTRIES).index:
        print(f"   {country:<20}: {last_dates[country]}")
    if len(last_dates) > REPORT_COUNTRIES:
        print(f"   ... and {len(last_dates) - REPORT_COUNTRIES} more countries")


def fetch() -> GdpData:
    """Return real GDP growth, one column per OECD member, and the per-capita levels and population."""
    levels = _oecd_table(GDP_FLOW, GDP_KEY, START, COUNTRIES)
    growth = (levels.pct_change(fill_method=None) * 100).dropna(how="all")
    _report_missing(growth)
    print("\nLatest quarterly growth rates:")
    print(growth.tail())
    for country, reason in PER_CAPITA_EXCLUDED.items():
        print(f"Per capita excludes {country}: {reason}")
    per_capita_countries = {k: v for k, v in COUNTRIES.items() if k not in PER_CAPITA_EXCLUDED}
    base = min(BASES)
    return GdpData(
        growth=growth,
        per_capita=_oecd_table(PER_CAPITA_FLOW, PER_CAPITA_KEY, base, per_capita_countries),
        population=_oecd_table(POPULATION_FLOW, POPULATION_KEY, base, per_capita_countries),
    )


# --- helpers
def _world_chart(data: pd.DataFrame, tag: str) -> None:
    """Australia's growth against every other country, with their mean and median."""
    others = [column for column in data.columns if column != AUSTRALIA]
    ax = line_plot(data[others], width=0.3, color="blue", alpha=0.5, label_series=False)
    if len(others) >= MIN_FOR_MEAN:
        enough = data[others].notna().sum(axis=1) >= len(others) * MEAN_MEDIAN
        mean = data[others].mean(axis=1).where(enough).rename("OECD mean")
        median = data[others].median(axis=1).where(enough).rename("OECD median")
        line_plot(mean, ax=ax, color="darkblue", style="--", width=2, label_series=True)
        line_plot(median, ax=ax, color="darkred", style=":", width=2, label_series=True)
    line_plot(data[AUSTRALIA].dropna(), ax=ax, color="darkorange", width=3, label_series=True)
    finalise_plot(
        ax,
        title="Australian quarterly GDP growth in world context",
        ylabel="Per cent per quarter",
        xlabel=None,
        y0=True,
        rfooter=SOURCE,
        lfooter=f"{NATIONS} Mean/median calculated when >{int(MEAN_MEDIAN * 100)}% report. {SEAS_ADJ}",
        tag=tag,
        legend={"loc": "best", "fontsize": "xx-small", "ncol": 3},
    )


def _latest(flags: pd.DataFrame, what: str) -> None:
    """Print the countries flagged in the latest quarter."""
    latest = flags.iloc[-1]
    print(f"Latest countries in {what} (N={int(latest.sum())}):")
    print(", ".join(latest[latest].index.tolist()))


def _contraction_counts(data: pd.DataFrame) -> pd.Series:
    """Count the countries with negative Q/Q growth, each quarter from COUNT_START."""
    counts = (data < 0).sum(axis=1)
    return counts[counts.index >= COUNT_START]


def _recession_counts(data: pd.DataFrame) -> pd.Series:
    """Count the countries in technical recession (two negative quarters running), each quarter."""
    counts = ((data < 0) & (data.shift(1) < 0)).sum(axis=1)
    return counts[counts.index >= COUNT_START]


def _post_covid(data: pd.DataFrame) -> pd.DataFrame:
    """Growth from POST_COVID_START, for countries with any data in that span."""
    return data.loc[data.index >= POST_COVID_START].dropna(how="all", axis=1)


def _cumulative_growth(data: pd.DataFrame, base: pd.Period) -> pd.Series:
    """Cumulative growth since base to each country's latest quarter, ascending.

    Each label carries the country's latest quarter (e.g. "Japan 25Q2").
    """
    since = data.loc[data.index > base].dropna(how="all", axis=1)
    growth = ((1 + since / 100).cumprod() - 1) * 100
    final = {}
    for country in growth.columns:
        last = growth[country].last_valid_index()
        if last is not None:
            final[f"{country} {str(last)[2:]}"] = growth[country].loc[last]
    return pd.Series(final).sort_values()


def _negative_quarters(data: pd.DataFrame) -> pd.Series:
    """Count each country's negative quarters since POST_COVID_START, ascending."""
    return (_post_covid(data) < 0).sum().sort_values()


def _cumulative_growth_per_capita(data: GdpData, base: pd.Period) -> pd.Series:
    """Cumulative real GDP per capita growth since base, ascending.

    Runs to each country's latest GDP quarter. Where that is past the per-capita series,
    per capita is extended by (1 + GDP growth) / (1 + population growth), with population
    growth projected at its average over the last PROJECTION_WINDOW published quarters
    where it is not yet published; those labels end in "*". A country whose population is
    more than MAX_PROJECTION quarters behind its GDP is dropped.
    """
    final: dict[str, float] = {}
    for country in data.per_capita.columns:
        levels = data.per_capita[country].dropna()
        if base not in levels.index:
            raise ValueError(f"OECD per capita: no {base} value for {country}")
        last, gdp_last = levels.index[-1], data.growth[country].last_valid_index()
        if not isinstance(last, pd.Period) or not isinstance(gdp_last, pd.Period):
            raise TypeError(f"{country}: expected quarterly periods")
        level, projected = levels.iloc[-1], False
        if gdp_last > last:
            population_growth = data.population[country].dropna().pct_change()
            average = population_growth.iloc[-PROJECTION_WINDOW:].mean()
            gap = gdp_last.ordinal - population_growth.index[-1].ordinal
            if gap > MAX_PROJECTION:
                print(f"{country}: population {gap} quarters behind GDP (cap {MAX_PROJECTION}); dropped")
                continue
            for quarter in pd.period_range(last + 1, gdp_last, freq="Q"):
                quarter_growth = population_growth.get(quarter, average)
                level *= (1 + data.growth.at[quarter, country] / 100) / (1 + quarter_growth)
            if gap > 0:
                projected = True
                print(f"{country}: population projected {gap} quarter(s) at {average:.2%} a quarter")
            last = gdp_last
        final[f"{country} {str(last)[2:]}{'*' if projected else ''}"] = (level / levels.loc[base] - 1) * 100
    return pd.Series(final).sort_values()


def _country_bars(values: pd.Series, title: str, ylabel: str, lfooter: str = "", rfooter: str = SOURCE) -> None:
    """Bar chart with one bar per country, names small and vertical."""
    ax = bar_plot(values, label_rotation=VERTICAL)
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_SIZE)
    finalise_plot(ax, title=title, ylabel=ylabel, rfooter=rfooter, lfooter=lfooter)


# --- charts
def world_context(data: GdpData) -> None:
    """Australia in world context: full history, since RECENT_FROM_YEAR, and the last LAST_N_QUARTERS."""
    growth = data.growth
    if not isinstance(growth.index, pd.PeriodIndex):
        raise TypeError("expected a quarterly PeriodIndex")
    _world_chart(growth, "full")
    _world_chart(growth[growth.index.year >= RECENT_FROM_YEAR], f"since-{RECENT_FROM_YEAR}")
    _world_chart(growth.iloc[-LAST_N_QUARTERS:], f"last-{LAST_N_QUARTERS}q")


def contractions(data: GdpData) -> None:
    """Chart the number of countries with a quarterly GDP contraction."""
    growth = data.growth
    bar_plot_finalise(
        _contraction_counts(growth),
        title="Number of OECD Countries with Quarterly GDP Contraction",
        ylabel="Count",
        rfooter=SOURCE,
        lfooter=f"{NATIONS} {SEAS_ADJ}",
    )
    _latest(growth < 0, "contraction")


def recessions(data: GdpData) -> None:
    """Chart the number of countries in technical recession."""
    growth = data.growth
    bar_plot_finalise(
        _recession_counts(growth),
        title="Number of OECD Countries in Technical Recession",
        ylabel="Count",
        rfooter=SOURCE,
        lfooter=f"{NATIONS} Technical recession = two consecutive quarters of negative GDP growth. {SEAS_ADJ}",
    )
    _latest((growth < 0) & (growth.shift(1) < 0), "technical recession")


def cumulative_growth(data: GdpData) -> None:
    """Cumulative GDP growth by country, since each of BASES."""
    for base in BASES:
        _country_bars(
            _cumulative_growth(data.growth, base),
            title=f"Cumulative GDP growth since {base}",
            ylabel="Per cent",
            lfooter=f"{NATIONS} To latest available data; see x-axis labels. {SEAS_ADJ}",
        )


def cumulative_growth_per_capita(data: GdpData) -> None:
    """Cumulative real GDP per capita growth by country, since each of BASES."""
    for base in BASES:
        _country_bars(
            _cumulative_growth_per_capita(data, base),
            title=f"Cumulative GDP per capita growth since {base}",
            ylabel="Per cent",
            lfooter=PER_CAPITA_LFOOTER,
            rfooter=PER_CAPITA_SOURCE,
        )


def negative_quarters(data: GdpData) -> None:
    """Chart each country's number of negative GDP quarters since 2022Q1."""
    _country_bars(
        _negative_quarters(data.growth),
        title=f"Number of negative GDP quarters since {POST_COVID_START}",
        ylabel="Count",
        lfooter=f"{NATIONS} {SEAS_ADJ}",
    )


# --- table of contents, in run order
CHARTS = (
    (world_context, ()),
    (contractions, ()),
    (recessions, ()),
    (cumulative_growth, ()),
    (cumulative_growth_per_capita, ()),
    (negative_quarters, ()),
)
