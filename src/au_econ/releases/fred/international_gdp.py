"""Quarterly real GDP growth across OECD countries, from FRED: Australia in context, contractions, recessions."""

# --- dependencies
from dataclasses import dataclass

import pandas as pd
from mgplot import bar_plot, bar_plot_finalise, finalise_plot, line_plot

from au_econ.sources import oecd
from au_econ.sources.fred import get_series
from au_econ.sources.http_cache import HttpError

# --- module contract
RELEASE = ("fred-gdp",)
TOPICS = ("international",)
TITLE = "International GDP Growth"

# --- constants
START = "2000-01-01"
SOURCE = "FRED"
NATIONS = "FRED monitored nations."  # starts every FRED chart's left footer
SEAS_ADJ = "Seas adj."  # ends it: every country charted is seasonally adjusted
MIN_OBSERVATIONS = 10
AUSTRALIA = "Australia"
COUNTRIES = {  # country: FRED series ID; real GDP levels, seasonally adjusted unless noted
    # Eurostat pattern (CLVMNAC)
    "France": "CLVMNACSCAB1GQFR",
    "Germany": "CLVMNACSCAB1GQDE",
    "Austria": "CLVMNACSCAB1GQAT",
    "Belgium": "CLVMNACSCAB1GQBE",
    "Czech Republic": "CLVMNACSCAB1GQCZ",
    "Denmark": "CLVMNACSCAB1GQDK",
    "Estonia": "CLVMNACSCAB1GQEE",
    "Finland": "CLVMNACSCAB1GQFI",
    "Hungary": "CLVMNACSCAB1GQHU",
    "Italy": "CLVMNACSCAB1GQIT",
    "Latvia": "CLVMNACSCAB1GQLV",
    "Lithuania": "CLVMNACSCAB1GQLT",
    "Luxembourg": "CLVMNACSCAB1GQLU",
    "Netherlands": "CLVMNACSCAB1GQNL",
    "Norway": "CLVMNACSCAB1GQNO",
    "Poland": "CLVMNACSCAB1GQPL",
    "Portugal": "CLVMNACSCAB1GQPT",
    "Slovenia": "CLVMNACSCAB1GQSI",
    "Spain": "CLVMNACSCAB1GQES",
    "Sweden": "CLVMNACSCAB1GQSE",
    "Switzerland": "CLVMNACSCAB1GQCH",
    "Greece": "CLVMNACSCAB1GQEL",
    # NGDPRSAXDC pattern
    "United States": "GDPC1",
    "United Kingdom": "NGDPRSAXDCGBQ",
    AUSTRALIA: "NGDPRSAXDCAUQ",
    "Canada": "NGDPRSAXDCCAQ",
    "South Korea": "NGDPRSAXDCKRQ",
    "Mexico": "NGDPRSAXDCMXQ",
    # other patterns
    "Japan": "JPNRGDPEXP",
    "New Zealand": "NAEXKP01NZQ657S",
}
GROWTH_RATE_COUNTRIES = {"New Zealand"}  # FRED series is already Q/Q growth, not a level
# Excluded for now: FRED has their real GDP only seasonally unadjusted, so their Q/Q
# growth is mostly seasonal swing (country: FRED series ID, kept for when they return)
EXCLUDED = {"Iceland": "CLVMNACNSAB1GQIS", "Ireland": "CLVMNACNSAB1GQIE"}
NOT_IN_FRED = ("Slovakia", "Chile", "Turkey", "Israel", "Colombia", "Costa Rica")
ID_HINTS = (
    "FRED ID patterns to try: European countries CLVMNACSCAB1GQ + 2-letter country code; "
    "others NGDPRSAXDC + 2-letter code + Q, or [COUNTRY]RGDPNASAQ; search https://fred.stlouisfed.org/"
)
REPORT_COUNTRIES = 10  # countries listed in the data-recency report
REPORT_NAMES = 5  # names listed per line of the latest-period report

MEAN_MEDIAN = 0.80  # share of countries reporting before the mean and median are drawn
MIN_FOR_MEAN = 3  # countries needed before the mean and median are drawn
RECENT_FROM_YEAR = 2022
LAST_N_QUARTERS = 17
COUNT_START = pd.Period("2000Q1", freq="Q")
POST_COVID_START = pd.Period("2022Q1", freq="Q")
TICK_LABEL_SIZE = "x-small"
VERTICAL = 90  # degrees: country names on the x-axis

# GDP per capita: OECD quarterly national accounts, seasonally adjusted, same countries
OECD_CODES = {  # country: OECD reference area
    "France": "FRA",
    "Germany": "DEU",
    "Austria": "AUT",
    "Belgium": "BEL",
    "Czech Republic": "CZE",
    "Denmark": "DNK",
    "Estonia": "EST",
    "Finland": "FIN",
    "Hungary": "HUN",
    "Italy": "ITA",
    "Latvia": "LVA",
    "Lithuania": "LTU",
    "Luxembourg": "LUX",
    "Netherlands": "NLD",
    "Norway": "NOR",
    "Poland": "POL",
    "Portugal": "PRT",
    "Slovenia": "SVN",
    "Spain": "ESP",
    "Sweden": "SWE",
    "Switzerland": "CHE",
    "Greece": "GRC",
    "United States": "USA",
    "United Kingdom": "GBR",
    AUSTRALIA: "AUS",
    "Canada": "CAN",
    "South Korea": "KOR",
    "Mexico": "MEX",
    "Japan": "JPN",
    "New Zealand": "NZL",
}
PER_CAPITA_FLOW = "OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA_EXPENDITURE_CAPITA,"
POPULATION_FLOW = "OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA_POP_EMPNC,"
# Keys have 13 dimensions: FREQ.ADJUSTMENT.REF_AREA.SECTOR.COUNTERPART_SECTOR.TRANSACTION.
# INSTR_ASSET.ACTIVITY.EXPENDITURE.UNIT_MEASURE.PRICE_BASE.TRANSFORMATION.TABLE_IDENTIFIER
PER_CAPITA_KEY = "Q.Y.{areas}........LR.."  # seasonally adjusted, chain-linked volumes
POPULATION_KEY = "Q.Y.{areas}...POP......."  # seasonally adjusted total population
PROJECTION_WINDOW = 4  # quarters of published population growth averaged for a projection
MAX_PROJECTION = 3  # quarters; a country further behind is dropped from the per-capita chart
PER_CAPITA_SOURCE = "OECD: QNA GDP per capita, population; FRED"
PER_CAPITA_LFOOTER = "Seas adj. * Population projected at its last-4-quarter average growth."


# --- data
def _growth(country: str, series_id: str) -> pd.Series:
    """Fetch one country's quarterly real GDP and return Q/Q growth, per cent."""
    try:
        levels = get_series(series_id, START, frequency="q")
    except (HttpError, ValueError) as exc:
        raise ValueError(f"{country} ({series_id}): {exc}. {ID_HINTS}") from exc
    if len(levels) < MIN_OBSERVATIONS:
        raise ValueError(f"{country} ({series_id}): only {len(levels)} quarters. {ID_HINTS}")
    levels.index = pd.PeriodIndex(levels.index, freq="Q")
    growth = levels if country in GROWTH_RATE_COUNTRIES else levels.pct_change(periods=1) * 100
    return growth.rename(country)


def _report_coverage() -> None:
    """Print how many OECD countries FRED covers, and which it does not."""
    print(f"OECD Country Coverage: {len(COUNTRIES)} countries with FRED series IDs")
    print(f"Note: {len(NOT_IN_FRED)} OECD countries not found in FRED: {', '.join(NOT_IN_FRED)}")
    print(f"Excluded, FRED real GDP not seasonally adjusted: {', '.join(EXCLUDED)}")


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


@dataclass(frozen=True)
class GdpData:
    """FRED Q/Q real GDP growth (per cent), and OECD per-capita GDP and population levels from the base quarter."""

    growth: pd.DataFrame
    per_capita: pd.DataFrame
    population: pd.DataFrame


def _oecd_levels(dataflow: str, key: str) -> pd.DataFrame:
    """Fetch one OECD measure from the quarter before POST_COVID_START, one column per country label."""
    areas = "+".join(OECD_CODES.values())
    rows = oecd.get_data(dataflow, key.format(areas=areas), start=str(POST_COVID_START - 1).replace("Q", "-Q"))
    frame = rows.pivot_table(index="TIME_PERIOD", columns="REF_AREA", values="OBS_VALUE")
    frame.index = pd.PeriodIndex(frame.index, freq="Q")
    missing = set(OECD_CODES.values()) - set(frame.columns)
    if missing:
        raise ValueError(f"OECD {dataflow}: no data for {sorted(missing)}")
    return frame.rename(columns={code: country for country, code in OECD_CODES.items()})


def fetch() -> GdpData:
    """Return FRED GDP growth, one column per country, and the OECD per-capita levels and population."""
    _report_coverage()
    frame = pd.DataFrame({country: _growth(country, series_id) for country, series_id in COUNTRIES.items()})
    frame = frame.dropna(how="all")
    if frame.empty:
        raise ValueError(f"FRED GDP: no data. {ID_HINTS}")
    _report_missing(frame)
    print("\nLatest quarterly growth rates:")
    print(frame.tail())
    return GdpData(
        growth=frame,
        per_capita=_oecd_levels(PER_CAPITA_FLOW, PER_CAPITA_KEY),
        population=_oecd_levels(POPULATION_FLOW, POPULATION_KEY),
    )


# --- helpers
def _world_chart(data: pd.DataFrame, tag: str) -> None:
    """Australia's growth against every other country, with their mean and median."""
    others = [column for column in data.columns if column != AUSTRALIA]
    ax = line_plot(data[others], width=0.3, color="blue", alpha=0.5, label_series=False)
    if len(others) >= MIN_FOR_MEAN:
        enough = data[others].notna().sum(axis=1) >= len(others) * MEAN_MEDIAN
        mean = data[others].mean(axis=1).where(enough).rename("FRED monitored mean")
        median = data[others].median(axis=1).where(enough).rename("FRED monitored median")
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


def _cumulative_growth(data: pd.DataFrame) -> pd.Series:
    """Cumulative growth since POST_COVID_START - 1 to each country's latest quarter, ascending.

    Each label carries the country's latest quarter (e.g. "Japan 25Q2").
    """
    growth = ((1 + _post_covid(data) / 100).cumprod() - 1) * 100
    final = {}
    for country in growth.columns:
        last = growth[country].last_valid_index()
        if last is not None:
            final[f"{country} {str(last)[2:]}"] = growth[country].loc[last]
    return pd.Series(final).sort_values()


def _negative_quarters(data: pd.DataFrame) -> pd.Series:
    """Count each country's negative quarters since POST_COVID_START, ascending."""
    return (_post_covid(data) < 0).sum().sort_values()


def _cumulative_growth_per_capita(data: GdpData) -> pd.Series:
    """Cumulative real GDP per capita growth since POST_COVID_START - 1, ascending.

    Runs to each country's latest FRED GDP quarter. Where that is past the OECD per-capita
    series, per capita is extended by (1 + GDP growth) / (1 + population growth), with
    population growth projected at its average over the last PROJECTION_WINDOW published
    quarters where it is not yet published; those labels end in "*". A country whose
    population is more than MAX_PROJECTION quarters behind its GDP is dropped.
    """
    base = POST_COVID_START - 1
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
        title="Number of FRED Monitored Countries with Quarterly GDP Contraction",
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
        title="Number of FRED Monitored Countries in Technical Recession",
        ylabel="Count",
        rfooter=SOURCE,
        lfooter=f"{NATIONS} Technical recession = two consecutive quarters of negative GDP growth. {SEAS_ADJ}",
    )
    _latest((growth < 0) & (growth.shift(1) < 0), "technical recession")


def cumulative_growth(data: GdpData) -> None:
    """Cumulative GDP growth since the end of 2021, by country."""
    _country_bars(
        _cumulative_growth(data.growth),
        title=f"Cumulative GDP growth since {POST_COVID_START - 1}",
        ylabel="Per cent",
        lfooter=f"{NATIONS} To latest available data; see x-axis labels. {SEAS_ADJ}",
    )


def cumulative_growth_per_capita(data: GdpData) -> None:
    """Cumulative real GDP per capita growth since the end of 2021, by country."""
    _country_bars(
        _cumulative_growth_per_capita(data),
        title=f"Cumulative GDP per capita growth since {POST_COVID_START - 1}",
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
