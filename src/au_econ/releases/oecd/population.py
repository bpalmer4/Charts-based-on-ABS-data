"""Population growth across OECD members and partners: Australia in world context, annualised growth."""

# --- dependencies
import pandas as pd
from mgplot import bar_plot_finalise, finalise_plot

from au_econ.charting.international import AUSTRALIA, MEAN_MEDIAN, world_context_axes
from au_econ.sources import oecd

# --- module contract
RELEASE = ("oecd-pop",)
TOPICS = ("international",)
TITLE = "Population Growth"

# --- constants
DATAFLOW = "OECD.ELS.SAE,DSD_POPULATION@DF_POP_HIST,"  # unversioned: the latest version
KEY = "..PS._T._T."  # total population, persons, both sexes, all ages
START = "2000"
SOURCE = "OECD: Population"
START_YEARS = (2000, 2010, 2022)  # each chart's base year (in its title, so its file name)
INDEX_BASE = 100
NAME_FONT_SIZE = 6
NAME_OFFSET = (5, 0)  # points right of a line's end
VERTICAL = 90  # degrees: country names on the x-axis


# --- data
def fetch() -> pd.DataFrame:
    """Return annual total population, one column per country label."""
    table = oecd.get_table(DATAFLOW, KEY, START)
    table.index = pd.PeriodIndex(table.index, freq="Y")
    population = oecd.national_only(table)
    oecd.report_missing(population)
    print(population.tail())
    return population.rename(columns=oecd.LABELS)


def _indexed(data: pd.DataFrame, start_year: int) -> pd.DataFrame:
    """Index population to start_year = INDEX_BASE, for countries with data."""
    base = pd.Period(str(start_year), freq="Y")
    base_row = data[data.index == base].iloc[0]
    return (data[data.index >= base].div(base_row) * INDEX_BASE).dropna(how="all", axis=1)


# --- charts
def population_world(data: pd.DataFrame) -> None:
    """Australia's population growth against every country, naming those that grew faster."""
    for start_year in START_YEARS:
        indexed = _indexed(data, start_year)
        ax = world_context_axes(indexed, annotate=True)
        australia_end = indexed[AUSTRALIA].dropna().iloc[-1]
        for country in indexed.columns:
            series = indexed[country].dropna()
            if country == AUSTRALIA or series.empty or series.iloc[-1] <= australia_end:
                continue
            last = series.index[-1]
            if not isinstance(last, pd.Period):
                raise TypeError(f"{country}: expected an annual PeriodIndex")
            ax.annotate(
                country,
                xy=(last.ordinal, series.iloc[-1]),
                fontsize=NAME_FONT_SIZE,
                color="black",
                ha="left",
                va="center",
                xytext=NAME_OFFSET,
                textcoords="offset points",
            )
        finalise_plot(
            ax,
            title=f"Australian population growth in the world context since {start_year}",
            ylabel=f"Index ({start_year} = {INDEX_BASE})",
            xlabel=None,
            y0=True,
            rfooter=SOURCE,
            lfooter=(
                "OECD monitored nations. Mean and median calculated where "
                f"{int(MEAN_MEDIAN * 100)}% or more nations report."
            ),
            legend={"loc": "best", "fontsize": "xx-small"},
        )


def population_growth(data: pd.DataFrame) -> None:
    """Compound annual population growth since each start year, by country."""
    for start_year in START_YEARS:
        indexed = _indexed(data, start_year)
        annualised: dict[str, float] = {}
        for country in indexed.columns:
            series = indexed[country].dropna()
            if series.empty:
                continue
            last = series.index[-1]
            if not isinstance(last, pd.Period):
                raise TypeError(f"{country}: expected an annual PeriodIndex")
            years = last.year - start_year
            if years > 0:
                annualised[country] = ((series.iloc[-1] / INDEX_BASE) ** (1 / years) - 1) * 100
        bar_plot_finalise(
            pd.Series(annualised).sort_values(),
            title=f"Annualised population growth since {start_year}",
            ylabel="Per cent per year",
            label_rotation=VERTICAL,
            rfooter=SOURCE,
            lfooter="OECD monitored nations. Compound annual growth rate.",
        )


# --- table of contents, in run order
CHARTS = (
    (population_world, ()),
    (population_growth, ()),
)
