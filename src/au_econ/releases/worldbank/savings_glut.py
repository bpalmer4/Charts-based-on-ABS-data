"""Global savings glut: current account balances of major economies (World Bank World Development Indicators).

The "global savings glut" hypothesis (Bernanke 2005): excess saving over investment in key
economies drove capital to deficit countries, depressing global interest rates.
"""

# --- dependencies
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd
from mgplot import finalise_plot, line_plot, line_plot_finalise

from au_econ.sources import worldbank

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# --- module contract
RELEASE = ("wb-savings-glut",)
TOPICS = ("international",)
TITLE = "Global Savings Glut"

# --- constants
COUNTRIES = {  # label: ISO3 code; G20 plus G10 members and Spain
    "United States": "USA",
    "United Kingdom": "GBR",
    "Germany": "DEU",
    "France": "FRA",
    "Italy": "ITA",
    "Japan": "JPN",
    "Canada": "CAN",
    "Netherlands": "NLD",
    "Belgium": "BEL",
    "Sweden": "SWE",
    "Switzerland": "CHE",
    "Australia": "AUS",
    "China": "CHN",
    "India": "IND",
    "Korea": "KOR",
    "Indonesia": "IDN",
    "Saudi Arabia": "SAU",
    "Brazil": "BRA",
    "Mexico": "MEX",
    "Argentina": "ARG",
    "South Africa": "ZAF",
    "Russia": "RUS",
    "Türkiye": "TUR",
    "Spain": "ESP",
}
WORLD = {"World": "WLD"}
INDICATORS = {  # label: World Bank indicator ID
    "current account, % of GDP": "BN.CAB.XOKA.GD.ZS",
    "current account, current USD": "BN.CAB.XOKA.CD",
    "GDP, current USD": "NY.GDP.MKTP.CD",
}
START = 1980  # the series run to the latest year published
SOURCE = "World Bank: WDI"
THRESHOLD = 1.5  # per cent of GDP: above is surplus, below its negative is deficit, else near balance
CLASSIFICATION_YEARS = 20  # recent years averaged to classify a nation
NAMES_PER_ROW = 6  # nations per line of the chart text box
SURPLUSES, DEFICITS = "Sum of surpluses", "Sum of deficits"
TEXT_BOX = {"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8, "edgecolor": "lightgray"}
TEXT_X = 0.02  # axes fraction
TEXT_FONT_SIZE = 6
GROUP_LEGEND = {"loc": "best", "fontsize": "x-small", "ncol": 2}


@dataclass(frozen=True)
class GlutData:
    """Current account balances (per cent of GDP; current USD) by nation, and world GDP (current USD)."""

    balance_share: pd.DataFrame
    balance_usd: pd.DataFrame
    world_gdp: pd.Series


# --- data
def fetch() -> GlutData:
    """Fetch the three indicators once; every chart function receives them."""
    codes, end = COUNTRIES.values(), pd.Timestamp.today().year
    share = worldbank.get_indicator(INDICATORS["current account, % of GDP"], codes, START, end)
    print(f"Nations ({len(share.columns)}): {list(share.columns)}")
    print(share.tail())
    usd = worldbank.get_indicator(INDICATORS["current account, current USD"], codes, START, end)
    world = worldbank.get_indicator(INDICATORS["GDP, current USD"], WORLD.values(), START, end)
    print(f"CA USD nations: {len(usd.columns)}")
    print(f"World GDP available: {world.index.min()} to {world.index.max()}")
    print(usd.tail())
    return GlutData(balance_share=share, balance_usd=usd, world_gdp=world.iloc[:, 0])


# --- helpers
def _world_shares(data: GlutData) -> pd.DataFrame:
    """Sum of positive and of negative current accounts each year, as a share of world GDP."""
    share = data.balance_usd.div(data.world_gdp, axis=0) * 100
    frame = pd.DataFrame(index=share.index)
    frame[SURPLUSES] = share.clip(lower=0).sum(axis=1)
    frame[DEFICITS] = share.clip(upper=0).sum(axis=1)
    return frame.dropna(how="all")


def _nation_box(ax: Axes, data: GlutData, *, y: float, va: str) -> None:
    """List the contributing nations in a small box on the chart."""
    names = sorted(data.balance_usd.columns)
    text = "\n".join(", ".join(names[i : i + NAMES_PER_ROW]) for i in range(0, len(names), NAMES_PER_ROW))
    ax.text(TEXT_X, y, text, transform=ax.transAxes, fontsize=TEXT_FONT_SIZE, ha="left", va=va, bbox=TEXT_BOX)


# --- charts
def current_account_groups(data: GlutData) -> None:
    """Chart current account balances for surplus, near-balance and deficit nations (by recent average)."""
    averages = data.balance_share.iloc[-CLASSIFICATION_YEARS:].mean().sort_values(ascending=False)
    groups = {
        "surplus": averages[averages > THRESHOLD].index.tolist(),
        "near-balance": averages[(averages >= -THRESHOLD) & (averages <= THRESHOLD)].index.tolist(),
        "deficit": averages[averages < -THRESHOLD].index.tolist(),
    }
    print(f"Surplus (>{THRESHOLD}% avg):  {groups['surplus']}")
    print(f"Near-balance:         {groups['near-balance']}")
    print(f"Deficit (<-{THRESHOLD}% avg): {groups['deficit']}")
    print(averages.to_frame(f"{CLASSIFICATION_YEARS}yr avg CA (% GDP)").round(1))
    for group, nations in groups.items():
        line_plot_finalise(
            data.balance_share[nations].dropna(how="all"),
            title=f"Current account {group} nations",
            ylabel="Per cent of GDP",
            y0=True,
            rfooter=SOURCE,
            lfooter="Current account balance as percentage of GDP. Positive = net capital exporter.",
            legend=GROUP_LEGEND,
        )


def surpluses_and_deficits(data: GlutData) -> None:
    """Sum of surpluses against sum of deficits, as a share of world GDP."""
    ax = line_plot(_world_shares(data))
    _nation_box(ax, data, y=0.02, va="bottom")
    finalise_plot(
        ax,
        title="Current account surpluses and deficits\nas share of world GDP",
        ylabel="Per cent of world GDP",
        y0=True,
        rfooter=SOURCE,
        lfooter="Each year: sum of all positive (negative) current account balances (USD) divided by world GDP.",
        legend={"loc": "best", "fontsize": "small"},
    )


def net_balance(data: GlutData) -> None:
    """Net current account balance (surpluses plus deficits), as a share of world GDP."""
    shares = _world_shares(data)
    ax = line_plot((shares[SURPLUSES] + shares[DEFICITS]).to_frame("Net current account"))
    _nation_box(ax, data, y=0.98, va="top")
    finalise_plot(
        ax,
        title="Net current account balance as share of world GDP",
        ylabel="Per cent of world GDP",
        y0=True,
        rfooter=SOURCE,
        lfooter="Sum of all current account balances (USD) divided by world GDP.",
        legend=False,
    )


# --- table of contents, in run order
CHARTS = (
    (current_account_groups, ()),
    (surpluses_and_deficits, ()),
    (net_balance, ()),
)
