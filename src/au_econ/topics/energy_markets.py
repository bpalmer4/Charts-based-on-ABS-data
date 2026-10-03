"""Energy markets: crude oil, refined products and natural gas, around the 2026 Strait of Hormuz disruption.

Prices come from several publishers: front-month futures and dated NYMEX contracts from
Yahoo Finance; physical spot and retail prices from the US EIA; the OPEC Reference
Basket from OPEC; Middle East benchmarks from Oilprice.com; and full forward settlement
curves from CME Group, where Yahoo carries only the near months. Most charts start in
2026 and mark the disruption's key dates.
"""

# --- dependencies
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mgplot as mg
import pandas as pd

from au_econ.sources import cme, eia, oilprice, opec, yahoo

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# --- module contract
RELEASE = ("energy",)
TOPICS = ("commodities",)
TITLE = "Energy Markets"

# --- constants
SOURCE_YAHOO = "Source: Yahoo Finance"
SOURCE_EIA_OPEC = "Source: U.S. EIA, OPEC Secretariat"
SOURCE_OILPRICE = "Source: Oilprice.com"
SHORT_START = "2026-01-01"  # history start for the oil, gas and refined-product charts
RETAIL_START = "2025-01-01"  # a full flat 2025 baseline ahead of the disruption
LEGEND = {"loc": "best", "fontsize": "x-small"}
EVENTS = [  # shared event lines; "text" labels the line on the chart itself
    {
        "x": pd.Period("2026-02-28", freq="D"),
        "color": "grey",
        "linestyle": "--",
        "linewidth": 1,
        "text": "Hormuz closure",
    },
    {
        "x": pd.Period("2026-06-17", freq="D"),
        "color": "green",
        "linestyle": "-.",
        "linewidth": 1,
        "text": "Ceasefire MOU signed",
    },
    {
        "x": pd.Period("2026-07-08", freq="D"),
        "color": "darkred",
        "linestyle": ":",
        "linewidth": 1,
        "text": "Ceasefire declared over",
    },
]

# crude benchmarks
EIA_WTI, EIA_BRENT, OPEC_BASKET = "WTI (Cushing)", "Brent (Europe)", "OPEC Basket"
WTI, BRENT, OMAN, DUBAI = "WTI (US)", "Brent (Europe)", "Oman (Middle East)", "Dubai (Middle East)"
OILPRICE_BLENDS = {WTI: oilprice.WTI, BRENT: oilprice.BRENT, OMAN: oilprice.OMAN, DUBAI: oilprice.DUBAI}
YAHOO_WTI, YAHOO_BRENT = "CL=F", "BZ=F"

# refined products: Singapore swap futures (USD/bbl) and US Gulf jet (USD/gallon)
YAHOO_GASOIL, YAHOO_MOGAS = "SGB=F", "N1B=F"
US_GAL_PER_BBL = 42
SING_GASOIL, SING_MOGAS = "Sing. Gasoil 10ppm", "Sing. Mogas 92"
CME_SINGAPORE_PRODUCTS = {SING_GASOIL: 5033, SING_MOGAS: 4544}  # CME product ids, USD/bbl

# natural gas
YAHOO_GAS = {"NG=F": "Henry Hub (US)", "TTF=F": "TTF (Europe)", "JKM=F": "JKM (Asia)"}
HENRY_HUB, TTF, JKM = "Henry Hub (US)", "TTF (Europe)", "JKM (Asia)"
YAHOO_EURUSD = "EURUSD=X"
MMBTU_PER_MWH = 3.412  # TTF is quoted in EUR per MWh
CME_TTF_EUR_MWH = 8378  # Dutch TTF calendar-month futures (EUR/MWh)
CME_JKM = 7049  # LNG Japan/Korea Marker (Platts) futures (USD/MMBtu)

# forward curves
N_FORWARD_MONTHS = 18
FWD_CURVE_COLORS = ["darkblue", "darkorange", "cornflowerblue"]  # the third is for the three-series gas curve
NORMAL_2025_MEAN = {  # calendar-2025 mean front-month close, USD per barrel
    WTI: 64.74,  # CL=F
    BRENT: 68.10,  # BZ=F
    SING_GASOIL: 87.40,  # SGB=F
    SING_MOGAS: 78.58,  # N1B=F
}
NORMAL_2025_MEAN_GAS = {  # calendar-2025 mean front-month close, USD per MMBtu (TTF at daily EUR/USD)
    HENRY_HUB: 3.62,
    TTF: 11.92,
    JKM: 12.24,
}
CONTANGO_THRESHOLD_PCT_PER_MONTH = 0.1  # forward average must exceed the trough by this much per remaining month
REFERENCE_LINE_WIDTH = 0.9
Y_PADDING = 0.02  # share of the range added above and below, so reference lines stay in view
CURVE_MARKER_SIZE = 4

# location premium, retail petrol
SPREAD_COLORS = ["cornflowerblue", "purple"]
SPREAD_WIDTH = 1.5
ZERO_LINE = {"y": 0, "color": "#555555", "linestyle": "-", "linewidth": 1.5}  # heavier than mgplot's y0
RETAIL_COLOR = "darkblue"
RETAIL_WIDTH = 1.5
BASELINE_YEAR, EVENT_YEAR, TROUGH_MONTH, PEAK_MONTH = 2025, 2026, 7, 3
NORMAL_DIFFERENTIAL_BAND = {"ymin": -3, "ymax": 0, "color": "grey", "alpha": 0.2}
PERCENT = 100
MIN_CURVE_POINTS = 2


@dataclass(frozen=True)
class EnergyData:
    """Daily prices, and the latest forward curves with their trade dates."""

    eia_spot: dict[str, pd.Series]
    opec_basket: pd.Series
    oilprice: dict[str, pd.Series]
    yahoo: dict[str, pd.Series]
    crude_curves: dict[str, pd.DataFrame]
    cme_curves: dict[int, tuple[pd.Series, pd.Timestamp]]
    gulf_jet: pd.Series
    retail_petrol: pd.Series


# --- data
def fetch() -> EnergyData:
    """Fetch every price series and forward curve the charts use."""
    yahoo_tickers = [YAHOO_WTI, YAHOO_BRENT, YAHOO_GASOIL, YAHOO_MOGAS, *YAHOO_GAS, YAHOO_EURUSD]
    retail = eia.get_series(eia.RETAIL_PETROL, "US regular, all formulations")
    print(f"US retail petrol: {len(retail)} rows  ({retail.index[0]} to {retail.index[-1]})")
    return EnergyData(
        eia_spot={
            EIA_WTI: eia.get_series(eia.WTI_SPOT, EIA_WTI),
            EIA_BRENT: eia.get_series(eia.BRENT_SPOT, EIA_BRENT),
        },
        opec_basket=opec.get_basket(),
        oilprice={name: oilprice.get_blend(blend, name) for name, blend in OILPRICE_BLENDS.items()},
        yahoo={ticker: yahoo.get_close(ticker, SHORT_START) for ticker in yahoo_tickers},
        crude_curves={root: yahoo.get_forward_curve(root, N_FORWARD_MONTHS) for root in ("CL", "BZ", "NG")},
        cme_curves={
            product: cme.get_settlement_curve(product)
            for product in (*CME_SINGAPORE_PRODUCTS.values(), CME_TTF_EUR_MWH, CME_JKM)
        },
        gulf_jet=eia.get_series(eia.GULF_JET_SPOT, "US Gulf Jet", scale=US_GAL_PER_BBL),
        retail_petrol=retail.loc[retail.index >= pd.Period(RETAIL_START, freq="D")],
    )


# --- helpers
def _period(value: object) -> pd.Period:
    """Return value as a Period, or raise."""
    if not isinstance(value, pd.Period):
        raise TypeError(f"Expected a Period, got {type(value).__name__}")
    return value


def _last_day(frame: pd.DataFrame | pd.Series) -> str:
    """Return the last date with any data, e.g. "2-Oct-2026"."""
    return _period(frame.dropna(how="all").index[-1]).strftime("%-d-%b-%Y")


def _since_short_start[T: (pd.Series, pd.DataFrame)](data: T) -> T:
    """Keep data from SHORT_START."""
    return data.loc[data.index >= pd.Period(SHORT_START, freq="D")]


def _summarise(frame: pd.DataFrame, label: str) -> None:
    """Print the date range and min/max of each column."""
    valid = frame.dropna(how="all")
    print(f"{label}: {valid.index[0]} to {valid.index[-1]}")
    for column in frame.columns:
        series = frame[column].dropna()
        if len(series):
            print(f"  {column:22s}  n={len(series):3d}  min={series.min():.2f}  max={series.max():.2f}")


def _annotate_point(
    ax: Axes,
    series: pd.Series,
    period: pd.Period,
    *,
    color: str,
    text: str,
    dx: float = 0,
    dy: float = 12,
    ha: str = "center",
    arrow: bool = False,
) -> None:
    """Label one interior point of a period-indexed series.

    mgplot annotates series ends only, and has no function for an interior point, so this
    uses ax.annotate. The x coordinate is the Period ordinal, which is what mgplot maps a
    PeriodIndex onto. arrow adds a short leader for labels that sit clear of the line.
    """
    ax.annotate(
        text,
        xy=(period.ordinal, float(series[series.index == period].iloc[0])),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize="small",
        color=color,
        ha=ha,
        fontweight="bold",
        arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.8} if arrow else None,
    )


def _first_contango_flip(curve: pd.Series) -> pd.Period | None:
    """Return the first month of a sustained backwardation-to-contango flip, or None.

    On a curve that begins in backwardation: the first contract month where the next
    month settles higher and the average of all later months exceeds it by more than
    CONTANGO_THRESHOLD_PCT_PER_MONTH times the number of remaining months.
    """
    valid = curve.dropna()
    if len(valid) < MIN_CURVE_POINTS or valid.iloc[1] >= valid.iloc[0]:  # must start in backwardation
        return None
    return next(
        (
            _period(valid.index[j])
            for j in range(len(valid) - 1)
            if valid.iloc[j + 1] > valid.iloc[j]
            and valid.iloc[j + 1 :].mean() / valid.iloc[j] - 1
            > CONTANGO_THRESHOLD_PCT_PER_MONTH / PERCENT * (len(valid) - 1 - j)
        ),
        None,
    )


def _reference_lines(columns: pd.Index, means: dict[str, float]) -> list[dict[str, Any]]:
    """Return the 2025-mean reference line for each column that has one, in its colour, dashed and finer."""
    return [
        {
            "y": means[column],
            "color": FWD_CURVE_COLORS[i % len(FWD_CURVE_COLORS)],
            "linestyle": "--",
            "linewidth": REFERENCE_LINE_WIDTH,
            "label": f"{column}: 2025 mean",
        }
        for i, column in enumerate(columns)
        if column in means
    ]


def _padded_limits(frame: pd.DataFrame, means: dict[str, float]) -> tuple[float, float]:
    """Y limits covering the data and the reference means (axhline does not autoscale)."""
    references = [means[column] for column in frame.columns if column in means]
    low = min([frame.min().min(), *references])
    high = max([frame.max().max(), *references])
    pad = Y_PADDING * (high - low)
    return low - pad, high + pad


def _forward_curve_chart(frame: pd.DataFrame, as_of: pd.Timestamp, *, title: str, rfooter: str) -> None:
    """Chart a forward curve frame (each column one contract series), with 2025 means and contango flips."""
    # implied front-to-back change, averaged across the series
    changes = []
    for column in frame.columns:
        valid = frame[column].dropna()
        if len(valid) >= MIN_CURVE_POINTS:
            changes.append((valid.iloc[-1] / valid.iloc[0] - 1) * PERCENT)
    average = sum(changes) / len(changes) if changes else 0.0
    back = _period(frame.dropna(how="all").index[-1])
    flips = [
        {
            "x": flip,
            "color": FWD_CURVE_COLORS[i % len(FWD_CURVE_COLORS)],
            "linestyle": "-.",
            "linewidth": REFERENCE_LINE_WIDTH,
            "label": f"{column}: backwardation to contango",
        }
        for i, column in enumerate(frame.columns)
        if (flip := _first_contango_flip(frame[column])) is not None
    ]
    mg.line_plot_finalise(
        frame,
        title=title,
        ylabel="USD per barrel",
        xlabel="Contract month",
        color=FWD_CURVE_COLORS[: frame.shape[1]],
        legend=LEGEND,
        annotate=True,
        rounding=2,
        marker="o",
        markersize=CURVE_MARKER_SIZE,
        rheader=f"Market prices {average:+.0f}% by {back.strftime('%b %Y')}",
        axhline=_reference_lines(frame.columns, NORMAL_2025_MEAN),
        axvline=flips,
        ylim=_padded_limits(frame, NORMAL_2025_MEAN),
        lfooter=(
            "Latest settle price for each dated contract (delivery within month). "
            f"As of {as_of.strftime('%-d-%b-%Y')}."
        ),
        rfooter=rfooter,
    )


def _physical_spot(data: EnergyData) -> pd.DataFrame:
    """EIA WTI and Brent spot and the OPEC basket, from SHORT_START."""
    prices = pd.DataFrame({**data.eia_spot, OPEC_BASKET: data.opec_basket}).sort_index()
    return _since_short_start(prices)


def _oilprice_benchmarks(data: EnergyData) -> pd.DataFrame:
    """Return the four Oilprice.com benchmarks, from SHORT_START."""
    return _since_short_start(pd.DataFrame(data.oilprice).sort_index())


# --- charts
def physical_spot(data: EnergyData) -> None:
    """EIA daily spot for WTI and Brent, and the OPEC Reference Basket."""
    prices = _physical_spot(data)
    _summarise(prices, "Physical spot")
    eia_last = _period(prices[EIA_WTI].dropna().index[-1]).strftime("%-d-%b")
    orb_last = _period(prices[OPEC_BASKET].dropna().index[-1]).strftime("%-d-%b")
    mg.line_plot_finalise(
        prices,
        title="Crude Oil Physical Spot: WTI, Brent and OPEC Basket",
        ylabel="USD per barrel",
        xlabel=None,
        legend=LEGEND,
        annotate=True,
        rounding=2,
        axvline=EVENTS,
        lfooter=f"Daily physical spot. EIA WTI/Brent to {eia_last}; OPEC Basket to {orb_last}.",
        rfooter=SOURCE_EIA_OPEC,
    )


def location_premium(data: EnergyData) -> None:
    """Chart the OPEC basket's spread to WTI and to Brent.

    The basket normally trades below Brent; the Hormuz closure pushed it far above. That
    sign flip is the point of the chart, so the zero line is drawn heavier. The legs are
    assessed at different times of day, so dates missing any leg are dropped.
    """
    legs = _physical_spot(data)[[OPEC_BASKET, EIA_WTI, EIA_BRENT]].dropna()
    to_wti, to_brent = "OPEC Basket minus WTI", "OPEC Basket minus Brent"
    spreads = pd.DataFrame(
        {to_wti: legs[OPEC_BASKET] - legs[EIA_WTI], to_brent: legs[OPEC_BASKET] - legs[EIA_BRENT]}
    )
    for column in spreads.columns:
        series = spreads[column]
        print(
            f"{column:24s}  peak {series.max():6.2f} on {series.idxmax()}  "
            f"current {series.iloc[-1]:6.2f} on {series.index[-1]}"
        )
    below = spreads[to_brent] < 0
    print(
        f"Basket below Brent on {below.sum()} of {len(below)} days; last 10: "
        f"{[round(v, 2) for v in spreads[to_brent].tail(10)]}"
    )
    peak = _period(spreads[to_wti].idxmax())
    ax = mg.line_plot(spreads, color=SPREAD_COLORS, width=SPREAD_WIDTH, annotate=True, rounding=2, dropna=True)
    _annotate_point(
        ax,
        spreads[to_wti],
        peak,
        color=SPREAD_COLORS[0],
        dx=10,
        dy=-4,
        ha="left",
        text=f"peak {spreads[to_wti].max():.2f}, {peak.strftime('%-d %b')}",
    )
    mg.finalise_plot(
        ax,
        title="Crude Location Premium: Gulf Grades Against the Benchmarks",
        ylabel="USD per barrel, spread",
        xlabel=None,
        legend=LEGEND,
        axhline=ZERO_LINE,
        axvline=EVENTS,
        lfooter=(
            "Daily spreads. OPEC Reference Basket less WTI (Cushing) and less "
            f"Brent (Europe). Data to {_last_day(spreads)}."
        ),
        rfooter=SOURCE_EIA_OPEC,
    )


def crude_benchmarks(data: EnergyData) -> None:
    """Front-month crude benchmarks from Oilprice.com, including Oman and Dubai."""
    prices = _oilprice_benchmarks(data)
    _summarise(prices, "Crude futures (Oilprice.com)")
    last_dates = {
        column: _period(prices[column].dropna().index[-1]).strftime("%-d-%b") for column in prices.columns
    }
    dates = ", ".join(f"{column.split(' ')[0]} {date}" for column, date in last_dates.items())
    mg.line_plot_finalise(
        prices,
        title="Crude Oil Front-Month Futures: Global Benchmarks",
        ylabel="USD per barrel",
        xlabel=None,
        legend=LEGEND,
        annotate=True,
        rounding=2,
        axvline=EVENTS,
        lfooter=f"Daily front-month futures (Dubai is a Platts assessment). Last: {dates}.",
        rfooter=SOURCE_OILPRICE,
    )


def wti_brent_futures(data: EnergyData) -> None:
    """WTI and Brent front-month futures."""
    frame = pd.DataFrame({WTI: data.yahoo[YAHOO_WTI], BRENT: data.yahoo[YAHOO_BRENT]})
    _summarise(frame, "Crude futures")
    mg.line_plot_finalise(
        frame,
        title="Crude Oil Futures: WTI vs Brent",
        ylabel="USD per barrel",
        xlabel=None,
        legend=LEGEND,
        annotate=True,
        rounding=2,
        axvline=EVENTS,
        lfooter=f"Front-month futures. NYMEX (WTI) and ICE (Brent). Data to {_last_day(frame)}.",
        rfooter=SOURCE_YAHOO,
    )


def crude_forward_curves(data: EnergyData) -> None:
    """WTI and Brent forward curves from dated NYMEX contracts."""
    wti, brent = data.crude_curves["CL"], data.crude_curves["BZ"]
    for label, curve in (("WTI", wti), ("Brent", brent)):
        if len(curve) < N_FORWARD_MONTHS:
            print(f"{label}: {len(curve)} of {N_FORWARD_MONTHS} consecutive monthly contracts listed on Yahoo")
    prices = pd.DataFrame({WTI: wti["price"], BRENT: brent["price"]})
    prices.index = pd.PeriodIndex(prices.index, freq="M")
    as_of = max(wti["date"].max(), brent["date"].max())
    _forward_curve_chart(
        prices,
        as_of,
        title="Crude Oil Forward Curves: WTI and Brent",
        rfooter="Source: Yahoo Finance (NYMEX CL, ICE BZ)",
    )


def singapore_forward_curves(data: EnergyData) -> None:
    """Singapore gasoil and Mogas 92 forward curves, from CME settlements."""
    columns: dict[str, pd.Series] = {}
    dates: list[pd.Timestamp] = []
    for label, product in CME_SINGAPORE_PRODUCTS.items():
        series, trade_date = data.cme_curves[product]
        columns[label] = series
        dates.append(trade_date)
    curves = pd.DataFrame(columns).sort_index()
    _forward_curve_chart(
        curves,
        max(dates),
        title="Singapore Refined Product Forward Curves: Gasoil and Petrol",
        rfooter="Source: CME Group (NYMEX SGB, N1B settlements)",
    )


def middle_east_differentials(data: EnergyData) -> None:
    """Oman and Dubai less Brent: a widening premium signals constrained Gulf transit."""
    prices = _oilprice_benchmarks(data)
    differentials = pd.DataFrame(
        {"Oman minus Brent": prices[OMAN] - prices[BRENT], "Dubai minus Brent": prices[DUBAI] - prices[BRENT]}
    ).dropna(how="all")
    _summarise(differentials, "ME crude differentials")
    mg.line_plot_finalise(
        differentials,
        title="Middle East Crude Differentials vs Brent",
        ylabel="USD per barrel",
        xlabel=None,
        legend=LEGEND,
        annotate=True,
        rounding=2,
        axvline=EVENTS,
        axhspan=NORMAL_DIFFERENTIAL_BAND,
        y0=True,
        lfooter=(
            f"Grey band = normal range (Dubai/Oman trade ~$2 below Brent). Data to {_last_day(differentials)}."
        ),
        rfooter=SOURCE_OILPRICE,
    )


def singapore_products(data: EnergyData) -> None:
    """Singapore gasoil and Mogas 92 front-month futures."""
    frame = pd.DataFrame(
        {"Gasoil 10ppm (diesel)": data.yahoo[YAHOO_GASOIL], "Mogas 92 (petrol)": data.yahoo[YAHOO_MOGAS]}
    )
    _summarise(frame, "Singapore products")
    mg.line_plot_finalise(
        frame,
        title="Singapore Refined Product Futures: Gasoil and Petrol",
        ylabel="USD per barrel",
        xlabel=None,
        legend=LEGEND,
        annotate=True,
        rounding=2,
        axvline=EVENTS,
        rheader="Refined products (not crude)",
        lfooter=(
            "NYMEX front-month futures on Platts Singapore Gasoil (diesel) "
            f"and Mogas 92 (petrol). Data to {_last_day(frame)}."
        ),
        rfooter=SOURCE_YAHOO,
    )


def crack_spreads(data: EnergyData) -> None:
    """Refined product less Brent: Singapore gasoil and Mogas 92, and US Gulf jet."""
    brent = data.yahoo[YAHOO_BRENT]
    cracks = pd.DataFrame(
        {
            "Sing. Gasoil 10ppm minus Brent": data.yahoo[YAHOO_GASOIL] - brent,
            "Sing. Mogas 92 minus Brent": data.yahoo[YAHOO_MOGAS] - brent,
            "US Gulf Jet minus Brent": data.gulf_jet.loc[pd.Period(SHORT_START, "D") :] - brent,
        }
    ).dropna(how="all")
    _summarise(cracks, "Refined product crack-spreads")
    mg.line_plot_finalise(
        cracks,
        title="Refined Product Crack-Spreads vs Brent",
        ylabel="USD per barrel",
        xlabel=None,
        legend=LEGEND,
        annotate=True,
        rounding=2,
        axvline=EVENTS,
        y0=True,
        lfooter=f"Sing.: NYMEX futures; US Gulf: EIA spot. Data to {_last_day(cracks)}.",
        rfooter="Source: Yahoo Finance, U.S. EIA",
    )


def retail_petrol(data: EnergyData) -> None:
    """EIA weekly US average pump price for regular petrol, against its 2025 mean."""
    petrol = data.retail_petrol
    index = petrol.index
    if not isinstance(index, pd.PeriodIndex):
        raise TypeError("Expected a PeriodIndex")
    mean_2025 = float(petrol[index.year == BASELINE_YEAR].mean())
    july = petrol[(index.year == EVENT_YEAR) & (index.month == TROUGH_MONTH)]
    march = petrol[(index.year == EVENT_YEAR) & (index.month == PEAK_MONTH)]
    trough = _period(july.idxmin())
    print(f"2025 mean        {mean_2025:.3f}")
    print(
        f"Current          {petrol.iloc[-1]:.3f} on {petrol.index[-1]}  "
        f"(week-on-week {petrol.iloc[-1] - petrol.iloc[-2]:+.3f})"
    )
    print(f"July trough      {july.min():.3f} on {trough}")
    print(f"March 2026 peak  {march.max():.3f} on {march.idxmax()}")
    print(f"Series peak      {petrol.max():.3f} on {petrol.idxmax()}")

    ax = mg.line_plot(petrol, color=[RETAIL_COLOR], width=RETAIL_WIDTH, annotate=True, rounding=3)
    _annotate_point(
        ax,
        petrol,
        trough,
        color=RETAIL_COLOR,
        dx=-16,
        dy=-46,
        ha="right",
        arrow=True,
        text=f"July trough {july.min():.3f}, {trough.strftime('%-d %b')}",
    )
    mg.finalise_plot(
        ax,
        title="US Retail Petrol: Weekly National Average",
        ylabel="USD per gallon",
        xlabel=None,
        legend=LEGEND,
        axhline={
            "y": mean_2025,
            "color": RETAIL_COLOR,
            "linestyle": "--",
            "linewidth": REFERENCE_LINE_WIDTH,
            "label": "2025 mean",
        },
        axvline=EVENTS,
        lfooter=f"Weekly US average, all formulations, regular grade. Data to {_last_day(petrol)}.",
        rfooter="Source: U.S. EIA",
    )


def gas_benchmarks(data: EnergyData) -> None:
    """Henry Hub, TTF and JKM front-month futures, TTF converted to USD per MMBtu."""
    gas = pd.DataFrame({label: data.yahoo[ticker] for ticker, label in YAHOO_GAS.items()}).dropna()
    eurusd = data.yahoo[YAHOO_EURUSD].reindex(gas.index, method="ffill")
    gas[TTF] = gas[TTF] * eurusd / MMBTU_PER_MWH
    _summarise(gas, "Natural gas")
    mg.line_plot_finalise(
        gas,
        title="Natural Gas Futures Benchmarks",
        ylabel="USD per MMBtu",
        xlabel=None,
        legend=LEGEND,
        annotate=True,
        rounding=2,
        axvline=EVENTS,
        lfooter=(
            f"Front-month futures. TTF converted from EUR/MWh using daily EUR/USD rate. Data to {_last_day(gas)}."
        ),
        rfooter=SOURCE_YAHOO,
    )


def gas_forward_curves(data: EnergyData) -> None:
    """Henry Hub, TTF and JKM forward curves on a common USD per MMBtu axis.

    Henry Hub comes from dated NYMEX contracts; TTF and JKM from CME settlements, since
    Yahoo carries only their near months. TTF is converted at the latest spot EUR/USD, an
    approximation (the forward FX curve is not flat), flagged in the header. Gas curves
    are seasonal, so the crude curves' contango flag is omitted.
    """
    henry_hub = data.crude_curves["NG"].copy()
    henry_hub.index = pd.PeriodIndex(henry_hub.index, freq="M")
    ttf_eur, ttf_date = data.cme_curves[CME_TTF_EUR_MWH]
    jkm_usd, jkm_date = data.cme_curves[CME_JKM]
    fx = float(data.yahoo[YAHOO_EURUSD].dropna().iloc[-1])
    gas = pd.DataFrame(
        {
            HENRY_HUB: henry_hub["price"],
            TTF: (ttf_eur * fx / MMBTU_PER_MWH).reindex(henry_hub.index),
            JKM: jkm_usd.reindex(henry_hub.index),
        }
    )
    as_of = max(henry_hub["date"].max(), ttf_date, jkm_date)
    for column in gas.columns:
        valid = gas[column].dropna()
        print(
            f"  {column:16s}  n={len(valid):2d}  front={valid.iloc[0]:.2f}  "
            f"back={valid.iloc[-1]:.2f}  min={valid.min():.2f}  max={valid.max():.2f}"
        )
    hh = gas[HENRY_HUB].dropna()
    mg.line_plot_finalise(
        gas,
        title="Natural Gas Forward Curves: Henry Hub, TTF and JKM",
        ylabel="USD per MMBtu",
        xlabel="Contract month",
        color=FWD_CURVE_COLORS[: gas.shape[1]],
        legend=LEGEND,
        annotate=True,
        rounding=2,
        marker="o",
        markersize=CURVE_MARKER_SIZE,
        axhline=_reference_lines(gas.columns, NORMAL_2025_MEAN_GAS),
        ylim=_padded_limits(gas, NORMAL_2025_MEAN_GAS),
        lheader="TTF converted EUR/MWh to USD/MMBtu at spot EUR/USD",
        rheader=f"Henry Hub seasonal range {hh.min():.2f} to {hh.max():.2f} USD/MMBtu",
        lfooter=f"Dashed = 2025 mean. Latest settle per dated contract. As of {as_of.strftime('%-d-%b-%Y')}.",
        rfooter="Source: Yahoo Finance, CME Group",
    )


# --- table of contents, in run order
CHARTS = (
    (physical_spot, ()),
    (location_premium, ()),
    (crude_benchmarks, ()),
    (wti_brent_futures, ()),
    (crude_forward_curves, ()),
    (singapore_forward_curves, ()),
    (middle_east_differentials, ()),
    (singapore_products, ()),
    (crack_spreads, ()),
    (retail_petrol, ()),
    (gas_benchmarks, ()),
    (gas_forward_curves, ()),
)
