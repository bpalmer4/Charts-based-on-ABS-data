"""RBA interest rates: the cash rate, money-market rates, OCR futures, housing loan payments and lending rates.

Tables: A2 (the cash rate target), F1.1 (the interbank overnight cash rate, monthly, from
1976), F17 (zero-coupon forward rates, daily, from 2017), F1 (daily money-market rates),
E13 (housing loan payments) and F5/F6 (lending rates).
"""

# --- dependencies
import textwrap
from dataclasses import dataclass

import mgplot as mg
import pandas as pd
import readabs as ra
import seaborn as sns

from au_econ.charting.inflation_backplane import (
    BACKPLANE_LHEADER,
    BACKPLANE_MONTHLY_LFOOTER,
    inflation_backplane,
)
from au_econ.series.rates import get_cash_rate
from au_econ.sources import rba

# --- module contract
RELEASE = ("rba-rates",)
TOPICS = ("rba",)
TITLE = "Interest Rates"

# --- constants
SOURCE = "Source: RBA"
MONTHS_PER_YEAR, QUARTERS_PER_YEAR, RECENT_YEARS = 12, 4, 5
plot_times = 0, -(RECENT_YEARS * MONTHS_PER_YEAR + 1)  # full history, and the most recent five years
CASH_RATE_NAME = "RBA Official Cash Rate"
INFLATION_TARGETING_FROM = pd.Period("1993-01-01", freq="M")
cash_rate_starts = INFLATION_TARGETING_FROM, plot_times[1]
CYCLES_FROM = "1994-01-01"  # the start of the RBA's inflation-targeting approach
CYCLES = (  # run_plot direction, title word, highlight label(s)
    ("up", "Tightening", "Tightening monetary policy"),
    ("down", "Easing", "Easing monetary policy"),
    ("both", "Both", ("Tightening monetary policy", "Easing monetary policy")),
)
BACKPLANE_FROM = pd.Period("1993-01", freq="M")
LINE_WIDTH, THIN_WIDTH = 2, 1.5

# F1.1, F1
OVERNIGHT_TITLE = "Interbank Overnight Cash Rate"
SHORT_RATES_1M = ["Cash Rate Target", "EOD 1-month BABs/NCDs"]
SHORT_RATES_3_6M = ["Cash Rate Target", "EOD 3-month BABs/NCDs", "EOD 6-month BABs/NCDs"]
short_rate_starts = pd.Period("2022-01-01", freq="D"), pd.Period("2025-01-01", freq="D")
BABS_KEY = "Key: EOD = end of day; BABs/NCDs = Bank Accepted Bills / Negotiable Certificates of Deposit. "

# F17: zero-coupon forward rates
FORWARD_YEARS = 1.5
DAYS_PER_YEAR = 365
EN_DASH = chr(0x2013)  # the F17 titles end with an en dash before the horizon, e.g. "... - 0.5 yrs"
ZC_PALETTE = "cool"
ZC_MONTHLY_WIDTH, ZC_CASH_RATE_WIDTH, ZC_DAILY_WIDTH, ZC_DAILY_ALPHA = 1, 2.5, 0.2, 0.5
ZC_LEGEND = {"loc": "upper left", "fontsize": "x-small"}

# E13: housing loan payments (some monthly, some quarterly)
e13_starts = 0, -(RECENT_YEARS * QUARTERS_PER_YEAR + 1)
SPLIT_TITLE_LENGTH = 50  # longer E13 titles break at their last semicolon

# F5 and F6: lending rates, by RBA series ID
LENDING_RATES = {
    "F5": {
        "Credit cards, standard": "FILRPLRCCS",
        "Banks' discounted variable housing rate, owner-occupier": "FILRHLBVD",
        "Banks' 3-year fixed housing rate, owner-occupier": "FILRHL3YF",
    },
    "F6": {
        "New owner-occupier loans, all": "FLRHOFTA",
        "New owner-occupier loans, variable": "FLRHOFVA",
        "New owner-occupier loans, fixed up to 3 years": "FLRHOFFA",
    },
}
LENDING_TITLE_WIDTH = 60


@dataclass(frozen=True)
class RatesData:
    """The monthly cash rate target, and each RBA table used, as (data, metadata)."""

    cash_rate: pd.Series
    tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]]


# --- data
def fetch() -> RatesData:
    """Fetch the cash rate and the F1.1, F17, E13, F1, F5 and F6 tables."""
    tables = {table: rba.get_table(table) for table in ("F1.1", "F17", "E13", "F1", "F5", "F6")}
    return RatesData(cash_rate=get_cash_rate().rename(CASH_RATE_NAME), tables=tables)


# --- helpers
def _by_title(data: RatesData, table: str, titles: list[str]) -> pd.DataFrame:
    """Columns of one table selected by series title, renamed to their titles."""
    frame, meta = data.tables[table]
    series_ids = [meta[meta.Title == title].index[0] for title in titles]
    return (
        frame[series_ids]
        .rename(dict(zip(series_ids, titles, strict=True)), axis=1)
        .dropna(how="all", axis=1)
        .dropna(how="all", axis=0)
        .infer_objects()
    )


def _zero_coupon(data: RatesData) -> pd.DataFrame:
    """F17 forward rates out to FORWARD_YEARS: one column per observation day, indexed by the forward date."""
    frame, meta = data.tables["F17"]
    horizons = (
        meta.Title.str.split(f" {EN_DASH} ").str[-1].str.replace(" yrs", "").str.replace(" yr", "").astype(float)
    )
    wanted = horizons[horizons <= FORWARD_YEARS].index
    horizons = horizons.loc[wanted]
    rates = frame.loc[:, wanted]
    rates.columns = horizons
    curves = {}
    for date, curve in rates.iterrows():
        if not isinstance(date, pd.Period | pd.Timestamp):
            raise TypeError(f"RBA F17: unexpected date {date!r}")
        day = pd.Period(date, freq="D")
        curve.index = pd.to_timedelta((curve.index * DAYS_PER_YEAR).astype(int), unit="D") + day
        curves[day] = curve
    return pd.DataFrame(curves)


def _last_day(index: pd.Index) -> object:
    """Return the last entry of an index (for "Data to" footers)."""
    return index[-1]


# --- charts
def cash_rate(data: RatesData) -> None:
    """Chart the cash rate since 1993 and recently, then its tightening and easing cycles since 1994."""
    rate = data.cash_rate
    mg.multi_start(
        rate,
        function=mg.line_plot_finalise,
        starts=cash_rate_starts,
        title=CASH_RATE_NAME,
        drawstyle="steps-post",
        ylabel="Per cent",
        zero_y=True,
        width=LINE_WIDTH,
        rfooter=f"{SOURCE} A2",
        lfooter=f"Australia. Monthly data to {_last_day(rate.index)}. ",
        pre_tag="a2-",
        annotate=True,
    )
    since_94 = rate[rate.index >= CYCLES_FROM]
    for direction, word, labels in CYCLES:
        mg.run_plot_finalise(
            since_94,
            width=LINE_WIDTH,
            direction=direction,
            title=f"{CASH_RATE_NAME} - {word} Cycles",
            ylabel="Per cent",
            rfooter=f"{SOURCE} A2",
            lfooter=f"Australia. Monthly data to {_last_day(since_94.index)}. ",
            highlight_label=labels,
            pre_tag="a2-",
            annotate=True,
            label_series=True,
            legend={"loc": "center left", "fontsize": "x-small"},
        )


def cash_rate_inflation_regime(data: RatesData) -> None:
    """Chart the cash rate over the inflation backplane, since 1993 and recently."""
    rate = data.cash_rate.dropna().astype(float)
    rate = rate[rate.index >= BACKPLANE_FROM]
    if rate.empty:
        raise ValueError(f"No official cash rate data from {BACKPLANE_FROM}")
    for start in plot_times:
        window = rate.iloc[start:]
        if not isinstance(window.index, pd.PeriodIndex):
            raise TypeError(f"Expected a PeriodIndex, got {type(window.index)}")
        ax = mg.line_plot(window, drawstyle="steps-post", annotate=True)
        ax = inflation_backplane(window.index, ax)
        mg.finalise_plot(
            ax,
            title="RBA Official Cash Rate: Against the Inflation Regime",
            ylabel="Per cent",
            lheader=BACKPLANE_LHEADER,
            legend={"loc": "best", "fontsize": "x-small", "ncol": 2},
            lfooter="Australia. Monthly. " + BACKPLANE_MONTHLY_LFOOTER,
            rfooter=f"{SOURCE} A2; ABS 6401.0",
            pre_tag="a2-",
            tag=f"start{start}",
        )


def long_run_policy_rate(data: RatesData) -> None:
    """Chart the interbank overnight cash rate (F1.1), the effective policy rate back to 1976."""
    frame, meta = data.tables["F1.1"]
    series_id = meta[meta.Title == OVERNIGHT_TITLE].index[0]
    policy = frame[series_id].dropna().astype(float).rename(OVERNIGHT_TITLE)
    mg.line_plot_finalise(
        policy,
        title="Interbank Overnight Cash Rate: RBA Long-run Policy Rate Proxy",
        drawstyle="steps-post",
        ylabel="Per cent",
        zero_y=True,
        width=THIN_WIDTH,
        rfooter=f"{SOURCE} F1.1",
        lfooter=f"Australia. Effective overnight cash rate. Monthly data to {_last_day(policy.index)}. ",
        pre_tag="f1-1-",
        annotate=True,
    )


def zero_coupon_minmax(data: RatesData) -> None:
    """Chart the highest and lowest forward rate over the next 18 months, each day."""
    curves = _zero_coupon(data)
    extremes = pd.DataFrame({"Max": curves.max(), "Min": curves.min()})
    mg.line_plot_finalise(
        extremes,
        color=["darkorange", "cornflowerblue"],
        width=THIN_WIDTH,
        title="Zero-coupon Forward Rates - Min/Max over forward 18m",
        ylabel="Rate (%/year)",
        rfooter=f"{SOURCE} F17",
        lfooter=f"Australia. Daily data. Data to {_last_day(curves.columns)}. ",
        legend=ZC_LEGEND,
        pre_tag="f17-",
    )


def zero_coupon_monthly(data: RatesData) -> None:
    """End-of-month forward curves, coloured from oldest to newest, with the cash rate."""
    by_day = _zero_coupon(data).T.to_timestamp()
    index = by_day.index
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("Expected a DatetimeIndex")
    curves = (
        by_day.groupby(by=[index.year, index.month])
        .last(skipna=False)  # the last day of each month
        .dropna(how="all", axis=1)
        .T
    )
    colors = list(sns.color_palette(ZC_PALETTE, len(curves.columns)).as_hex())
    curves.columns = pd.Index([f"_{column}" for column in curves.columns])  # unlabelled in the legend
    ocr = data.cash_rate.asfreq("D")
    ocr = ocr[ocr.index >= curves.index.min()].rename(CASH_RATE_NAME)
    frame = pd.concat([curves, ocr], axis=1).sort_index()
    n_curves = len(curves.columns)
    ax = mg.line_plot(
        frame,
        color=[*colors, "r"],
        width=[ZC_MONTHLY_WIDTH] * n_curves + [ZC_CASH_RATE_WIDTH],
        drawstyle=["default"] * n_curves + ["steps-post"],
        style="-",
        label_series=[False] * n_curves + [True],
    )
    mg.finalise_plot(
        ax,
        title="EOM Zero-coupon Forward Rates (over forward 18 months)",
        ylabel="Rate (%/year)",
        rfooter=f"{SOURCE} A2 F17",
        lfooter=f"Australia. EOM=End of month. Data to {index[-1].date()}. ",
        legend=ZC_LEGEND,
        pre_tag="f17-",
    )


def zero_coupon_daily(data: RatesData) -> None:
    """Every day's forward curve, coloured from oldest to newest."""
    curves = _zero_coupon(data)
    colors = list(sns.color_palette(ZC_PALETTE, len(curves.columns)).as_hex())
    ax = mg.line_plot(curves, color=colors, width=ZC_DAILY_WIDTH, alpha=ZC_DAILY_ALPHA, style="-")
    mg.finalise_plot(
        ax,
        title="Zero-coupon Forward Rates (over forward 18 months)",
        ylabel="Rate (%/year)",
        rfooter=f"{SOURCE} F17",
        lfooter=f"Australia. Daily data. Data to {_last_day(curves.columns)}. ",
        legend=False,
        pre_tag="f17-",
    )


def housing_repayments(data: RatesData) -> None:
    """Every E13 housing loan payment series, over the full history and recently."""
    frame, meta = data.tables["E13"]
    print(f"Last date: {frame.index[-1]}")
    for _, row in meta.iterrows():
        title, unit = str(row["Title"]), str(row["Units"])
        series = frame[row["Series ID"]].astype(float).dropna()
        series, unit = ra.recalibrate(series, unit)
        if len(title) > SPLIT_TITLE_LENGTH:
            title = "\n".join(title.rsplit(";", 1))
        mg.multi_start(
            series,
            starts=e13_starts,
            function=mg.line_plot_finalise,
            pre_tag="e13-",
            title=title,
            ylabel=unit,
            rfooter=f"{SOURCE} E13",
            lfooter=(f"Australia. {row['Type']}. Data to {series.index[-1]}: {series.iloc[-1]:.03f} {unit}. "),
            width=LINE_WIDTH,
            annotate=True,
        )


def _short_rates(data: RatesData, titles: list[str], *, title: str, annotate: list[bool]) -> None:
    """Chart F1 money-market rates from each of the short-rate start dates."""
    frame = _by_title(data, "F1", titles)
    mg.multi_start(
        frame,
        function=mg.line_plot_finalise,
        starts=short_rate_starts,
        title=title,
        drawstyle="steps-post",
        ylabel="Per cent",
        rfooter=f"{SOURCE} F1 Daily",
        lfooter=BABS_KEY + f"Data to {frame.index[-1]}.",
        width=LINE_WIDTH,
        pre_tag="f1-",
        annotate=annotate,
    )


def short_term_rates_1m(data: RatesData) -> None:
    """Chart the cash rate target against the 1-month bank bill rate."""
    print(f"Last date: {data.tables['F1'][0].index[-1]}")
    _short_rates(
        data, SHORT_RATES_1M, title="Australia: short-term Interest rates (1 month)", annotate=[False, True]
    )


def short_term_rates_3_6m(data: RatesData) -> None:
    """Chart the cash rate target against the 3- and 6-month bank bill rates."""
    _short_rates(
        data,
        SHORT_RATES_3_6M,
        title="Australia: short-term Interest rates (3 and 6 month)",
        annotate=[False, True, True],
    )


def _lending_charts(data: RatesData, table: str, wanted: dict[str, str], pre_tag: str) -> None:
    """Chart each wanted lending-rate series of one table separately."""
    frame, meta = data.tables[table]
    if not meta["Series ID"].is_unique:
        raise ValueError("Series IDs not unique")
    for series_id in wanted.values():
        series = frame[meta[meta["Series ID"] == series_id].index[0]].dropna().astype(float)
        series, unit = ra.recalibrate(series, str(meta.loc[series_id, "Units"]))
        series.name = series_id
        mg.line_plot_finalise(
            series,
            title=textwrap.fill(str(meta.loc[series_id, "Title"]), LENDING_TITLE_WIDTH),
            ylabel=unit,
            rfooter=f"{SOURCE} {table}",
            lfooter=f"Australia. Data to {series.index[-1]}. ",
            width=LINE_WIDTH,
            pre_tag=pre_tag,
            annotate=True,
        )


def lending_rates(data: RatesData) -> None:
    """Chart selected F5 and F6 lending rates, one chart each."""
    for table, wanted in LENDING_RATES.items():
        _lending_charts(data, table, wanted, f"{table.lower()}-")


# --- table of contents, in run order
CHARTS = (
    (cash_rate, ()),
    (cash_rate_inflation_regime, ()),
    (long_run_policy_rate, ()),
    (zero_coupon_minmax, ()),
    (zero_coupon_monthly, ()),
    (zero_coupon_daily, ()),
    (housing_repayments, ()),
    (short_term_rates_1m, ()),
    (short_term_rates_3_6m, ()),
    (lending_rates, ()),
)
