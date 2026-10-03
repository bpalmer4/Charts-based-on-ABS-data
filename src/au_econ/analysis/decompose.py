"""A simple seasonal decomposition of a monthly or quarterly series: trend, seasonal and irregular.

Based on ABS (2005), "An Introductory Course on Times Series Analysis -- Electronic
Delivery", Catalogue 1346.0.55.001. It does not adjust for moving holidays, public
holidays or trading days, so it is a naive decomposition.
"""

import warnings
from operator import sub, truediv
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from statsmodels.tsa.stattools import acf, kpss

from au_econ.analysis.henderson import hma

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# --- decomposition process steps: the returned frame's columns
ORIGINAL: Final[str] = "Original"
EXTENDED: Final[str] = "ARIMA Extended"
PERIOD: Final[str] = "Period"

FIRST_TREND: Final[str] = "1st Trend Estimate"
SECOND_TREND: Final[str] = "2nd Trend Estimate"

FIRST_SEAS: Final[str] = "1st Seasonal Weights Estimate"
SECOND_SEAS: Final[str] = "2nd Seasonal Weights Estimate"
THIRD_SEAS: Final[str] = "3rd Seasonal Weights Estimate"

FIRST_SEASADJ: Final[str] = "1st Seasonally Adjusted Estimate"

FINAL_SEASONAL: Final[str] = "Seasonal Weights"
FINAL_SEASADJ: Final[str] = "Seasonally Adjusted"
FINAL_TREND: Final[str] = "Trend"
FINAL_IRREGULAR: Final[str] = "Irregular"

QUARTERS, MONTHS = 4, 12
PERIODS_PER_YEAR = {"Q": QUARTERS, "M": MONTHS}
HENDERSON_TERMS = {MONTHS: 13, QUARTERS: 9}  # the ABS uses 7 terms for quarterly; 9 here
SEASONAL_SMOOTHER_YEARS = 20

_HENDERSON = "Henderson"
_MAX_ARIMA_ORDER = 2
_KPSS_ALPHA = 0.05
_MIN_KPSS_LENGTH = 12
_SEASONAL_ACF_THRESHOLD = 0.64
_SEASONAL_SMOOTHER_SLACK = 3  # years beyond the smoother's window needed to use it
_MIN_SEASON_LENGTH = 2  # a season needs at least two periods
_MAX_DIFFERENCES_FOR_CONSTANT = 2  # no constant term with this many differences or more


# --- public decomposition function
def decompose(
    s: pd.Series,
    model: str = "multiplicative",
    *,
    arima_extend: bool = False,
    constant_seasonal: bool = False,
    len_seasonal_smoother: int = SEASONAL_SMOOTHER_YEARS,
    discontinuity_list: Sequence[pd.Period] = (),
    ignore_years: Sequence[int] = (),
) -> pd.DataFrame:
    """Decompose a series into trend, seasonal and irregular components.

    Multiplicative (the default): Original = Trend x Seasonal x Irregular.
    Additive: Original = Trend + Seasonal + Irregular.

    Args:
        s: the series, sorted, with no missing values, on a monthly or quarterly PeriodIndex.
        model: "multiplicative" or "additive".
        arima_extend: extend the series both ways by auto-ARIMA before decomposing, so the
            endpoint Henderson trend gets symmetric weights; the padding is clipped back.
        constant_seasonal: hold the seasonal component constant, rather than slowly varying.
        len_seasonal_smoother: years in the slowly varying seasonal smoother.
        discontinuity_list: the last periods before breaks in the series (e.g. the pandemic).
        ignore_years: years left out of a constant seasonal estimate (e.g. 2020 and 2021).

    Returns:
        A frame with a column for each step; the results are "Seasonally Adjusted",
        "Trend", "Seasonal Weights" and "Irregular", beside "Original" and "ARIMA Extended".

    """
    n_periods = _check_input_validity(s, discontinuity_list)
    h = HENDERSON_TERMS[n_periods]
    oper = truediv if model == "multiplicative" else sub
    result = _extend_series_by_arima(s, n_periods, h, arima_extend=arima_extend)

    # - intermediate decomposition
    result[FIRST_TREND] = _get_trend(result[EXTENDED], h, (), "Other")  # a simpler first-pass smoother
    result[FIRST_SEAS] = oper(result[EXTENDED], result[FIRST_TREND])
    result[SECOND_SEAS] = _smooth_seasonal(
        result[FIRST_SEAS], n_periods, ignore_years, constant=constant_seasonal, years=len_seasonal_smoother
    )
    result[FIRST_SEASADJ] = oper(result[EXTENDED], result[SECOND_SEAS])
    result[SECOND_TREND] = _get_trend(result[FIRST_SEASADJ], h, discontinuity_list)
    result[THIRD_SEAS] = oper(result[EXTENDED], result[SECOND_TREND])

    # - final decomposition results
    result[FINAL_SEASONAL] = _smooth_seasonal(
        result[THIRD_SEAS], n_periods, ignore_years, constant=constant_seasonal, years=len_seasonal_smoother
    )
    result[FINAL_SEASADJ] = oper(result[ORIGINAL], result[FINAL_SEASONAL])
    # The final trend is taken on the extended seasonally adjusted series, so its endpoints
    # get symmetric Henderson weights when arima_extend is on (extended == original otherwise).
    result[FINAL_TREND] = _get_trend(oper(result[EXTENDED], result[FINAL_SEASONAL]), h, discontinuity_list)
    result[FINAL_IRREGULAR] = oper(result[FINAL_SEASADJ], result[FINAL_TREND])

    # Clip back to the original dates: the projected padding rows are dropped.
    return result.loc[s.index]


# --- private helpers
def _period_index(index: pd.Index) -> pd.PeriodIndex:
    """Return the index as a PeriodIndex, or raise."""
    if not isinstance(index, pd.PeriodIndex):
        raise TypeError("Expected a PeriodIndex")
    return index


def _get_trend(
    s: pd.Series,
    h: int,
    discontinuity_list: Sequence[pd.Period],
    methodology: str = _HENDERSON,
) -> pd.Series:
    """Trend a series by Henderson moving average, piece by piece between discontinuities.

    Any other methodology is a simple weighted smoother, which ignores discontinuities.
    """
    weights = np.array([1] + [2] * max(1, h - 2) + [1])
    weights = weights / np.sum(weights)  # weights sum to one
    breaks = [*discontinuity_list, s.index[-1]] if methodology == _HENDERSON else [s.index[-1]]
    remainder = s.dropna().copy()
    trend = pd.Series()
    for d in breaks:
        core = remainder[remainder.index <= d]
        remainder = remainder[remainder.index > d]
        if len(core) < h:
            raise ValueError("Not enough data to trend")
        result = (
            hma(core, h)
            if methodology == _HENDERSON
            else core.rolling(window=len(weights), center=True).apply(func=lambda x: (x * weights).sum())
        )
        trend = result if len(trend) == 0 else pd.concat([trend, result])
    return trend


def _extend_series_by_arima(s: pd.Series, freq: int, h: int, *, arima_extend: bool) -> pd.DataFrame:
    """Start the results frame: the original, the (optionally ARIMA-extended) series and the period in year."""
    if arima_extend:
        p_length = int(h / 2)
        forward = _make_projection(s, freq=freq, p_length=p_length, direction=1)
        back = _make_projection(s, freq=freq, p_length=p_length, direction=-1)
        combined = forward.combine_first(back)
    else:
        combined = s
    result = pd.DataFrame(combined)
    result.columns = pd.Index([EXTENDED])
    result.insert(0, ORIGINAL, s)  # put in first position
    index = _period_index(result.index)
    result[PERIOD] = index.quarter if index.freqstr[0] == "Q" else index.month
    return result


def _check_input_validity(s: pd.Series, discontinuity_list: Sequence[pd.Period]) -> int:
    """Check the series and the discontinuities; return the number of periods in a year."""
    if not isinstance(s, pd.Series):
        raise TypeError("The s parameter should be a pandas Series")
    if not isinstance(s.index, pd.PeriodIndex):
        raise TypeError("The s.index parameter should be a pandas PeriodIndex")
    if not (s.index.is_monotonic_increasing and s.index.is_unique):
        raise ValueError("The index for the s parameter should be unique and sorted")
    if any(s.isna()) or not all(np.isfinite(s)):
        raise ValueError("The s parameter contains NA or infinite values")

    for d in discontinuity_list:
        if not isinstance(d, pd.Period):
            raise TypeError("The values in the discontinuity_list should be Periods")
        if d not in s.index:
            raise ValueError(f"The discontinuity {d} not in the index of s")

    if s.index.freqstr[0] not in PERIODS_PER_YEAR:
        raise ValueError("The index for the s parameter should be monthly or quarterly data")
    n_periods = PERIODS_PER_YEAR[s.index.freqstr[0]]
    if len(s) < (n_periods * 2) + 1:
        raise ValueError("The input series is not long enough to decompose")
    return n_periods


def _ndiffs(y: np.ndarray, alpha: float = _KPSS_ALPHA, max_d: int = _MAX_ARIMA_ORDER) -> int:
    """Non-seasonal differencing order by successive KPSS level-stationarity tests (Hyndman-Khandakar)."""
    d, x = 0, np.asarray(y, dtype=float)
    while d < max_d and len(x) > _MIN_KPSS_LENGTH:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _stat, pval, *_ = kpss(x, regression="c", nlags="auto")
        except ValueError, OverflowError:
            break
        if pval >= alpha:  # fail to reject stationarity -> stop differencing
            break
        x = np.diff(x)
        d += 1
    return d


def _nsdiffs(y: np.ndarray, m: int, d: int, max_big_d: int = 1, thresh: float = _SEASONAL_ACF_THRESHOLD) -> int:
    """Seasonal differencing order: detrend by d differences, then test the autocorrelation at lag m."""
    if m < _MIN_SEASON_LENGTH or len(y) < 2 * m + d:
        return 0
    x = np.asarray(y, dtype=float)
    for _ in range(d):
        x = np.diff(x)
    big_d = 0
    while big_d < max_big_d and len(x) > 2 * m:
        a = acf(x, nlags=m, fft=False)
        if len(a) <= m or a[m] < thresh:
            break
        x = x[m:] - x[:-m]
        big_d += 1
    return big_d


def _neighbours(key: tuple[int, int, int, int], *, seasonal: bool) -> list[tuple[int, int, int, int]]:
    """List the orders one step (+/-1) from key; the seasonal orders only for a seasonal model."""
    p, q, big_p, big_q = key
    steps = [
        (p + 1, q, big_p, big_q),
        (p - 1, q, big_p, big_q),
        (p, q + 1, big_p, big_q),
        (p, q - 1, big_p, big_q),
    ]
    if seasonal:
        steps += [
            (p, q, big_p + 1, big_q),
            (p, q, big_p - 1, big_q),
            (p, q, big_p, big_q + 1),
            (p, q, big_p, big_q - 1),
        ]
    return steps


def _hill_climb(
    fit: Callable[[tuple[int, int, int, int]], tuple[SARIMAXResults | None, float]],
    best: tuple[tuple[int, int, int, int], SARIMAXResults | None, float],
    *,
    seasonal: bool,
) -> SARIMAXResults | None:
    """Move to the best improving single-order neighbour until none improves; return its fit.

    AIC strictly decreases each step over a finite order box, so this ends.
    """
    best_key, best_res, best_aic = best
    improved = True
    while improved:
        improved = False
        for neighbour in _neighbours(best_key, seasonal=seasonal):
            res, aic = fit(neighbour)
            if aic < best_aic:
                best_key, best_res, best_aic = neighbour, res, aic
                improved = True
    return best_res


def _auto_arima(y: np.ndarray, m: int, max_order: int = _MAX_ARIMA_ORDER) -> SARIMAXResults:
    """Stepwise auto-ARIMA on statsmodels SARIMAX, a maintained stand-in for pmdarima.auto_arima.

    The differencing orders d and D are fixed by tests, so AIC is comparable; the AR/MA
    orders p, q, P, Q are then chosen by a Hyndman-Khandakar stepwise search: fit a few
    seed models, then hill-climb to whichever single-order neighbour lowers the AIC,
    until none does. Seasonal orders are explored only when seasonal differencing is
    indicated. Returns a fitted SARIMAXResults.
    """
    y = np.asarray(y, dtype=float)
    d = _ndiffs(y, max_d=max_order)
    big_d = _nsdiffs(y, m, d, max_big_d=1)
    seasonal = big_d > 0
    trend = "c" if (d + big_d) < _MAX_DIFFERENCES_FOR_CONSTANT else "n"

    fitted: dict[tuple[int, int, int, int], tuple[SARIMAXResults | None, float]] = {}

    def fit(order: tuple[int, int, int, int]) -> tuple[SARIMAXResults | None, float]:
        """Fit one SARIMAX model, memoised; return (results or None, aic)."""
        p, q, big_p, big_q = order
        if any(not 0 <= v <= max_order for v in order):
            return None, np.inf
        if order in fitted:
            return fitted[order]
        seasonal_order = (big_p, big_d, big_q, m) if (big_d or big_p or big_q) else (0, 0, 0, 0)
        res: SARIMAXResults | None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = SARIMAX(
                    y,
                    order=(p, d, q),
                    seasonal_order=seasonal_order,
                    trend=trend,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)
            aic = float(res.aic) if np.isfinite(res.aic) else np.inf
        except ValueError, np.linalg.LinAlgError:
            res, aic = None, np.inf
        fitted[order] = (res, aic)
        return res, aic

    # Hyndman-Khandakar seed models (seasonal terms only when D > 0).
    seeds = (
        [(2, 2, 1, 1), (0, 0, 0, 0), (1, 0, 1, 0), (0, 1, 0, 1)]
        if seasonal
        else [(2, 2, 0, 0), (0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0)]
    )
    best_key: tuple[int, int, int, int] | None = None
    best_res: SARIMAXResults | None = None
    best_aic = np.inf
    for seed in seeds:
        p, q, big_p, big_q = (min(v, max_order) for v in seed)
        key = (p, q, big_p, big_q)
        res, aic = fit(key)
        if aic < best_aic:
            best_key, best_res, best_aic = key, res, aic

    if best_key is not None:
        best_res = _hill_climb(fit, (best_key, best_res, best_aic), seasonal=seasonal)

    if best_res is None:  # fallback: random walk with drift
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            best_res = SARIMAX(y, order=(0, 1, 0), trend="c").fit(disp=False)
    return best_res


def _make_projection(s: pd.Series, freq: int, p_length: int, direction: int) -> pd.Series:
    """Project a series p_length periods forward (direction 1) or back (-1) by auto-ARIMA; return it joined on."""
    values = s.to_numpy(dtype=float)
    if direction < 0:
        values = values[::-1]

    res = _auto_arima(values, m=freq)
    forecast = np.asarray(res.forecast(p_length))

    freqstr = _period_index(s.index).freqstr
    if direction < 0:
        idx = pd.period_range(end=s.index[0] - 1, periods=p_length, freq=freqstr)
        return pd.concat([pd.Series(forecast[::-1], index=idx), s])
    idx = pd.period_range(s.index[-1] + 1, periods=p_length, freq=freqstr)
    return pd.concat([s, pd.Series(forecast, index=idx)])


def _smooth_seasonal(
    series: pd.Series,
    n_periods: int,
    ignore_years: Sequence[int],
    *,
    constant: bool,
    years: int,
) -> pd.Series:
    """Smooth each period-in-year's seasonal effects across years: a constant mean or a centred rolling mean."""
    if n_periods not in (QUARTERS, MONTHS):
        raise ValueError(f"Expected {QUARTERS} or {MONTHS} periods a year, got {n_periods}")

    # one row per year, one column per period in the year
    index = _period_index(series.index)
    frame = pd.DataFrame(
        {
            "value": series.to_numpy(),
            "year": index.year,
            "period": index.quarter if n_periods == QUARTERS else index.month,
        }
    )
    ptable = frame.pivot_table(index="year", columns="period", values="value", dropna=False)

    for col in ptable:
        if constant and ignore_years:
            ptable[col] = ptable[col].mask(ptable.index.isin(ignore_years))
        if constant or (len(ptable) < years + _SEASONAL_SMOOTHER_SLACK):
            ptable[col] = ptable[col].mean(skipna=True)
        else:
            ptable[col] = ptable[col].rolling(window=years, center=True).mean()

    # back to a series, in year then period order
    long = (
        ptable.reset_index()
        .melt(id_vars="year", var_name="period", value_name="value")
        .sort_values(["year", "period"], kind="stable")
    )
    freq = index.freqstr
    returnable = pd.Series(
        long["value"].to_numpy(dtype=float),
        index=pd.PeriodIndex(
            [
                pd.Period(year=year, quarter=period, freq=freq)
                if n_periods == QUARTERS
                else pd.Period(year=year, month=period, freq=freq)
                for year, period in zip(long["year"], long["period"], strict=True)
            ]
        ),
    )
    if returnable.isna().any():
        returnable = _extend_series(returnable, n_periods)
    return returnable


def _extend_series(s: pd.Series, n_periods: int) -> pd.Series:
    """Fill missing seasonal factors at either end from the nearest same period in the cycle."""
    if s.notna().all():
        return s

    s = s.copy()  # do no harm
    half = int(len(s) / 2)
    core = s[s.notna()]
    core_index = _period_index(core.index)
    attribute = "quarter" if n_periods == QUARTERS else "month"

    def populate(destinations: pd.Index, from_which_end: int) -> None:
        for dest in destinations:
            source = core_index[getattr(core_index, attribute) == getattr(dest, attribute)][from_which_end]
            s.loc[dest] = core.loc[source]

    head = s.iloc[:half]
    populate(head[head.isna()].index, 0)
    tail = s.iloc[-half:]
    populate(tail[tail.isna()].index, -1)
    return s
