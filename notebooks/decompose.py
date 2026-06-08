"""Produce a naive time series decomposition."""

import warnings
from operator import sub, truediv
from typing import cast, Final, Sequence

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, kpss

from henderson import hma


# --- decomposition process steps ---
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


# --- public Decomposition function
def decompose(  # pylint: disable=too-many-arguments,disable=too-many-positional-arguments
    s: pd.Series,
    model: str = "multiplicative",
    arima_extend=False,
    constant_seasonal: bool = False,
    len_seasonal_smoother: int = 20,  # years
    discontinuity_list: Sequence[pd.Period] = (),
    ignore_years: Sequence[int] = (),
) -> None | pd.DataFrame:
    """The simple decomposition of a pandas Series s into its trend, seasonal
    and irregular components. The default is a multiplicative model:
    --> Original(t) = Trend(t) * Seasonal(t) * Irregular(t).
    Can specify an additive model:
    --> Original(t) = Trend(t) + Seasonal(t) + Irregular(t).

    Parameters:
    -   s - the pandas Series, without any missing or NA values,
        and sorted in ascending order - the index must be a period
        index with a frequency of M or Q
    -   model - string - either 'multiplicative' or 'additive'
    -   arima_extend - bool - whether to apply ARIMA extentions to the
        priginal series before seasonal decomposition
    -   constant_seasonal - bool - whether the seasonal component is
        constant or (slowly) variable
    -   len_seasonal_smoother - int - number of years for slowly variable
        seasonal smoothing (if not constant_seasonal)
    -   discontinuity_list - Sequence[pd.Period] - list of the last
        dates immediately before a sequence discontinuity. Useful for
        reflecting abrupt movements in the data that can be pegged to
        a significant or meaningful event (eg the COVID pandemic)
    -   ignore_years - Sequence[int] - allow for years to be removed from
        constant_seasonal analysis - useful for removing COVID impacts in
        2020 and 2021 from the seasonal weights.

    Returns a pandas DataFrame with columns for each step in the
    decomposition process (largely for debugging). The key result
    columns in the DataFrame are:
    -   'Original' - the original series
    -   'Extended' - the ARIMA extended series before decomposition
    -   'Seasonally Adjusted' - the seasonally adjusted series
    -   'Trend' - the trend of the seasonally adjusted series
    -   'Seasonal Weights' - the seasonal component found through the
        decomposition process
    -   'Irregular' - the irregular component found through the
        decomposition process

    Notes:
    Based on ideas gleaned from the Australian Bureau of Statistics:
        ABS (2005), "An Introductory Course on Times Series
        Analysis -- Electronic Delivery", Catalogue: 1346,0.55.001.
    Does not adjust for moving holidays, public holidays, variable number
    of working days in month, etc. (ie. it is quite a simple decomposition)."""

    # - preliminaries
    n_periods = _check_input_validity(s, discontinuity_list)
    h = _calculate_henderson_length(n_periods)
    oper = truediv if model == "multiplicative" else sub
    result = _extend_series_by_arima(s, n_periods, h, arima_extend)

    # - intermediate decomposition
    result[FIRST_TREND] = _get_trend(
        result[EXTENDED],
        h,
        (),  # no discontinuities at this initial stage
        "Other",  # first pass trend is a simpler rolling average
    )
    result[FIRST_SEAS] = oper(result[EXTENDED], result[FIRST_TREND])
    result[SECOND_SEAS] = _smooth_seasonal(
        result[FIRST_SEAS],
        constant_seasonal,
        len_seasonal_smoother,
        n_periods,
        ignore_years,
    )
    result[FIRST_SEASADJ] = oper(result[EXTENDED], result[SECOND_SEAS])
    result[SECOND_TREND] = _get_trend(result[FIRST_SEASADJ], h, discontinuity_list)
    result[THIRD_SEAS] = oper(result[EXTENDED], result[SECOND_TREND])

    # - final decomposition results
    result[FINAL_SEASONAL] = _smooth_seasonal(
        result[THIRD_SEAS],
        constant_seasonal,
        len_seasonal_smoother,
        n_periods,
        ignore_years,
    )
    result[FINAL_SEASADJ] = oper(result[ORIGINAL], result[FINAL_SEASONAL])
    # Compute the final trend on the EXTENDED seasonally-adjusted series so its
    # endpoints get symmetric (not Musgrave-lagged) Henderson weights when
    # arima_extend is on. With arima_extend off, EXTENDED == ORIGINAL, so this is
    # identical to trending FINAL_SEASADJ.
    result[FINAL_TREND] = _get_trend(
        oper(result[EXTENDED], result[FINAL_SEASONAL]), h, discontinuity_list
    )
    result[FINAL_IRREGULAR] = oper(result[FINAL_SEASADJ], result[FINAL_TREND])
    return result


#  === private methods below ===
_HENDERSON = "Henderson"


def _get_trend(
    s: pd.Series,
    h: int,
    discontinuity_list: Sequence[pd.Period],
    methodology: str = _HENDERSON,
) -> pd.Series:
    """Get trend data taking account of discontinuities."""

    if methodology != _HENDERSON:
        # construct a simple weighted smoother ...
        weights = np.array([1] + [2] * max(1, h - 2) + [1])
        weights = weights / np.sum(weights)  # weights sum to one
        discontinuity_list = ()  # can only do this with HMA

    discontinuity_list = list(discontinuity_list) + [s.index[-1]]
    remainder = s.dropna().copy()
    trend = pd.Series()
    for d in discontinuity_list:
        core = remainder[remainder.index <= d]
        remainder = remainder[remainder.index > d]
        if len(core) < h:
            raise ValueError("Not enough data to trend")
        result = (
            hma(core, h)
            if methodology == "Henderson"
            else (
                # use a simple weighted smoother ...
                core.rolling(window=len(weights), center=True).apply(
                    func=lambda x: (x * weights).sum()
                )
            )
        )
        trend = result if len(trend) == 0 else pd.concat([trend, result])
    return trend


def _extend_series_by_arima(
    s: pd.Series, freq: int, h: int, arima_extend: bool
) -> pd.DataFrame:
    """Use auto_arima() to extend the series in each direction.
    Returns the results dataframe that will be subsequently populated."""

    if arima_extend:
        p_length = int(h / 2)
        forward = _make_projection(s, freq=freq, p_length=p_length, direction=1)
        back = _make_projection(s, freq=freq, p_length=p_length, direction=-1)
        combined = forward.combine_first(back)
    else:
        combined = s
    result = pd.DataFrame(combined)
    result.columns = pd.Index([EXTENDED])
    result.insert(0, ORIGINAL, s)  # put in first prosition
    result[PERIOD] = {
        "Q": cast(pd.PeriodIndex, result.index).quarter,
        "M": cast(pd.PeriodIndex, result.index).month,
    }[cast(pd.PeriodIndex, result.index).freqstr[0]]

    return result


def _calculate_henderson_length(n_periods: int) -> int:
    """Settle the length of the Henderson moving average (must be odd).
    Note: ABS uses 13-term HMA for monthly and 7-term for quarterly
    Here we are using 13-term for monthly and 9-term for quarterly.
    Returns zero (error) if n_periods not in [4, 12]."""

    return {12: 13, 4: 9}.get(n_periods, 0)


def _check_input_validity(
    s: pd.Series,
    discontinuity_list: Sequence,
) -> int:
    """Perform sanity checks. Return number of periods in PerdioIndex."""

    if not isinstance(s, pd.Series):
        raise TypeError("The s parameter should be a pandas Series")
    if not isinstance(s.index, pd.PeriodIndex):
        raise TypeError("The s.index parameter should be a pandas PeriodIndex")
    if not (s.index.is_monotonic_increasing and s.index.is_unique):
        raise ValueError("The index for the s parameter should be unique and sorted")
    if any(s.isnull()) or not all(np.isfinite(s)):
        raise ValueError("The s parameter contains NA or infinite values")

    for d in discontinuity_list:
        if not isinstance(d, pd.Period):
            raise TypeError("The values in the discontinuity_list should be Periods")
        if d not in s.index:
            raise ValueError(f"The discontinuity {d} not in the index of s")

    acceptable = {"Q": 4, "M": 12}
    if s.index.freqstr[0] not in acceptable:
        raise ValueError(
            "The index for the s parameter should be monthly or quarterly data"
        )

    n_periods = acceptable[s.index.freqstr[0]]
    if len(s) < (n_periods * 2) + 1:
        raise ValueError("The input series is not long enough to decompose")

    return n_periods


_MAX_ARIMA_ORDER = 2


def _ndiffs(y: np.ndarray, alpha: float = 0.05, max_d: int = _MAX_ARIMA_ORDER) -> int:
    """Non-seasonal differencing order via successive KPSS level-stationarity
    tests (the Hyndman-Khandakar approach)."""

    d, x = 0, np.asarray(y, dtype=float)
    while d < max_d and len(x) > 12:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _stat, pval, *_ = kpss(x, regression="c", nlags="auto")
        except (ValueError, OverflowError):
            break
        if pval >= alpha:  # fail to reject stationarity -> stop differencing
            break
        x = np.diff(x)
        d += 1
    return d


def _nsdiffs(
    y: np.ndarray, m: int, d: int, max_big_d: int = 1, thresh: float = 0.64
) -> int:
    """Seasonal differencing order: detrend by d differences, then test the
    autocorrelation at the seasonal lag m."""

    if m < 2 or len(y) < 2 * m + d:
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


def _auto_arima(y: np.ndarray, m: int, max_order: int = _MAX_ARIMA_ORDER) -> object:
    """Compact auto-ARIMA on statsmodels SARIMAX -- a maintained stand-in for the
    abandoned pmdarima.auto_arima. Differencing orders d, D are fixed by tests
    (so AIC is comparable); p, q, P, Q are chosen by AIC over [0, max_order].
    The seasonal grid is skipped entirely when no seasonal differencing is
    indicated. Returns a fitted SARIMAXResults."""

    y = np.asarray(y, dtype=float)
    d = _ndiffs(y, max_d=max_order)
    big_d = _nsdiffs(y, m, d, max_big_d=1)
    seasonal_grid = range(max_order + 1) if big_d > 0 else [0]
    trend = "c" if (d + big_d) < 2 else "n"

    best, best_aic = None, np.inf
    for p in range(max_order + 1):
        for q in range(max_order + 1):
            for big_p in seasonal_grid:
                for big_q in seasonal_grid:
                    seasonal_order = (
                        (big_p, big_d, big_q, m)
                        if (big_d or big_p or big_q)
                        else (0, 0, 0, 0)
                    )
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
                    except (ValueError, np.linalg.LinAlgError):
                        continue
                    if np.isfinite(res.aic) and res.aic < best_aic:
                        best, best_aic = res, res.aic

    if best is None:  # fallback: random walk with drift
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            best = SARIMAX(y, order=(0, 1, 0), trend="c").fit(disp=False)
    return best


def _make_projection(
    s: pd.Series, freq: int, p_length: int, direction: int
) -> pd.Series:
    """Use a statsmodels auto-ARIMA to project a series p_length periods into the
    future (direction=1) or past (direction=-1), and return the series with the
    projection concatenated on.
    Arguments
    s - Series to be projected (PeriodIndex).
    freq - positive int - period frequency - 4 or 12.
    direction - 1 or -1 - direction we are projecting towards.
    Returns - the input series with the projection concatenated on."""

    values = s.to_numpy(dtype=float)
    if direction < 0:
        values = values[::-1]

    res = _auto_arima(values, m=freq)
    forecast = np.asarray(res.forecast(p_length))

    freqstr = cast(pd.PeriodIndex, s.index).freqstr
    if direction < 0:
        idx = pd.period_range(end=s.index[0] - 1, periods=p_length, freq=freqstr)
        projection = pd.Series(forecast[::-1], index=idx)
        return pd.concat([projection, s])
    idx = pd.period_range(s.index[-1] + 1, periods=p_length, freq=freqstr)
    projection = pd.Series(forecast, index=idx)
    return pd.concat([s, projection])


def _smooth_seasonal(
    series: pd.Series,
    constant_seasonal: bool,
    len_seasonal_smoother: int,  # in years
    n_periods: int,
    ignore_years: Sequence[int],
) -> pd.Series:
    # preliminary
    assert n_periods in (4, 12)

    # put into a table with seasonal columns
    frame = pd.DataFrame(series.copy())  # copy to avoid any harm
    frame.columns = pd.Index(["value"])
    frame["year"] = cast(pd.PeriodIndex, frame.index).year
    attribute = "quarter" if n_periods == 4 else "month"
    frame["period"] = getattr(frame.index, attribute)
    ptable = frame.pivot(columns="period", index="year", values="value")

    # average seasonal effects
    for col in ptable:
        if constant_seasonal and ignore_years:
            # mypy chokes on this next line - but it is fine ...
            ptable.loc[ptable.index.isin(ignore_years), col] = np.nan  # type: ignore[index]
            # print(ptable[col].values)
        if constant_seasonal or (len(ptable) < len_seasonal_smoother + 3):
            ptable[col] = ptable[col].mean(skipna=True)
        else:
            ptable[col] = (
                ptable[col].rolling(window=len_seasonal_smoother, center=True).mean()
            )

    # return to a series
    returnable = cast(pd.Series, ptable.stack(future_stack=True))
    year = returnable.index.get_level_values(0).values
    period = returnable.index.get_level_values(1).values
    freq = cast(pd.PeriodIndex, series.index).freqstr
    if n_periods == 4:
        index = pd.PeriodIndex.from_fields(  # type: ignore[attr-defined]
            year=year, quarter=period, freq=freq
        )
    else:
        index = pd.PeriodIndex.from_fields(  # type: ignore[attr-defined]
            year=year, month=period, freq=freq
        )
    returnable.index = index
    if returnable.isna().any():
        returnable = _extend_series(returnable, n_periods)

    return returnable


def _extend_series(s: pd.Series, n_periods: int) -> pd.Series:
    """Extend a seasonal series forward and back at the tails,
    to fill in any NA values, using closest period in cycle
    to select gap filler."""

    if s.notna().all():
        # nothing to do
        return s

    # preliminaries
    s = s.copy()  # do no harm
    half = int(len(s) / 2)
    core = s[s.notna()]
    attribute = "quarter" if n_periods == 4 else "month"

    # a short copy procedure ...
    def populate(destinations: pd.PeriodIndex, from_which_end: int):
        for dest in destinations:
            source = core[
                getattr(core.index, attribute) == getattr(dest, attribute)
            ].index[from_which_end]
            s.at[dest] = core.at[source]

    # fix up front-end
    head = s.iloc[:half]
    head_fix = cast(pd.PeriodIndex, head[head.isna()].index)
    populate(head_fix, 0)

    # fix up back-end
    tail = s.iloc[-half:]
    tail_fix = cast(pd.PeriodIndex, tail[tail.isna()].index)
    populate(tail_fix, -1)

    return s
