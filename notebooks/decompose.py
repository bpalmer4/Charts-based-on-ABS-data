"""Produce a naive time series decomposition."""

import sys
from operator import sub, truediv
from typing import cast, Final, Sequence

import numpy as np
import pandas as pd
try:
    from pmdarima.arima import auto_arima
except ImportError:
    print("Could not import auto_arima from pmdarima")

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
def decompose(
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
    result[FINAL_TREND] = _get_trend(result[FINAL_SEASADJ], h, discontinuity_list)
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
    s: pd.Series, 
    freq: int, 
    h: int, 
    arima_extend: bool
) -> pd.DataFrame:
    """Use auto_arima() to extend the series in each direction.
    Returns the results dataframe that will be subsequently populated."""

    if arima_extend and "pmdarima" in sys.modules:
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


def _make_projection(
    s: pd.Series, freq: int, p_length: int, direction: int
) -> pd.Series:
    """Use auto_arima to project a series into the future/past.
    Arguments
    s - Series to be projected.
    freq - positive int - period frequency - 4 or 12
    direction - 1 or -1 - direction we are projecting towards.
    Returns - projected series."""

    t = s
    if direction < 0:
        t = s[::-1]
        t = t.reset_index(drop=True)

    model = auto_arima(
        t,
        m=freq,
        seasonal=True,
        d=None,
        test="adf",
        start_p=0,
        start_q=0,
        max_p=3,
        max_q=3,
        D=None,
        trace=False,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )
    # print(model.summary())

    projection = model.predict(n_periods=p_length, return_conf_int=False)

    if direction < 0:
        projection.index = s.index[0] - pd.Series(range(1, p_length + 1))
        projection = projection.sort_index()
        extended_series = pd.concat([projection, s])
    else:
        extended_series = pd.concat([s, projection])
    return extended_series


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
            ptable.loc[ptable.index.isin(ignore_years), col] = np.nan
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
    freq = series.index.freqstr
    if n_periods == 4:
        index = pd.PeriodIndex.from_fields(year=year, quarter=period, freq=freq)
    else:
        index = pd.PeriodIndex.from_fields(year=year, month=period, freq=freq)
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
