"""Produce a simple time series decomposition."""

from operator import sub, truediv
from typing import cast

import numpy as np
import pandas as pd
from pmdarima.arima import auto_arima

from henderson import hma

# --- A selection of seasonal smoothing weights, from which you can select
#     Note: these are end-weights, they are reversed for the start of a series
#     Note: your own weights in this form should also work
#     Note: the central smoother should be the last one in the tuple.
#           the edge case smoothers come before the central smoother.
s3x3 = (
    np.array([5, 11, 11]) / 27.0,
    np.array([3, 7, 10, 7]) / 27.0,
    np.array([1, 2, 3, 2, 1]) / 9.0,
)

s3x5 = (
    np.array([9, 17, 17, 17]) / 60.0,
    np.array([4, 11, 15, 15, 15]) / 60.0,
    np.array([4, 8, 13, 13, 13, 9]) / 60.0,
    np.array([1, 2, 3, 3, 3, 2, 1]) / 15.0,
)

s3x9 = (
    np.array([0.051, 0.112, 0.173, 0.197, 0.221, 0.246]),
    np.array([0.028, 0.092, 0.144, 0.160, 0.176, 0.192, 0.208]),
    np.array([0.032, 0.079, 0.123, 0.133, 0.143, 0.154, 0.163, 0.173]),
    np.array([0.034, 0.075, 0.113, 0.117, 0.123, 0.128, 0.132, 0.137, 0.141]),
    np.array([0.034, 0.073, 0.111, 0.113, 0.114, 0.116, 0.117, 0.118, 0.120, 0.084]),
    np.array([1, 2, 3, 3, 3, 3, 3, 3, 3, 2, 1]) / 27.0,
)


# --- public Decomposition function
def decompose(
    s: pd.Series,
    model: str = "multiplicative",
    arima_extend=False,
    constant_seasonal: bool = False,
    seasonal_smoother: tuple[np.ndarray, ...] = s3x5,
) -> None | pd.DataFrame:
    """The simple decomposition of a pandas Series s into its trend, seasonal
    and irregular components. The default is a multiplicative model:
    --> Original(t) = Trend(t) * Seasonal(t) * Irregular(t).
    Can specify an additive model:
    --> Original(t) = Trend(t) + Seasonal(t) + Irregular(t).

    Parameters:
    -   s - the pandas Series, without any missing or NA values,
        and sorted in ascending order - the index should be a period
        index with a frequency of M or Q
    -   model - string - either 'multiplicative' or 'additive'
    -   arima_extend - whether to apply arima extentions
    -   constant_seasonal - bool - whether the seasonal component is
        constant or (slowly) variable
    -   seasonal_smoother - when not using a constantSeasonal, which
        of the seasonal smoothers to use (s3x3, s3x5 or s3x9) -
        default is s3x5 (ie over 7 years for monthly or quarterly data)

    Returns a pandas DataFrame with columns for each step in the
    decomposition process (enables debugging). The key columns in the
    DataFrame are:
    -   'Original' - the original series
    -   'Extended' - the arima extended series for decomposition
    -   'Seasonally Adjusted' - the seasonally adjusted series
    -   'Trend' - the trend of the seasonally adjusted series
    -   'Seasonal' - the seasonal component found through the
        decomposition process
    -   'Irregular' - the irregular component found through the
        decomposition process

    Notes:
    Based on ideas gleaned from the Australian Bureau of Statistics:
        ABS (2005), "An Introductory Course on Times Series
        Analysis -- Electronic Delivery", Catalogue: 1346,0.55.001.
    Does not adjust for moving holidays, public holidays, variable number
    of working days in month, etc. (ie. it is quite a simple decomposition)."""

    # --- sanity checks
    if not isinstance(s, pd.Series):
        raise TypeError("The s parameter should be a pandas Series")
    if not isinstance(s.index, pd.PeriodIndex):
        raise TypeError("The s.index parameter should be a pandas PeriodIndex")
    if not (s.index.is_monotonic_increasing and s.index.is_unique):
        raise ValueError("The index for the s parameter should be unique and sorted")
    if any(s.isnull()) or not all(np.isfinite(s)):
        raise ValueError("The s parameter contains NA or infinite values")
    ACCEPTABLE = {"Q": 4, "M": 12}
    if s.index.freqstr[0] not in ACCEPTABLE.keys():
        raise ValueError(
            "The index for the s parameter should be monthly or quarterly data"
        )

    # --- determine the period
    n_periods = ACCEPTABLE[s.index.freqstr[0]]
    if len(s) < (n_periods * 2) + 1:
        raise ValueError("The input series is not long enough to decompose")

    # --- settle the length of the Henderson moving average
    h = max(n_periods, 7)  # ABS uses 13-term HMA for monthly and 7-term for quarterly
    if h % 2 == 0:
        h += 1  # we need an odd number

    # --- On to the decomposition process:
    # - 0 - use auto_arima() to extend the series in each direction
    if arima_extend:
        forward = make_projection(s, freq=n_periods, direction=1)
        back = make_projection(s, freq=n_periods, direction=-1)
        combined = forward.combine_first(back)
    else:
        combined = s
    result = pd.DataFrame(combined)
    result.columns = pd.Index(["Extended"])
    result["Original"] = s
    result["period"] = {
        "Q": cast(pd.PeriodIndex, result.index).quarter,
        "M": cast(pd.PeriodIndex, result.index).month,
    }[cast(pd.PeriodIndex, result.index).freqstr[0]]
    s = combined

    # - 1 - derive an initial estimate for the trend component
    result["1stTrendEst"] = s.rolling(
        window=n_periods + 1, min_periods=n_periods + 1, center=True
    ).mean()
    # Note: rolling mean leaves NA values at the start/end of the trend estimate.

    # - 2 - preliminary estimate of the seasonal component
    oper = truediv if model == "multiplicative" else sub
    result["1stSeasonalEst"] = oper(result["Extended"], result["1stTrendEst"])

    # - 3 - smooth the seasonal
    result = _smooth_seasonal_component(
        result,
        constant_seasonal=constant_seasonal,
        seasonal_smoother=seasonal_smoother,
        column_to_be_smoothed="1stSeasonalEst",
        new_column="2ndSeasonalEst",
    )

    # - 4 - extend the smoothed seasonal estimate to full scale
    if any(result["2ndSeasonalEst"].isnull()):
        result = _extend_series(
            result,
            periods=n_periods,
            column_to_be_extended="2ndSeasonalEst",
            new_column="3rdSeasonalEst",
        )
    else:
        result["3rdSeasonalEst"] = result["2ndSeasonalEst"]

    # - 5 - preliminary estimate of the seasonally adjusted data
    result["1stSeasAdjEst"] = oper(result["Extended"], result["3rdSeasonalEst"])

    # - 6 - a better estimate of the trend
    result["2ndTrendEst"] = hma(result["1stSeasAdjEst"], h)

    # - 7 - final estimate of the seasonal component
    result["4thSeasonalEst"] = oper(result["Extended"], result["2ndTrendEst"])

    result = _smooth_seasonal_component(
        result,
        constant_seasonal=constant_seasonal,
        seasonal_smoother=seasonal_smoother,
        column_to_be_smoothed="4thSeasonalEst",
        new_column="Seasonal",
    )

    # - 8 - calculate remaining final estimates
    result["Seasonally Adjusted"] = oper(result["Original"], result["Seasonal"])
    result["Trend"] = hma(result["Seasonally Adjusted"].dropna(), h)
    result["Irregular"] = oper(result["Seasonally Adjusted"], result["Trend"])

    # - 9 - our job here is done
    return result


# --- projection
def make_projection(s: pd.Series, freq: int, direction: int) -> pd.Series:
    """Use auto_arima to project a series into the future/past.
    Arguments
    s - Series to be projected.
    freq - positive int - frequency of series - should be 4 or 12
    direction - 1 or -1 - direction we are projecting towards.
    Returns - projected series."""

    t = s
    if direction < 0:
        orig_index = s.index
        t = s[::-1]
        t = t.reset_index(drop=True)

    arima = auto_arima(
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
        trace=True,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )
    print(arima.summary())
    forward = int(freq / 2) if freq >= 12 else freq
    fc = arima.predict(n_periods=int(forward), return_conf_int=False)

    if direction < 0:
        fc.index = s.index[0] - pd.Series(range(1, forward + 1))
        fc = fc.sort_index()
        returnable = pd.concat([fc, s])
    else:
        returnable = pd.concat([s, fc])
    return returnable


# --- apply seasonal smoother
def _smooth_seasonal_component(
    result: pd.DataFrame,
    constant_seasonal,
    seasonal_smoother,
    column_to_be_smoothed: str,
    new_column: str,
):
    # get the key smoothing constants
    if not constant_seasonal:
        n_smoothers = len(seasonal_smoother)
        smoother_size = (n_smoothers * 2) - 1
        central_smoother = seasonal_smoother[n_smoothers - 1]

    # establish an empty return column ...
    result[new_column] = np.repeat(np.nan, len(result))

    # populate the return column ...
    for u in result["period"].unique():
        # get each of of the seasonals
        this_season = result.loc[result["period"] == u, column_to_be_smoothed]

        # smooth to a constant seasonal value
        if constant_seasonal:
            this_season_smoothed = pd.Series(
                np.repeat(this_season.mean(skipna=True), len(this_season)),
                index=this_season.index,
            )

        # smooth to a slowly changing seasonal value
        else:
            # drop NA values which result from step 1 in the decomp process
            this_season = this_season.dropna()

            # apply the seasonal_smoother
            this_season_smoothed = this_season.rolling(
                window=smoother_size, min_periods=smoother_size, center=True
            ).apply(func=lambda x: (x * central_smoother).sum())
            # this_season_smoothed = pd.rolling_apply(this_season, window=smoother_size,
            #    func=lambda x: (x * central_smoother).sum(), min_periods=smoother_size,
            #    center=True)

            # for short series this process results in no data, find a simple mean
            if all(this_season_smoothed.isnull()):
                # same treatment as constant seasonal value above
                this_season_smoothed = pd.Series(
                    np.repeat(this_season.mean(skipna=True), len(this_season)),
                    index=this_season.index,
                )

            # handle the end-point problem ...
            for i in range(n_smoothers - 1):
                if np.isnan(this_season_smoothed.iat[i]):
                    this_season_smoothed.iat[i] = (
                        this_season.iloc[0 : (i + n_smoothers)]
                        * (seasonal_smoother[i][::-1])
                    ).sum()  # note: reversed order at start

            for i in range(len(this_season) - 1, len(this_season) - n_smoothers, -1):
                if np.isnan(this_season_smoothed.iat[i]):
                    this_season_smoothed.iat[i] = (
                        this_season.iloc[(i - (n_smoothers - 1)) : len(this_season)]
                        * seasonal_smoother[len(this_season) - 1 - i]
                    ).sum()

        # package up season by season ...
        result[new_column] = result[new_column].where(
            result[new_column].notnull(), other=this_season_smoothed
        )

    return result


# --- extend seasonal components to the full length of series
def _extend_series(result, periods, column_to_be_extended, new_column):
    result[new_column] = result[column_to_be_extended].copy()

    def fillup(result, fill, start_point, end_point):
        i = start_point
        while True:
            p = result.index[i]
            result[new_column].iat[i] = fill[new_column].at[result["period"].iat[i]]
            if p >= end_point:
                break
            i += 1

    # back-cast
    if np.isnan(result.iat[0, result.columns.get_loc(new_column)]):
        fill = pd.DataFrame(result[new_column].dropna().iloc[0:periods])
        fill["period"] = result["period"][fill.index[0] : fill.index[len(fill) - 1]]
        end_point = fill.index[0] - 1
        fill.index = fill["period"]
        fillup(result=result, fill=fill, start_point=0, end_point=end_point)

    # forward-cast
    if np.isnan(result.iat[len(result) - 1, result.columns.get_loc(new_column)]):
        fill = result[new_column].dropna()
        fill = pd.DataFrame(fill[(len(fill) - periods) : len(fill)])
        fill["period"] = result["period"][fill.index[0] : fill.index[len(fill) - 1]]
        start_point = result.index.get_loc(fill.index[len(fill) - 1] + 1)
        fill.index = fill["period"]
        end_point = result.index[len(result) - 1]
        fillup(result=result, fill=fill, start_point=start_point, end_point=end_point)

    return result
