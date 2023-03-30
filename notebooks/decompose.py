"""Produce a simple time series decomposition."""

from operator import sub, truediv

import numpy as np
import pandas as pd

from henderson import hma

# --- A selection of seasonal smoothing weights, from which you can select
#     Note: these are end-weights, they are reversed for the start of a series
#     Note: your own weights in this form should also work
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
    constantSeasonal: bool = False,
    seasonalSmoother: tuple[np.array] = s3x5,
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
    -   constantSeasonal - bool - whether the seasonal component is
        constant or (slowly) variable
    -   seasonalSmoother - when not using a constantSeasonal, which
        of the seasonal smoothers to use (s3x3, s3x5 or s3x9) -
        default is s3x5 (ie over 7 years for monthly or quarterly data)

    Returns a pandas DataFrame with columns for each step in the
    decomposition process (enables debugging). The key columns in the
    DataFrame are:
    -   'Original' - the original series
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

    # --- initialise
    result = pd.DataFrame(s)
    result.columns = ["Original"]
    result["period"] = {
        "Q": result.index.quarter,
        "M": result.index.month,
    }[result.index.freqstr[0]]

    # --- determine the period
    n_periods = {"Q": 4, "M": 12}[s.index.freqstr[0]]
    if len(s) < (n_periods * 2) + 1:
        raise ValueError("The input series is not long enough to decompose")

    # --- settle the length of the Henderson moving average
    h = max(n_periods, 9)  # ABS uses 13-term HMA for monthly and 7-term for quarterly
    if h % 2 == 0:
        h += 1  # we need an odd number

    # --- On to the decomposition process:
    # - 1 - derive an initial estimate for the trend component
    result["1stTrendEst"] = s.rolling(
        window=n_periods + 1, min_periods=n_periods + 1, center=True
    ).mean()
    # Note: rolling mean leaves NA values at the start/end of the trend estimate.

    # - 2 - preliminary estimate of the seasonal component
    oper = truediv if model == "multiplicative" else sub
    result["1stSeasonalEst"] = oper(result["Original"], result["1stTrendEst"])

    # - 3 - smooth the seasonal
    result = _smoothSeasonalComponent(
        result,
        constantSeasonal=constantSeasonal,
        seasonalSmoother=seasonalSmoother,
        columnToBeSmoothed="1stSeasonalEst",
        newColumn="2ndSeasonalEst",
    )

    # - 4 - extend the smoothed seasonal estimate to full scale
    if any(result["2ndSeasonalEst"].isnull()):
        result = _extendSeries(
            result,
            periods=n_periods,
            columnToBeExtended="2ndSeasonalEst",
            newColumn="3rdSeasonalEst",
        )
    else:
        result["3rdSeasonalEst"] = result["2ndSeasonalEst"]

    # - 5 - preliminary estimate of the seasonally adjusted data
    result["1stSeasAdjEst"] = oper(result["Original"], result["3rdSeasonalEst"])

    # - 6 - a better estimate of the trend
    result["2ndTrendEst"] = hma(result["1stSeasAdjEst"], h)

    # - 7 - final estimate of the seasonal component
    result["4thSeasonalEst"] = oper(result["Original"], result["2ndTrendEst"])

    result = _smoothSeasonalComponent(
        result,
        constantSeasonal=constantSeasonal,
        seasonalSmoother=seasonalSmoother,
        columnToBeSmoothed="4thSeasonalEst",
        newColumn="Seasonal",
    )

    # - 8 - calculate remaining final estimates
    result["Seasonally Adjusted"] = oper(result["Original"], result["Seasonal"])
    result["Trend"] = hma(result["Seasonally Adjusted"], h)
    result["Irregular"] = oper(result["Seasonally Adjusted"], result["Trend"])

    # - 9 - our job here is done
    return result


# --- apply seasonal smoother
def _smoothSeasonalComponent(
    result: pd.DataFrame,
    constantSeasonal,
    seasonalSmoother,
    columnToBeSmoothed: str,
    newColumn: str,
):
    # get the key smoothing constants
    if not constantSeasonal:
        kS = len(seasonalSmoother)
        lenS = (len(seasonalSmoother) * 2) - 1
        centralS = seasonalSmoother[len(seasonalSmoother) - 1]

    # establish an empty return column ...
    result[newColumn] = np.repeat(np.nan, len(result))

    # populate the return column ...
    for u in result["period"].unique():
        # get each of of the seasonals
        thisSeason = result.loc[result["period"] == u, columnToBeSmoothed]

        # smooth to a constant seasonal value
        if constantSeasonal:
            thisSeasonSmoothed = pd.Series(
                np.repeat(thisSeason.mean(skipna=True), len(thisSeason)),
                index=thisSeason.index,
            )

        # smooth to a slowly changing seasonal value
        else:
            # drop NA values which result from step 1 in the decomp process
            thisSeason = thisSeason.dropna()

            # apply the seasonalSmoother
            thisSeasonSmoothed = thisSeason.rolling(
                window=lenS, min_periods=lenS, center=True
            ).apply(func=lambda x: (x * centralS).sum())
            # thisSeasonSmoothed = pd.rolling_apply(thisSeason, window=lenS,
            #    func=lambda x: (x * centralS).sum(), min_periods=lenS, center=True)

            # for short series this process results in no data, find a simple mean
            if all(thisSeasonSmoothed.isnull()):
                # same treatment as constant seasonal value above
                thisSeasonSmoothed = pd.Series(
                    np.repeat(thisSeason.mean(skipna=True), len(thisSeason)),
                    index=thisSeason.index,
                )

            # handle the end-point problem ...
            for i in range(kS - 1):
                if np.isnan(thisSeasonSmoothed.iat[i]):
                    thisSeasonSmoothed.iat[i] = (
                        thisSeason.iloc[0 : (i + kS)] * (seasonalSmoother[i][::-1])
                    ).sum()  # note: reversed order at start

            for i in range(len(thisSeason) - 1, len(thisSeason) - kS, -1):
                if np.isnan(thisSeasonSmoothed.iat[i]):
                    thisSeasonSmoothed.iat[i] = (
                        thisSeason.iloc[(i - (kS - 1)) : len(thisSeason)]
                        * seasonalSmoother[len(thisSeason) - 1 - i]
                    ).sum()

        # package up season by season ...
        result[newColumn] = result[newColumn].where(
            result[newColumn].notnull(), other=thisSeasonSmoothed
        )

    return result


# --- extend seasonal components to the full length of series
def _extendSeries(result, periods, columnToBeExtended, newColumn):
    result[newColumn] = result[columnToBeExtended].copy()

    def fillup(result, fill, startPoint, endPoint):
        i = startPoint
        while True:
            p = result.index[i]
            result[newColumn].iat[i] = fill[newColumn].at[result["period"].iat[i]]
            if p >= endPoint:
                break
            i += 1

    # back-cast
    if np.isnan(result.iat[0, result.columns.get_loc(newColumn)]):
        fill = pd.DataFrame(result[newColumn].dropna().iloc[0:periods])
        fill["period"] = result["period"][fill.index[0] : fill.index[len(fill) - 1]]
        endPoint = fill.index[0] - 1
        fill.index = fill["period"]
        fillup(result=result, fill=fill, startPoint=0, endPoint=endPoint)

    # forward-cast
    if np.isnan(result.iat[len(result) - 1, result.columns.get_loc(newColumn)]):
        fill = result[newColumn].dropna()
        fill = pd.DataFrame(fill[(len(fill) - periods) : len(fill)])
        fill["period"] = result["period"][fill.index[0] : fill.index[len(fill) - 1]]
        startPoint = result.index.get_loc(fill.index[len(fill) - 1] + 1)
        fill.index = fill["period"]
        endPoint = result.index[len(result) - 1]
        fillup(result=result, fill=fill, startPoint=startPoint, endPoint=endPoint)

    return result
