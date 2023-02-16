"""Calculate a Henderson moving average.
   Use: from henderson import hma"""

import pandas as pd
import numpy as np

_cache:dict[int | tuple[int, int], np.ndarray] = {}

def hma_symmetric_weights(n: int) -> np.ndarray:
    """ derive an n-term array of symmetric 'Henderson Moving Average' weights
        formula from ABS (2003), 'A Guide to Interpreting Time Series', page 41.
        returns a numpy array of symmetric Henderson weights indexed from 0 to n-1"""

    if n in _cache:
        return _cache[n]

    # calculate the constant denominator and terms
    m: int = int((n - 1) // 2) # the mid point - n must be odd
    m1: int = (m + 1) * (m + 1)
    m2: int = (m + 2) * (m + 2)
    d: float = float(8 * (m + 2) * (m2 - 1) * (4 * m2 -1) * (4 * m2 - 9) * ( 4 * m2 - 25))
    m3: int = (m + 3) * (m + 3)

    # calculate the weights
    sym_weights = np.repeat(np.nan, n) # indexed from 0 to n-1
    for j in range(m + 1):
        j2: int = j * j
        a_weight: float = (
            (315 * (m1 - j2) * (m2 - j2) * (m3 - j2) * (3 * m2 - 11 * j2 - 16)) / d
        )
        sym_weights[(m + j)] = a_weight
        if j > 0:
            sym_weights[(m - j)] = a_weight

    sym_weights.flags.writeable = False # make quasi-immutable
    _cache[n] = sym_weights
    return sym_weights


def hma_asymmetric_weights(m: int, sym_weights: np.ndarray) -> np.ndarray:
    """Calculate the asymmetric end-weights

        sym_weights --> an array of symmetrical henderson weights (from above function)
        m --> the number of asymmetric weights sought; where m < len(w);

        returns a numpy array of asymmetrical weights, indexed from 0 to m-1;

        formula from Mike Doherty (2001), 'The Surrogate Henderson Filters in X-11',
        Aust, NZ J of Stat. 43(4), 2001, pp901-999; see formula (1) on page 903"""

    # - get n from the weights array and check the cache
    n: int = len(sym_weights)
    cache_key = (n, m)
    if cache_key in _cache:
        return _cache[cache_key]

    # --- let's build up Doherty's formula (1) from the top of page 903
    # - the second chunk of the formula
    sum_residual: float = sym_weights[range(m, n)].sum() / float(m)

    # - the last chunk of the formula
    sum_end: float = 0.0
    for i in range(m + 1, n + 1):
        sum_end += (float(i) - ((m + 1.0) / 2.0)) * sym_weights[i - 1]

    # - the beta squared / sigma squared - formula at the bottom of page 904
    ic: float = 1.0
    if 13 <= n < 15:
        ic = 3.5
    elif n >= 15:
        ic = 4.5
    b2s2: float = (4.0 / np.pi) / (ic * ic)

    # - the gnarly bit in the middle of the formula
    denominator: float = 1.0 + ((m * (m - 1.0) * (m + 1.0) / 12.0 ) * b2s2)
    asym_wts: np.ndarray = np.repeat(np.nan, m) # return series - created empty
    for r in range(m): # r ranges 0 to m-1; but the formulae assumes 1 to m
        numerator = ((r + 1.0) - (m + 1.0) / 2.0) * b2s2
        # - finally putting it all together
        asym_wts[r] = sym_weights[r] + sum_residual + (numerator / denominator) * sum_end

    asym_wts.flags.writeable = False # make quasi-immutable
    _cache[cache_key] = asym_wts
    return asym_wts


def hma(series: pd.Series, n: int) -> pd.Series:
    """ Calculate an n-term Henderson Moving Average for the pandas Series series
        Note: series is ordered, contiguous and without missing data."""

    # - some simple sanity checks
    if not isinstance(series, pd.core.series.Series):
        raise TypeError('The series argument must be a pandas Series')
    if series.isna().sum() > 0:
        raise ValueError('The series argument must not contain missing data')
    if not isinstance(n, int):
        raise TypeError('The n argument must be an integer')
    minimum_n: int = 3
    if n < minimum_n:
        raise ValueError(f'The n argument must be >= {minimum_n}')
    if n % 2 == 0:
        raise ValueError('The n argument must be odd')
    if len(series) < n:
        raise ValueError('The series argument should be a Series longer than n')

    # - Build the Henderson moving average.
    #   We start with the middle section, using a
    #   .rolling.apply(), which is vectorised and
    #   has the side effect of creating the return
    #   vehicle (henderson). We then move to the tails.

    # 0, preliminaries
    sym_weights = hma_symmetric_weights(n)
    mid_point: int = int((n - 1) // 2)

    # 1, the middle:
    henderson = (
        series
        .rolling(n, min_periods=n, center=True)
        .apply( lambda x: x.mul(sym_weights).sum() )
    )

    # 2, the tails:
    for i in range(1, mid_point + 1):
        asym_wts = hma_asymmetric_weights(mid_point + i, sym_weights)
        henderson.iloc[i - 1] = (series.iloc[:(i + mid_point)] * asym_wts[::-1]).sum()
        henderson.iloc[-i] = (series.iloc[(-mid_point - i):] * asym_wts).sum()

    return henderson



### --- test code
#--------------
# Check against Table 1 in B Quenneville and B Lefrancois (2001),
# "Implicit Forecasts in Musgrave Asymmetric Averages",
# Proceedings of the Annual Meeting of the American Statistical Association,
# August 5-9, 2001.
#--------------

if __name__ == "__main__":
    print('Testing ...\n')

    # --- check symmetric weights
    N = 9
    weights = hma_symmetric_weights(N)
    print(f'{N} symmetric weights: {weights}')
    print(weights.sum(), ' <-- Sum of weights should be one\n')

    # --- check asymmetric weights
    M = 7
    asym_weights = hma_asymmetric_weights(M, weights)
    print(f'{M} asymmetric weights: {asym_weights}')
    print(asym_weights.sum(), ' <-- Sum of weights should be one\n')

    # --- check cache
    cache_weights = hma_symmetric_weights(N)
    print(cache_weights, ' <-- Should be the same as symmetric weights above\n')
    print(_cache.keys(), f' <-- should contain two keys: {N} and ({N}, {M})\n')

    # --- check it altogether
    LENGTH = 30
    print('\nSimple case: all ones; should return all ones')
    print(hma(pd.Series(np.repeat(1, LENGTH)), N))

    print('\nSimple case: arithmetic increasing; '
          '\n Should be the same as the series index in the centre. '
          '\n But a little tapered at each end.')
    print(hma(pd.Series(range(LENGTH)), N))

    print('\nComplex case: stochastic increasing; '
          '\n Series should be numbers broadly around the series index')
    print(hma(pd.Series(range(LENGTH))
              + pd.Series(np.random.randn(LENGTH)), N))
