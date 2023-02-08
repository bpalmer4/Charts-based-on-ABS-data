## henderson.py 
## Use: from henderson import HMA
## calculate a Henderson moving average

import pandas as pd
import numpy as np
from typing import Dict, Union, Tuple

_cache:Dict[Union[int, Tuple[int, int]], np.ndarray] = {}

def hmaSymmetricWeights(n:int) -> np.ndarray:
    """ derive an n-term array of symmetric 'Henderson Moving Average' weights
        formula from ABS (2003), 'A Guide to Interpreting Time Series', page 41.
        returns a numpy array of symmetric Henderson weights indexed from 0 to n-1"""

    if n in _cache:
        return _cache[n]
    
    # calculate the constant denominator and terms
    m = int((n - 1) // 2) # the mid point - n must be odd
    m1 = (m + 1) * (m + 1)
    m2 = (m + 2) * (m + 2)
    d = float(8 * (m + 2) * (m2 - 1) * (4 * m2 -1) * (4 * m2 - 9) * ( 4 * m2 - 25))
    m3 = (m + 3) * (m + 3)

    # calculate the weights
    weights = np.repeat(np.nan, n) # indexed from 0 to n-1
    for j in range(m + 1):
        j2 = j * j
        v = (315 * (m1 - j2) * (m2 - j2) * (m3 - j2) * (3 * m2 - 11 * j2 - 16)) / d
        weights[(m + j)] = v
        if j > 0:
            weights[(m - j)] = v

    weights.flags.writeable = False # make quasi-immutable
    _cache[n] = weights
    return weights


def hmaAsymmetricWeights(m:int, w: np.ndarray) -> np.ndarray:
    """Calculate the asymmetric end-weights

        w --> an array of symmetrical henderson weights (from above function)
        m --> the number of asymmetric weights sought; where m < len(w);

        returns a numpy array of asymmetrical weights, indexed from 0 to m-1;

        formula from Mike Doherty (2001), 'The Surrogate Henderson Filters in X-11',
        Aust, NZ J of Stat. 43(4), 2001, pp901-999; see formula (1) on page 903"""

    # - get n from the weights array
    n = len(w) # the number of symmetric weights
    cache_key = (n, m)
    if cache_key in _cache:
        return _cache[cache_key]
    
    # --- let's build up Doherty's formula (1) from the top of page 903
    # - the second chunk of the formula
    sumResidual = w[range(m, n)].sum() / float(m)

    # - the last chunk of the formula
    sumEnd = 0.0
    for i in range(m + 1, n + 1):
        sumEnd += (float(i) - ((m + 1.0) / 2.0)) * w[i - 1] # w indexed from 0 to n-1

    # - the beta squared / sigma squared - formula at the bottom of page 904
    ic = 1.0
    if n >= 13 and n < 15:
        ic = 3.5
    elif n >= 15:
        ic = 4.5
    b2s2 = (4.0 / np.pi) / (ic * ic)

    # - the gnarly bit in the middle of the formula
    denominator = 1.0 + ((m * (m - 1.0) * (m + 1.0) / 12.0 ) * b2s2)
    u = np.repeat(np.nan, m) # return series - created empty
    for r in range(m): # r ranges 0 to m-1; but the formulae assumes 1 to m
        numerator = ((r + 1.0) - (m + 1.0) / 2.0) * b2s2
        # - finally putting it all together
        u[r] = w[r] + sumResidual + (numerator / denominator) * sumEnd

    u.flags.writeable = False # make quasi-immutable
    _cache[cache_key] = u
    return (u)


def HMA(s: pd.Series, n: int) -> pd.Series:
    """ Calculate an n-term Henderson Moving Average for the pandas Series s
        Note: s is ordered, contiguous and without missing data."""

    # - some simple sanity checks
    if not isinstance(s, pd.core.series.Series):
        raise TypeError('The s argument must be a pandas Series')
    if s.isna().sum() > 0:
        raise ValueError('The s argument must not contain missing data')
    if not isinstance(n, int):
        raise TypeError('The n argument must be an integer')
    MIN_N = 3
    if n < MIN_N:
        raise ValueError(f'The n argument must be >= {MIN_N}')
    if n % 2 == 0:
        raise ValueError('The n argument must be odd')
    if len(s) < n:
        raise ValueError('The s argument should be a Series longer than n')

    # preliminaries
    w = hmaSymmetricWeights(n)
    m = int((n - 1) // 2)
    l = len(s)

    # - and now move over the length of the series ...
    r = (
        # apply symmetric weights to the middle of the series
        s
        .rolling(n, min_periods=n, center=True)
        .apply( lambda x: x.mul(w).sum() )
    )
    for i in range(m): # at the beginning
        u = hmaAsymmetricWeights(m + i + 1, w)[::-1] # reverse asymmetric to the left
        r.iloc[i] = (s.iloc[0:( i + m + 1)] * u).sum()
    for i in range(l-m, l): # at the end
        u = hmaAsymmetricWeights(m + l - i, w)
        r.iloc[i] = (s.iloc[(i - m):l] * u).sum()
    
    return r


### - test code
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
    w = hmaSymmetricWeights(N)
    print(f'{N} symmetric weights: ', w)
    print(w.sum(), ' <-- Sum of weights should be one\n')
                      
    # --- check asymmetric weights
    M = 7
    u = hmaAsymmetricWeights(M, w)
    print(f'{M} asymmetric weights: ', u)
    print(u.sum(), ' <-- Sum of weights should be one\n')
                      
    # --- check cache
    x = hmaSymmetricWeights(N)
    print(x, ' <-- Should be the same as symmetric weights above\n')
    print(_cache.keys(), f' <-- should contain two keys: {N} and ({N}, {M})\n')
                      
    # --- check it altogether
    LENGTH = 30
    print('\nSimple case: all ones; should return all ones')
    print(HMA(pd.Series(np.repeat(1, LENGTH)), N))

    print('\nSimple case: arithmetic increasing; '
          '\n Should be the same as the series index in the centre. '
          '\n But a little tapered at each end.')
    print(HMA(pd.Series(range(LENGTH)), N))
    
    print('\nComplex case: stochastic increasing; '
          '\n Series should be numbers broadly around the series index')
    print(HMA(pd.Series(range(LENGTH))
              + pd.Series(np.random.randn(LENGTH)), N))