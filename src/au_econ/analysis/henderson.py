"""Henderson moving average, with Doherty's asymmetric end weights.

Symmetric weights: ABS (2003), "A Guide to Interpreting Time Series", page 41.
Asymmetric end weights: M. Doherty (2001), "The Surrogate Henderson Filters in X-11",
Aust. NZ J. Stat. 43(4), formula (1) on page 903.
"""

from functools import cache
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

MINIMUM_TERMS = 3
# Doherty's I/C ratio, by filter length
IC_THRESHOLD_LOW, IC_THRESHOLD_HIGH = 13, 15
IC_DEFAULT, IC_LOW, IC_HIGH = 1.0, 3.5, 4.5


@cache
def symmetric_weights(n: int) -> np.ndarray:
    """Return the n symmetric Henderson weights (n odd), indexed 0 to n-1."""
    m = (n - 1) // 2
    m1, m2, m3 = (m + 1) ** 2, (m + 2) ** 2, (m + 3) ** 2
    denominator = float(8 * (m + 2) * (m2 - 1) * (4 * m2 - 1) * (4 * m2 - 9) * (4 * m2 - 25))
    weights = np.repeat(np.nan, n)
    for j in range(m + 1):
        j2 = j * j
        weight = (315 * (m1 - j2) * (m2 - j2) * (m3 - j2) * (3 * m2 - 11 * j2 - 16)) / denominator
        weights[m + j] = weight
        if j > 0:
            weights[m - j] = weight
    weights.flags.writeable = False
    return weights


@cache
def asymmetric_weights(m: int, n: int) -> np.ndarray:
    """Return the m asymmetric end weights for an n-term filter (m < n), indexed 0 to m-1."""
    sym = symmetric_weights(n)
    sum_residual = sym[range(m, n)].sum() / float(m)
    sum_end = 0.0
    for i in range(m + 1, n + 1):
        sum_end += (float(i) - ((m + 1.0) / 2.0)) * sym[i - 1]

    ic = IC_DEFAULT
    if IC_THRESHOLD_LOW <= n < IC_THRESHOLD_HIGH:
        ic = IC_LOW
    elif n >= IC_THRESHOLD_HIGH:
        ic = IC_HIGH
    b2s2 = (4.0 / np.pi) / (ic * ic)

    denominator = 1.0 + ((m * (m - 1.0) * (m + 1.0) / 12.0) * b2s2)
    weights = np.repeat(np.nan, m)
    for r in range(m):  # r runs 0 to m-1; the formula counts 1 to m
        numerator = ((r + 1.0) - (m + 1.0) / 2.0) * b2s2
        weights[r] = sym[r] + sum_residual + (numerator / denominator) * sum_end
    weights.flags.writeable = False
    return weights


def hma(series: pd.Series, n: int) -> pd.Series:
    """Return the n-term Henderson moving average of an ordered series with no missing data."""
    if series.isna().any():
        raise ValueError("The series must not contain missing data")
    if n < MINIMUM_TERMS or n % 2 == 0:
        raise ValueError(f"n must be odd and at least {MINIMUM_TERMS}, got {n}")
    if len(series) < n:
        raise ValueError(f"The series (length {len(series)}) is shorter than n ({n})")

    sym = symmetric_weights(n)
    mid_point = (n - 1) // 2

    # the middle, from a centred rolling window; then the tails, from the end weights
    henderson = series.rolling(n, min_periods=n, center=True).apply(lambda x: x.mul(sym).sum())
    for i in range(1, mid_point + 1):
        end_weights = asymmetric_weights(mid_point + i, n)
        henderson.iloc[i - 1] = (series.iloc[: (i + mid_point)] * end_weights[::-1]).sum()
        henderson.iloc[-i] = (series.iloc[(-mid_point - i) :] * end_weights).sum()
    return henderson
