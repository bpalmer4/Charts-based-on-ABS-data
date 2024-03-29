from typing import TypeVar, Optional, cast
from pandas import Series, DataFrame, PeriodIndex, DatetimeIndex

# - define a useful typevar for working with both Series and DataFrames
DataT = TypeVar("DataT", Series, DataFrame)


def percent_change(data: DataT, periods: int) -> DataT:
    """Calculate an n-periods percentage change."""

    return (data / data.shift(periods) - 1) * 100


def annualise_rates(data: DataT, periods: int|float = 12) -> DataT:
    """Annualise a growth rate for a period.
    Note: returns a percentage (and not a rate)!"""

    return (((1 + data) ** periods) - 1) * 100


def annualise_percentages(data: DataT, periods: int|float = 12) -> DataT:
    """Annualise a growth rate (expressed as a percentage) for a period."""

    rates = data / 100.0
    return annualise_rates(rates, periods)


def qtly_to_monthly(
    data: DataT,
    interpolate: bool = True,
    limit: Optional[int] = 2,  # only used if interpolate is True
    dropna: bool = True,
) -> DataT:
    """Convert a pandas timeseries with a Quarterly PeriodIndex to an
    timeseries with a Monthly PeriodIndex.
    Arguments:
    ==========
    data - either a pandas Series or DataFrame - assumes the index is unique.
    interpolate - whether to interpolate the missing monthly data.
    dropna - whether to drop NA data
    Notes:
    ======
    Necessitated by Pandas 2.2, which removed .resample()
    from pandas objects with a PeriodIndex."""

    # sanity checks
    assert isinstance(data.index, PeriodIndex)
    assert data.index.freqstr[0] == "Q"
    assert data.index.is_unique
    assert data.index.is_monotonic_increasing

    def set_axis_monthly_periods(x: DataT) -> DataT:
        """Convert a DatetimeIndex to a Monthly PeriodIndex."""

        return x.set_axis(
            labels=cast(DatetimeIndex, x.index).to_period(freq="M"), axis="index"
        )

    # do the heavy lifting
    data = (
        data.set_axis(
            labels=data.index.to_timestamp(how="end"), axis="index", copy=True
        )
        .resample(rule="ME")  # adds in every missing month
        .first(min_count=1)  # generates nans for new months
        # assumes only one value per quarter (ie. unique index)
        .pipe(set_axis_monthly_periods)
    )

    if interpolate:
        data = data.interpolate(limit_area="inside", limit=limit)
    if dropna:
        data = data.dropna()

    return data


def monthly_to_qtly(data: DataT, q_ending="DEC") -> DataT:
    """Convert monthly PeriodIndex data to quarterly PeriodIndex data."""

    return (
        data.pipe(
            lambda x: x.set_axis(
                labels=cast(PeriodIndex, x.index).to_timestamp(how="end"),
                axis="index",
                copy=True,
            )
        )
        .resample(rule="QE")
        .mean()
        .pipe(
            lambda x: x.set_axis(
                labels=cast(DatetimeIndex, x.index).to_period(freq=f"Q-{q_ending}"),
                axis="index",
            )
        )
    )
