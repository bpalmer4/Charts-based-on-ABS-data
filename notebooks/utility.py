"""utility functions. Mostly for pandas DataFrames."""
from typing import TypeVar, cast
from pandas import Series, DataFrame, PeriodIndex, DatetimeIndex

# - define a useful typevar for working with both Series and DataFrames
_DataT = TypeVar("_DataT", Series, DataFrame)


def qtly_to_monthly(data: _DataT, interpolate: bool = True) -> _DataT:
    """Convert a pandas timeseries with a Quarterly PeriodIndex to an
    timeseries with a Monthly PeriodIndex.
    Arguments:
    ==========
    data - either a pandas Series or DataFrame - assumes the index is unique.
    interpolate - whether to interpolate the missing monthly data, or
                  delete the missing row.
    Notes:
    ======
    Necessitated by Pandas 2.2, which removed .resample()
    from pandas objects with a PeriodIndex."""

    # sanity checks
    assert isinstance(data.index, PeriodIndex)
    assert data.index.freqstr[0] == "Q"
    assert data.index.is_unique

    def set_axis_monthly_periods(x: _DataT) -> _DataT:
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

    data = (
        # interpolate or remove the added months.
        data.interpolate(limit_area="inside", limit=2)
        if interpolate
        else data.dropna()
    )

    return data
