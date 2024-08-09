"""abs_summary_plot.py
======================
Produce a summary plot of the most recent (specified) key data items in
their historical context. The data is normalised to z-scores and scaled."""

# system imports
from typing import Any
from functools import cache

# analytic third-party imports
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from IPython.display import display

# custom imports
from readabs import metacol as mc
from plotting import finalise_plot


# private
def _get_summary_data(
    to_get: dict[str, tuple[str, int]],
    abs_data: dict[str, DataFrame],
    md: DataFrame,
    verbose: bool = False,
) -> DataFrame:
    """Get required data items. If period is specified,
    calculate the percentage change over that period.
    Return a DataFrame with the data."""

    data = DataFrame()
    for label, [code, period] in to_get.items():
        selected = md[md[mc.id] == code].iloc[0]
        table_desc = selected[mc.tdesc]
        table = selected[mc.table]
        did = selected[mc.did]
        stype = selected[mc.stype]
        if verbose:
            print(code, table, table_desc, did, stype)
        series = abs_data[table][code]
        if period:
            series = (
                series.pct_change(periods=period, fill_method=None) * 100  # type: ignore[arg-type]
            )
        data[label] = series
    return data


@cache
def _calc_quantiles(middle: float) -> np.ndarray:
    """Calculate the quantiles for the middle of the data."""
    return np.array([(1 - middle) / 2.0, 1 - (1 - middle) / 2.0])


def _calculate_z(
    original: DataFrame,  # only contains the data points of interest
    middle: float,  # middle proportion of data to highlight (eg. 0.8)
    verbose: bool = False,  # print the summary data
) -> tuple[DataFrame, DataFrame]:
    """Calculate z-scores, scaled z-scores and middle quantiles.
    Return z_scores, z_scaled, q (which are the quantiles for the
    start/end of the middle proportion of data to highlight)."""

    # calculate z-scores, scaled scores and middle quantiles
    z_scores = (original - original.mean()) / original.std()
    z_scaled = (
        # scale z-scores between -1 and +1
        (((z_scores - z_scores.min()) / (z_scores.max() - z_scores.min())) - 0.5)
        * 2
    )
    q_middle = _calc_quantiles(middle)

    if verbose:
        frame = DataFrame(
            {
                "count": original.count(),
                "mean": original.mean(),
                "median": original.median(),
                "min shaded": original.quantile(q=q_middle[0]),
                "max shaded": original.quantile(q=q_middle[1]),
                "z-scores": z_scores.iloc[-1],
                "scaled": z_scaled.iloc[-1],
            }
        )
        display(frame)

    return z_scores, z_scaled


def _horizontal_bar_plot(
    original: DataFrame,
    adjusted: DataFrame,
    middle: float,
    plot_type: str,
    kwargs: dict[str, Any],
) -> plt.Axes:
    """Plot horizontal bars for the middle of the data."""

    # horizontal bar plot the middle of the data
    q = _calc_quantiles(middle)
    lo_hi: DataFrame = adjusted.quantile(q=q).T  # get the middle section of data
    span = 1.15
    space = 0.15
    low = min(adjusted.iloc[-1].min(), lo_hi.min().min(), -span) - space
    high = max(adjusted.iloc[-1].max(), lo_hi.max().max(), span) + space
    kwargs["xlim"] = (low, high)
    _fig, ax = plt.subplots()
    ax.barh(
        y=lo_hi.index,
        width=lo_hi[q[1]] - lo_hi[q[0]],
        left=lo_hi[q[0]],
        color="#bbbbbb",
        label=f"Middle {middle*100:0.0f}% of prints",
    )

    # plot the latest data
    ax.scatter(adjusted.iloc[-1], adjusted.columns, color="darkorange")
    f_size = 10
    row = adjusted.index[-1]
    for col_num, col_name in enumerate(original.columns):
        ax.text(
            x=adjusted.at[row, col_name],
            y=col_num,
            s=f"{original.at[row, col_name]:.1f}",
            ha="center",
            va="center",
            size=f_size,
        )

    # label extremes in the scaled plots
    if plot_type == "scaled":
        ax.axvline(-1, color="#555555", linewidth=0.5, linestyle="--")
        ax.axvline(1, color="#555555", linewidth=0.5, linestyle="--")
        ax.scatter(
            adjusted.median(),
            adjusted.columns,
            color="darkorchid",
            marker="x",
            s=5,
            label="Median",
        )
        for col_num, col_name in enumerate(original.columns):
            ax.text(
                low,
                col_num,
                f" {original[col_name].min():.1f}",
                ha="left",
                va="center",
                size=f_size,
            )
            ax.text(
                high,
                col_num,
                f"{original[col_name].max():.1f} ",
                ha="right",
                va="center",
                size=f_size,
            )

    return ax


# public
def plot_summary(
    to_get: dict[
        str, tuple[str, int]
    ],  # {label: (series_id: str, n_periods_growth: int), ...}
    abs_data: dict[str, DataFrame],  # abs data tables
    md: DataFrame,  # meta data
    start: str,  # starting period for z-score calculation
    **kwargs: Any,
) -> None:
    """Calculate z-scores and scaled z-scores and plot."""

    # optional arguments
    verbose = kwargs.pop("verbose", False)
    middle = kwargs.pop("middle", 0.8)
    plot_types = kwargs.pop("plot_types", ["zscores", "scaled"])

    kwargs["show"] = kwargs.get("show", False)
    kwargs["ylabel"] = kwargs.get("ylabel", None)
    kwargs["legend"] = kwargs.get(
        "legend",
        {
            # put the legend below the x-axis label
            "loc": "upper center",
            "fontsize": "xx-small",
            "bbox_to_anchor": (0.5, -0.125),
            "ncol": 4,
        },
    )
    kwargs["x0"] = kwargs.get("x0", True)

    # get the data, calculate z-scores and scaled scores based on the start period
    original = _get_summary_data(
        to_get=to_get, abs_data=abs_data, md=md, verbose=verbose
    ).loc[lambda x: x.index >= start]
    z_scores, z_scaled = _calculate_z(original, middle, verbose=verbose)

    # plot as required by the plot_types argument
    kwargs["title"] = kwargs.get("title", f"Summary at {original.index[-1]}")
    for plot_type in plot_types:
        if "xlabel" in kwargs:
            print(f"Overriding xlabel: {kwargs['xlabel']}")
        if "x0" in kwargs:
            print(f"Overriding x0: {kwargs['x0']}")
        if plot_type == "zscores":
            adjusted = z_scores
            kwargs["xlabel"] = f"Z-scores for prints since {start}"
            kwargs["x0"] = True
        elif plot_type == "scaled":
            adjusted = z_scaled
            kwargs["xlabel"] = f"-1 to 1 scaled z-scores since {start}"
        else:
            print(f"Unknown plot type {plot_type}")
            continue

        ax = _horizontal_bar_plot(original, adjusted, middle, plot_type, kwargs)

        # finalise
        kwargs["pre_tag"] = plot_type
        ax.tick_params(axis="y", labelsize=10)
        finalise_plot(ax, **kwargs)

        # pre-pare for next loop
        kwargs.pop("xlabel", None)
        kwargs.pop("x0", None)
