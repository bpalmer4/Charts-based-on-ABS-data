"""abs_summary_plot.py
======================
Produce a summary plot of the most recent key data items in their
historical context. The data is normalised to z-scores and scaled."""


# imports
import matplotlib.pyplot as plt
import pandas as pd
from typing import Any, Sequence
from pandas import DataFrame
from abs_data_capture import AbsDict, metacol
from plotting import finalise_plot


# private
def _get_summary_data(
    to_get: dict[str, Sequence],
    abs_data: AbsDict,
    md: pd.DataFrame,
    verbose: bool = False,
) -> pd.DataFrame:
    """Get required data items. If period is specified,
    calculate the percentage change over that period.
    Return a DataFrame with the data."""

    data = pd.DataFrame()
    for label, [code, period] in to_get.items():
        selected = md[md[metacol.id] == code].iloc[0]
        table_desc = selected[metacol.tdesc]
        table = selected[metacol.table]
        did = selected[metacol.did]
        stype = selected[metacol.stype]
        if verbose:
            print(code, table, table_desc, did, stype)
        series = abs_data[table][code]
        if period:
            series = series.pct_change(periods=period, fill_method=None) * 100
        data[label] = series
    return data


# public
def plot_summary(
    to_get: dict[str, list],  # dictionary of daya items to get
    abs_data: AbsDict,  # abs data tables
    md: pd.DataFrame,  # meta data
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
            "bbox_to_anchor": (0.5, -0.15),
            "ncol": 4,
        },
    )
    kwargs["x0"] = kwargs.get("x0", True)

    # get the data, calculate z-scores and scaled scores based on the start period
    summary = _get_summary_data(to_get=to_get, abs_data=abs_data, md=md, verbose=verbose)
    original = summary[start:]
    z_scores = ((original - original.mean()) / original.std())
    z_scaled = (
        (((z_scores - z_scores.min()) 
        / (z_scores.max() - z_scores.min())) - 0.5) * 2
    )

    # display a summary of the data
    q = (round((1 - middle) / 2, 2), round(1 - (1 - middle) / 2, 2))
    if verbose:
        display(pd,DataFrame({
            "count": original.count(),
            "mean": original.mean(),
            "median": original.median(),
            "min shaded": original.quantile(q=q[0]),
            "max shaded": original.quantile(q=q[1]),
            "final z-scores": z_scores.iloc[-1],
            "final scaled": z_scaled.iloc[-1],
        }))

    # plot as required by the plot_types argument
    kwargs["title"] = kwargs.get("title", f"Summary at {summary.index[-1]}")
    for plot_type in plot_types:
        if "xlabel" in kwargs:
            print(f"Overriding xlabel: {kwargs['xlabel']}") 
        if plot_type == "zscores":
            adjusted = z_scores
            kwargs["xlabel"] = f"Z-scores for prints since {start}"
        elif plot_type == "scaled":
            adjusted = z_scaled
            kwargs["xlabel"] = f"Scaled z-scores since {start}"
        else:
            print(f"Unknown plot type {plot_type}")
            continue

        # horizontal bar plot the middle of the data
        lo_hi = adjusted.quantile(q=q).T  # get the middle section of data
        scope = max(adjusted.iloc[-1].abs().max(), lo_hi.abs().max().max())
        ends = 1.2 if plot_type == "scaled" else scope + 0.1
        kwargs["xlim"] = (-ends, ends)
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
        for col in original.columns:
            ax.text(
                adjusted.at[row, col],
                col,
                f"{original.at[row, col]:.1f}",
                ha="center",
                va="center",
                size=f_size,
            )

        # label extremes
        if plot_type == "scaled":
            ax.scatter(
                adjusted.median(),
                adjusted.columns,
                color="darkorchid",
                marker="x",
                s=5,
                label="Median",
            )
            for col in original.columns:
                ax.text(
                    -ends,
                    col,
                    f" {original[col].min():.1f}",
                    ha="left",
                    va="center",
                    size=f_size,
                )
                ax.text(
                    ends,
                    col,
                    f"{original[col].max():.1f} ",
                    ha="right",
                    va="center",
                    size=f_size,
                )

        # finalise
        kwargs["pre_tag"] = plot_type
        ax.tick_params(axis="y", labelsize=10)
        finalise_plot(ax, **kwargs)

        # pre-pare for next loop
        kwargs.pop("xlabel", None)
