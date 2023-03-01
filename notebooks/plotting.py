"""A set of functions for plotting with matplotlib."""

# --- imports
# system imports
import re
from pathlib import Path
from operator import mul, truediv

# data science imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
import statsmodels.formula.api as smf


# --- clear_chart_dir()


def clear_chart_dir(chart_dir):
    """Remove all .png files from the chart_dir."""

    for fs_object in Path(chart_dir).glob("*.png"):
        if fs_object.is_file():
            fs_object.unlink()


# --- finalise_plot()

# - constants - default settings for finalise_plot()
DEFAULT_FILE_TYPE = "png"
DEFAULT_FIG_SIZE = (9, 4.5)
DEFAULT_DPI = 125
DEFAULT_CHART_DIR = "."

# global chart_dir - modified by set_chart_dir() below
_chart_dir: str | None = DEFAULT_CHART_DIR

# filename limitations - used to map the plot title to a filename
_remove = re.compile(r"[^0-9A-Za-z]")  # make sensible file names
_reduce = re.compile(r"[-]+")  # eliminate multiple hyphens

# map of the acceptable kwargs for finalise_plot()
_ACCEPTABLE_KWARGS = frozenset(
    (
        "title",
        "xlabel",
        "ylabel",
        "pre_tag",
        "tag",
        "chart_dir",
        "file_type",
        "lfooter",
        "rfooter",
        "figsize",
        "show",
        "concise_dates",
        "zero_y",
        "dont_save",
        "dont_close",
        "dpi",
        "legend",
    )
)


# - private utility functions for finalise_plot()


def _check_kwargs(**kwargs):
    """Report unrecognised keyword arguments."""

    for k in kwargs:
        if k not in _ACCEPTABLE_KWARGS:
            print(f"Warning: {k} was an unrecognised keyword argument")


def _apply_kwargs(axes, **kwargs):
    """Apply settings found in kwargs."""

    fig = axes.figure

    # annotate plot
    settings = ("title", "xlabel", "ylabel")
    for setting in settings:
        value = kwargs.get(setting, None)
        axes.set(**{setting: value})

    if "legend" in kwargs:
        axes.legend(**kwargs["legend"])

    if "rfooter" in kwargs:
        fig.text(
            0.99,
            0.001,
            kwargs["rfooter"],
            ha="right",
            va="bottom",
            fontsize=9,
            fontstyle="italic",
            color="#999999",
        )

    if "lfooter" in kwargs:
        fig.text(
            0.01,
            0.001,
            kwargs["lfooter"],
            ha="left",
            va="bottom",
            fontsize=9,
            fontstyle="italic",
            color="#999999",
        )

    if "figsize" in kwargs:
        fig.set_size_inches(*kwargs["figsize"])
    else:
        fig.set_size_inches(*DEFAULT_FIG_SIZE)

    if "concise_dates" in kwargs and kwargs["concise_dates"]:
        locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
        formatter = mdates.ConciseDateFormatter(locator)
        axes.xaxis.set_major_locator(locator)
        axes.xaxis.set_major_formatter(formatter)
        for label in axes.get_xticklabels(which="major"):
            label.set(rotation=0, horizontalalignment="center")

    if "zero_y" in kwargs and kwargs["zero_y"]:
        bottom, top = axes.get_ylim()
        adj = (top - bottom) * 0.02
        if bottom > -adj:
            axes.set_ylim(bottom=-adj)
        if top < adj:
            axes.set_ylim(top=adj)


def _save_to_file(fig, **kwargs) -> None:
    """Save the figure to file."""

    saving = True if "dont_save" not in kwargs else not kwargs["dont_save"]
    if saving:
        chart_dir = None if "chart_dir" not in kwargs else kwargs["chart_dir"]
        if chart_dir is None:
            chart_dir = _chart_dir
            if chart_dir is None:
                chart_dir = ""

        title = "" if "title" not in kwargs else kwargs["title"]
        pre_tag = "" if "pre_tag" not in kwargs else kwargs["pre_tag"]
        tag = "" if "tag" not in kwargs else kwargs["tag"]
        file_title = re.sub(_remove, "-", title).lower()
        file_title = re.sub(_reduce, "-", file_title)
        file_type = (
            DEFAULT_FILE_TYPE if "file_type" not in kwargs else kwargs["file_type"]
        )
        dpi = DEFAULT_DPI if "dpi" not in kwargs else kwargs["dpi"]
        fig.savefig(f"{chart_dir}/{pre_tag}{file_title}-{tag}.{file_type}", dpi=dpi)


# - public functions for finalise_plot()


def set_chart_dir(chart_dir: str | None) -> None:
    """A function to set a global chart directory for finalise_plot(),
    so that it does not need to be included as an argument in each
    call to finalise_plot()."""

    global _chart_dir
    _chart_dir = chart_dir


def finalise_plot(axes, **kwargs):
    """A function to finalise and save plots to the file system. The filename
    for the saved plot is constructed from the chart_dir, the plot's title,
    any specified tag text, and the file_type for the plot.
     Arguments:
       - axes - matplotlib axes object - required
      kwargs
       - title - string - plot title, also used to save the file
       - xlabel - string - label for the x-axis
       - ylabel - string - label for the y-axis
       - pre_tag - string - text before the title in file name
       - tag - string - text after the title - used in file name
         to make similar plots have unique file names
       - chart_dir - string - location of the chartr directory
       - file_type - string - specify a file type - eg. 'png' or 'svg'
       - lfooter - string - text to display on bottom left of plot
       - rfooter - string - text to display of bottom right of plot
       - figsize - tuple - figure size in inches - eg. (8, 4)
       - show - Boolean - whether to show the plot or not
       - concise_dates - bool - use the matplotlib concise dates formatter
       - zero_y - bool - ensure y=0 is included in the plot.
       - dont_save - bool - dont save the plot to the file system
       - dont_close - bool - dont close the plot
       - dpi - int - dots per inch for the saved chart
       - legend - dict - arguments to pass to axes.legend()
     Returns:
       - None
    """

    _check_kwargs(**kwargs)

    # margins
    axes.use_sticky_margins = False
    axes.margins(0.02)
    axes.autoscale(tight=False)  # This is problematic ...

    _apply_kwargs(axes, **kwargs)

    # tight layout
    fig = axes.figure
    fig.tight_layout(pad=1.1)

    _save_to_file(fig, **kwargs)

    # show the plot in Jupyter Lab
    _ = plt.show() if "show" in kwargs and kwargs["show"] else None

    # And close
    closing = True if "dont_close" not in kwargs else not kwargs["dont_close"]
    if closing:
        plt.close()


# --- plot_series_highlighted()


def plot_series_highlighted(series: pd.Series, **kwargs) -> plt.Axes:
    """Plot a series of percentage rates, highlighting the increasing runs.
    Arguments
     - series - ordered pandas Series of percentages, with PeriodIndex
     - threshold - float - used to ignore micro noise near zero
       (for example, threshhold=0.001)
     - round - int - rounding for highlight text
    Return
     - matplotlib Axes object"""

    # default arguments - in **kwargs
    threshold = 0.001 if "threshold" not in kwargs else kwargs["threshold"]
    round_ = 2 if "round" not in kwargs else kwargs["round"]  # int

    # identify the runs
    diffed = series.diff()
    change_points = pd.concat(
        [diffed[diffed.gt(threshold)], diffed[diffed.lt(-threshold)]]
    ).sort_index()
    if series.index[0] not in change_points.index:
        starting_point = pd.Series([0], index=[series.index[0]])
        change_points = pd.concat([change_points, starting_point]).sort_index()
    rising = change_points > 0
    cycles = (rising & ~rising.shift().astype(bool)).cumsum()
    rising_stretches = cycles[rising]

    # chart the series
    axes = series.plot(drawstyle="steps-post", lw=2, c="#dd0000")

    # highlight the runs
    for k in range(1, rising_stretches.max() + 1):
        stretch = rising_stretches[rising_stretches == k]
        axes.axvspan(stretch.index.min(), stretch.index.max(), color="gold", zorder=-1)
        if series[stretch.index].min() < (series.max() + series.min()) / 2:
            y_pos, vert_align = series.max(), "top"
        else:
            y_pos, vert_align = series.min(), "bottom"
        text = axes.text(
            x=stretch.index.min(),
            y=y_pos,
            s=(change_points[stretch.index].sum().round(round_).astype(str) + " pp"),
            va=vert_align,
            ha="left",
            rotation=90,
        )
        text.set_path_effects([pe.withStroke(linewidth=5, foreground="w")])

    return axes


# --- plot_covid_recovery()


def get_projection(original: pd.Series, to_period: pd.Period) -> pd.Series:
    """Projection based on data from the start of a series
    to the to_period (inclusive). Returns projection over the whole
    period of the original series."""

    y_regress = original[original.index <= to_period]
    x_regress = np.arange(len(y_regress))
    regress_data = pd.DataFrame(
        {
            "y": y_regress.values,
            "x": x_regress,
        }
    )
    model = smf.ols(formula="y ~ x", data=regress_data).fit()
    # print(model.summary())
    # print(model.params)

    x_complete = np.arange(len(original))
    projection = pd.Series(
        x_complete * model.params["x"] + model.params["Intercept"], index=original.index
    )

    return projection


def plot_covid_recovery(series: pd.Series, verbose=False, **kwargs) -> None:
    """Plots a series with a PeriodIndex.
    Arguments
     - series to be plotted
     - **kwargs - same as for finalise_plot()."""

    # sanity checks
    if not isinstance(series, pd.Series):
        raise TypeError("The series argument must be a pandas Series")
    if not isinstance(series.index, pd.PeriodIndex):
        raise TypeError("The series must have a pandas PeriodIndex")
    if not series.index.freqstr[:1] in ("Q", "M", "D"):
        raise ValueError("The series index must have a D, M or Q freq")

    # plot COVID counterfactural
    freq = series.index.freqstr[0]
    if "start_r" in kwargs and "end_r" in kwargs:
        # set by argument - note must set both start_r and end_r
        if verbose:
            print(
                f'Using special start/end dates: {kwargs["start_r"]} - {kwargs["end_r"]}'
            )
        start_regression = pd.Period(kwargs["start_r"], freq=freq)
        end_regression = pd.Period(kwargs["end_r"], freq=freq)
        del kwargs["start_r"]
        del kwargs["end_r"]
    else:
        # set programatically
        if freq in ["M", "D"]:
            # assume last unaffected month is January 2020
            start_regression = pd.Period("2017-01-31", freq=freq)
            end_regression = pd.Period("2020-01-31", freq=freq)
        else:
            # assume last unaffected quarter ends in December 2019
            start_regression = pd.Period("2016-12-31", freq=freq)
            end_regression = pd.Period("2019-12-31", freq=freq)
    recent = series[series.index >= start_regression]
    projection = get_projection(recent, end_regression)

    axes = recent.plot(lw=2, c="dodgerblue", label=series.name)
    projection.plot(
        lw=2,
        c="darkorange",
        ls="--",
        label="Pre-COVID projection",
        ax=axes,
    )
    axes.legend(loc="best")

    # augment left-footer
    kwargs["lfooter"] = "" if "lfooter" not in kwargs else kwargs["lfooter"]
    kwargs[
        "lfooter"
    ] += f"Projection on data from {start_regression} to {end_regression}. "

    finalise_plot(axes, **kwargs)


# --- plot_growth(), plot_growth_finalise() and calc_growth()


def plot_growth(
    annual: pd.Series,
    periodic: pd.Series,
    from_: pd.Period | None = None,
) -> None | plt.Axes:
    """Plot a bar and line percentage growth chart."""

    # sanity checks
    for series in (annual, periodic):
        assert isinstance(
            series, pd.Series
        ), "initial arguments should be pandas Series"
        assert isinstance(
            series.index, pd.PeriodIndex
        ), "Series index should be a PeriodIndex"

    # put our two series into a datadrame
    frame = pd.DataFrame(
        [annual.copy(), periodic.copy()], index=["Annual", "Periodic"]
    ).T

    if frame.index.freq == "Q":
        period, adjustment = "Quarterly", 45
    elif frame.index.freq == "M":
        period, adjustment = "Monthly", 15
    else:
        print("Unrecognised frequency")
        return None

    # set index to the middle of the period for selection
    if from_:
        frame = frame[frame.index >= from_]
    frame = frame.to_timestamp(how="start")
    frame.index = frame.index + pd.Timedelta(days=adjustment)

    # plot
    _, axes = plt.subplots()
    axes.plot(
        frame[frame.columns[0]].index,
        frame[frame.columns[0]].values,
        lw=1,
        color="#0000dd",
        label="Annual growth",
    )
    axes.bar(
        frame[frame.columns[1]].index,
        frame[frame.columns[1]].values,
        color="#dd0000",
        width=0.8 * adjustment * 2,
        label=f"{period} growth",
    )

    locator = mdates.AutoDateLocator(minticks=4, maxticks=13)
    formatter = mdates.ConciseDateFormatter(locator)
    axes.xaxis.set_major_locator(locator)
    axes.xaxis.set_major_formatter(formatter)
    return axes


def plot_growth_finalise(
    annual: pd.Series, periodic: pd.Series, from_: pd.Period | None = None, **kwargs
) -> None:
    """Plot growth and finalise the plot."""

    axes = plot_growth(
        annual,
        periodic,
        from_,
    )
    if axes:
        axes.legend(loc="best", fontsize="small")
        if "ylabel" not in kwargs:
            kwargs["ylabel"] = "Per cent"
        finalise_plot(axes, **kwargs)


def calc_growth(
    series: pd.Series, ppy: int | None = None
) -> list[pd.Series, pd.Series]:
    """Calculate annual and periodic growth for a pandas Series,
    with ppy periods per year."""

    returnable = []
    if ppy is None:
        ppy = {"Q": 4, "M": 12}[series.index.freqstr[:1]]
    for periods in ppy, 1:
        returnable.append(series.pct_change(periods=periods) * 100)
    return returnable  # [annual, periodic]


# --- data recalibration

_keywords = {
    "Number": 0,
    "Thousand": 3,
    "Million": 6,
    "Billion": 9,
    "Trillion": 12,
    "Quadrillion": 15,
}
_r_keywords = {v: k for k, v in _keywords.items()}


def _find_calibration(units: str) -> str | None:
    found = None
    for keyword in _keywords:
        if keyword in units or keyword.lower() in units:
            found = keyword
            break
    return found


def _dont_recalibrate(series: pd.Series, units: str, verbose: bool = False) -> bool:
    if series.max() < 0:
        if verbose:
            print("Negative max numbers will not be adjusted")
        return True
    if not pd.api.types.is_numeric_dtype(series):
        if verbose:
            print(f"Series not numeric {series.dtype}")
        return True
    if _find_calibration(units) is None:
        if verbose:
            print(f"Units not calibrated {units}")
        return True
    if series.max() <= 1000 and series.max() >= 1:
        if verbose:
            print("No adjustments needed")
        return True
    return False


def recalibrate_series(series: pd.Series, units: str) -> tuple[pd.Series, str]:
    """Recalibrate a series of floating point numbers."""

    if _dont_recalibrate(series, units):
        return series, units

    def _recalibrate(factor, step, operator):
        if factor + step in _r_keywords:
            replacement = _r_keywords[factor + step]
            nonlocal units, series  # a bit ugly
            units = units.replace(text, replacement)
            units = units.replace(text.lower(), replacement)
            series = operator(series, 1000)
            return True
        return False

    again = True
    while again:
        text = _find_calibration(units)
        factor = _keywords[text]

        if series.max() > 1000:
            if _recalibrate(factor, 3, truediv):
                continue

        if series.max() < 1:
            if _recalibrate(factor, -3, mul):
                continue

        again = False
    return series, units


def recalibrate_value(value: float, units: str) -> tuple[float, str]:
    """Recalibrate a floating point value."""

    input_ = pd.Series([value])
    output, units = recalibrate_series(input_, units)
    return output[0], units
