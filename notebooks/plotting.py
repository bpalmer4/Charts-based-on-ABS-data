"""A set of functions for plotting with matplotlib."""

# --- imports
# system imports
import re
import sys
from operator import mul, truediv
from pathlib import Path

# data science imports
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# --- constants - default settings


DEFAULT_FILE_TYPE = "png"
DEFAULT_FIG_SIZE = (9, 4.5)
DEFAULT_DPI = 300
DEFAULT_CHART_DIR = "."

COLOR_AMBER = "darkorange"
COLOR_BLUE = "mediumblue"
COLOR_RED = "#dd0000"
COLOR_GREEN = "mediumseagreen"

NARROW_WIDTH = 1.0
WIDE_WIDTH = 2.0
LEGEND_FONTSIZE = "x-small"
LEGEND_SET = {"loc": "best", "fontsize": LEGEND_FONTSIZE}


def _states():
    """Abbreviation names and standardised state colors."""
    
    state_colors = {
        "NSW": "lightblue",
        "Vic": "navy",
        "Qld": "maroon",
        "SA": "red",
        "WA": "gold",
        "Tas": "green",
        "NT": "lightsalmon",  # ochre?
        "ACT": "royalblue",
    }

    state_abbr = {
        "New South Wales": "NSW",
        "Victoria": "Vic",
        "Queensland": "Qld",
        "South Australia": "SA",
        "Western Australia": "WA",
        "Tasmania": "Tas",
        "Northern Territory": "NT",
        "Australian Capital Territory": "ACT",
    }

    for name, abbr in state_abbr.items():
        state_colors[name] = state_colors[abbr]
        state_colors[name.lower()] = state_colors[abbr]
        state_colors[abbr.lower()] = state_colors[abbr]

    return state_colors, state_abbr

state_colors, state_abbr = _states()


def abbreviate(name: str) -> str:
    return state_abbr.get(name, name)


# --- clear_chart_dir()


def clear_chart_dir(chart_dir):
    """Remove all .png files from the chart_dir."""

    for fs_object in Path(chart_dir).glob("*.png"):
        if fs_object.is_file():
            fs_object.unlink()


# --- finalise_plot()

# global chart_dir - modified by set_chart_dir() below
_chart_dir: str | None = DEFAULT_CHART_DIR

# filename limitations - used to map the plot title to a filename
_remove = re.compile(r"[^0-9A-Za-z]")  # make sensible file names
_reduce = re.compile(r"[-]+")  # eliminate multiple hyphens

# map of the acceptable kwargs for finalise_plot()
# make sure "legend" is last in the _splat_kwargs tuple ...
_splat_kwargs = ("axhspan", "axvspan", "axhline", "axvline", "legend")
_value_must_kwargs = ("title", "xlabel", "ylabel")
_value_may_kwargs = ("ylim", "xlim", "yscale", "xscale")
_value_kwargs = _value_must_kwargs + _value_may_kwargs
_annotation_kwargs = ("lfooter", "rfooter", "lheader", "rheader")

_file_kwargs = ("pre_tag", "tag", "chart_dir", "file_type", "dpi")
_fig_kwargs = ("figsize", "show")
_oth_kwargs = ("zero_y", "y0", "dont_save", "dont_close")
_ACCEPTABLE_KWARGS = frozenset(
    _value_kwargs
    + _splat_kwargs
    + _file_kwargs
    + _annotation_kwargs
    + _fig_kwargs
    + _oth_kwargs
)


# - private utility functions for finalise_plot()


# private
def _check_kwargs(**kwargs):
    """Report any unrecognised keyword arguments."""

    for k in kwargs:
        if k not in _ACCEPTABLE_KWARGS:
            print(f"Warning: {k} was an unrecognised keyword argument")


# private
def _apply_value_kwargs(axes, settings: tuple, **kwargs) -> None:
    """Set matplotlib elements by name using Axes.set()."""

    for setting in settings:
        value = kwargs.get(setting, None)
        if value is None and setting not in _value_must_kwargs:
            continue
        axes.set(**{setting: value})


# private
def _apply_splat_kwargs(axes, settings: tuple, **kwargs) -> None:
    """Set matplotlib elements dynamically using setting_name and splat."""

    for method_name in settings:
        if method_name in kwargs:
            if isinstance(kwargs[method_name], dict):
                method = getattr(axes, method_name)
                method(**kwargs[method_name])
            else:
                print(f"Warning expected dict argument: {method_name}")


# private
def _apply_kwargs(axes, **kwargs):
    """Apply settings found in kwargs."""

    def check_kwargs(name):
        return name in kwargs and kwargs[name]

    _apply_value_kwargs(axes, _value_kwargs, **kwargs)
    _apply_splat_kwargs(axes, _splat_kwargs, **kwargs)

    fig = axes.figure
    fs = DEFAULT_FIG_SIZE if "figsize" not in kwargs else kwargs["figsize"]
    fig.set_size_inches(*fs)

    annotations = {
        "rfooter": (0.99, 0.001, "right", "bottom"),
        "lfooter": (0.01, 0.001, "left", "bottom"),
        "rheader": (0.99, 0.999, "right", "top"),
        "lheader": (0.01, 0.999, "left", "top"),
    }
    for annotation in _annotation_kwargs:
        if annotation in kwargs:
            x, y, ha, va = annotations[annotation]
            fig.text(
                x,
                y,
                kwargs[annotation],
                ha=ha,
                va=va,
                fontsize=9,
                fontstyle="italic",
                color="#999999",
            )

    if check_kwargs("zero_y"):
        bottom, top = axes.get_ylim()
        adj = (top - bottom) * 0.02
        if bottom > -adj:
            axes.set_ylim(bottom=-adj)
        if top < adj:
            axes.set_ylim(top=adj)

    if check_kwargs("y0"):
        lo, hi = axes.get_ylim()
        if lo < 0 < hi:
            axes.axhline(y=0, lw=0.75, c="#555555")


# private
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


# public
def set_chart_dir(chart_dir: str | None) -> None:
    """A function to set a global chart directory for finalise_plot(),
    so that it does not need to be included as an argument in each
    call to finalise_plot()."""

    global _chart_dir
    _chart_dir = chart_dir


# public
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
       - lheader - string - text to display on top left of plot
       - rheader - string - text to display of top right of plot
       - figsize - tuple - figure size in inches - eg. (8, 4)
       - show - Boolean - whether to show the plot or not
       - zero_y - bool - ensure y=0 is included in the plot.
       - y0 - bool - highlight the y=0 line on the plot
       - dont_save - bool - dont save the plot to the file system
       - dont_close - bool - dont close the plot
       - dpi - int - dots per inch for the saved chart
       - legend - dict - arguments to pass to axes.legend()
       - axhspan - dict - arguments to pass to axes.axhspan()
       - axvspan - dict - arguments to pass to axes.axvspan()
       - axhline - dict - arguments to pass to axes.axhline()
       - axvline - dict - arguments to pass to axes.axvline()
       - ylim - tuple[float, float] - set lower and upper y-axis limits
       - xlim - tuple[float, float] - set lower and upper x-axis limits
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


# --- line_plot()

# constants
STARTS, TAGS = "starts", "tags"
STYLE, WIDTH, COLOR = "style", "width", "color"
ALPHA, LEGEND, DROPNA = "alpha", "legend", "dropna"
DRAWSTYLE, MARKER = "drawstyle", "marker"
MARKERSIZE = "markersize"


# private
def _apply_defaults(list_len: int, defaults: dict, kwargs: dict) -> tuple[dict, dict]:
    """Get arguments from kwargs, and apply a default from the
    defaults dict if not there."""

    r = {}  # return vehicle
    for option, default in defaults.items():
        if option in kwargs:
            # get the argument, ensure it is in a list
            r[option] = kwargs[option]
            if not isinstance(r[option], list) and not isinstance(r[option], tuple):
                r[option] = [r[option]]  # str -> list[str]
            del kwargs[option]
        else:
            # use the default argument
            r[option] = default if isinstance(default, list) else [default]

        # repeat list if not long enough for all lines to be plotted
        if len(r[option]) < list_len and list_len > 1:
            multiplier = (list_len // len(r[option])) + 1
            r[option] = r[option] * multiplier

    return r, kwargs


# private
def _get_multi_starts(**kwargs) -> tuple[dict[str, list], dict]:
    """Get the multi-starting point arguments."""

    defaults = {  # defaults
        STARTS: None,  # should be first item in dictionary
        TAGS: "",
    }
    stags, kwargs = _apply_defaults(1, defaults, kwargs)

    if len(stags[TAGS]) < len(stags[STARTS]):
        stags[TAGS] = stags[TAGS] * len(stags[STARTS])
    # Ensure that the tags are not identical ...
    if len(stags[TAGS]) > 1 and stags[TAGS].count(stags[TAGS][0]) == len(stags[TAGS]):
        stags[TAGS] = [
            e + f"{i:02d}" if i > 0 else e for i, e in enumerate(stags[TAGS])
        ]

    return stags, kwargs


def _get_style_width_color_etc(n, **kwargs) -> tuple[dict[str, list], dict]:
    """Get the plot-line attributes arguemnts."""

    colours = {
        # default colours change depending on the number of lines
        1: COLOR_RED,
        5: [COLOR_BLUE, COLOR_AMBER, COLOR_GREEN, COLOR_RED, "#888888"],
        9: [
            "#332288",
            "#88CCEE",
            "#44AA99",
            "#117733",
            "#999933",
            "#DDCC77",
            "#CC6677",
            "#882255",
            "#AA4499",
        ],  # Tol
    }
    k = colours.keys()
    minimum = min(i for i in list(k) + [float("inf")] if i >= n)
    n_colours = minimum if minimum is not float("inf") else max(k)
    defaults = {  # defaults
        STYLE: "-",
        WIDTH: WIDE_WIDTH,
        COLOR: colours[n_colours],
        ALPHA: 1.0,
        DRAWSTYLE: None,
        MARKER: None,
        MARKERSIZE: 10,
    }
    swce, kwargs = _apply_defaults(n, defaults, kwargs)

    swce[LEGEND] = None if LEGEND not in kwargs else kwargs[LEGEND]
    if swce[LEGEND] is None and n > 1:
        swce[LEGEND] = LEGEND_SET
    if LEGEND in kwargs:
        del kwargs[LEGEND]
    swce[DROPNA] = False if DROPNA not in kwargs else kwargs[DROPNA]
    if DROPNA in kwargs:
        del kwargs[DROPNA]

    return swce, kwargs


# public
def line_plot(data: pd.Series | pd.DataFrame, **kwargs) -> None:
    """Plot a series or a dataframe over multiple (starting_point) time horizons.
    The data must be a pandas Series or DataFrame with a PeriodIndex.
    Arguments:
    - starts - str| pd.Period | list[str] | list[pd.Period] -
      starting dates for plots.
    - tags - str | list[str] - unique file name tages for multiple plots.
    - color - str | list[str] - line colors.
    - width - float | list[float] - line widths.
    - style - str | list[str] - line styles.
    - alpha - float | list[float] - line transparencies.
    - legend - dict | False - arguments to splat in a call to plt.Axes.legend()
    - drawstyle - str | list[str] - pandas drawing style
      if False, no legend will be displayed.
    - dropna - bool - whether to delete NAs before plotting
    - Remaining arguments as for finalise_plot() [but note, the tag
      argument to finalise_plot cannot be used. Use tags instead.]"""

    # sanity checks
    assert isinstance(data, pd.Series) or isinstance(data, pd.DataFrame)
    assert isinstance(data.index, pd.PeriodIndex)

    # really we are only plotting DataFrames
    if isinstance(data, pd.Series):
        data = pd.DataFrame(data)

    # get extra plotting parameters - not passed to finalise_plot()
    n = len(data.columns)
    stags, kwargs = _get_multi_starts(**kwargs)  # time horizons
    swce, kwargs = _get_style_width_color_etc(n, **kwargs)  # lines

    # And plot
    for start, tag in zip(stags[STARTS], stags[TAGS]):
        if start and not isinstance(start, pd.Period):
            start = pd.Period(start, freq=data.index.freq)
        recent = data[data.index >= start] if start else data
        axes = None
        for i, p in enumerate(recent.columns):
            if recent[p].isna().all():
                continue
            series = (
                recent[p].dropna() if DROPNA in swce and swce[DROPNA] else recent[p]
            )
            axes = series.plot(
                ls=swce[STYLE][i],
                lw=swce[WIDTH][i],
                color=swce[COLOR][i],
                alpha=swce[ALPHA][i],
                marker=swce[MARKER][i],
                ms=swce[MARKERSIZE][i],
                drawstyle=swce[DRAWSTYLE][i],
                ax=axes,
            )

        if LEGEND in swce and isinstance(swce[LEGEND], dict):
            extra = {} if LEGEND not in kwargs else kwargs[LEGEND]
            kwargs[LEGEND] = {**swce[LEGEND], **extra}
            # let finalise plot will add the legend.
        if axes:
            finalise_plot(axes, tag=tag, **kwargs)


def seas_trend_plot(data: pd.DataFrame, **kwargs) -> None:
    """Plot a DataFrame, where the first column is seasonally
    adjusted data, and the second column is trend data."""

    colors = [COLOR_BLUE, COLOR_AMBER]
    widths = [NARROW_WIDTH, WIDE_WIDTH]
    styles = "-"

    if DROPNA not in kwargs:
        kwargs[DROPNA] = True

    line_plot(
        data,
        width=widths,
        color=colors,
        style=styles,
        legend=LEGEND_SET,
        **kwargs,
    )


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
    if series.index.freqstr[:1] not in ("Q", "M", "D"):
        raise ValueError("The series index must have a D, M or Q freq")

    # plot COVID counterfactural
    freq = series.index.freqstr[0]
    if "start_r" in kwargs and "end_r" in kwargs:
        # set by argument - note must set both start_r and end_r
        if verbose:
            print(
                "Using special start/end dates: "
                f'{kwargs["start_r"]} - {kwargs["end_r"]}'
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
            # but allow for odd quarters such as with Job Vacancies
            full_freq = series.index.freq
            start_regression = pd.Period("2016-11-01", freq=full_freq)
            end_regression = pd.Period("2019-11-01", freq=full_freq)

    recent = series[series.index >= start_regression]
    projection = get_projection(recent, end_regression)
    projection.name = "Pre-COVID projection"
    data_set = pd.DataFrame([projection, recent]).T
    kwargs["lfooter"] = (
        kwargs.get("lfooter", "")
        + f"Projection from {start_regression} to {end_regression}. "
    )

    if DROPNA not in kwargs:
        kwargs[DROPNA] = True

    line_plot(
        data_set,
        color=[COLOR_AMBER, COLOR_BLUE],
        width=[NARROW_WIDTH, WIDE_WIDTH],
        style=["--", "-"],
        legend=LEGEND_SET,
        **kwargs,
    )


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
    line_c = "#dd0000"  # rich red
    hilite_c = "gold"

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
    axes = series.plot(drawstyle="steps-post", lw=WIDE_WIDTH, c=line_c)

    # highlight the runs
    for k in range(1, rising_stretches.max() + 1):
        stretch = rising_stretches[rising_stretches == k]
        axes.axvspan(
            stretch.index.min(), stretch.index.max(), color=hilite_c, zorder=-1
        )
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


# --- plot_growth(), plot_growth_finalise() and calc_growth()


def plot_growth(
    annual: pd.Series,
    periodic: pd.Series,
    from_: str | pd.Timestamp | pd.Period | None = None,
    # Note: from_ is neither a list nor a tuple ...
) -> None | plt.Axes:
    """Plot a bar and line percentage growth chart.
    Both pandas Series should have a quarterly or monthly
    PeriodIndex."""

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

    period, adjustment = {
        "Q": ("Quarterly", 45),
        "M": ("Monthly", 15),
    }.get(p := frame.index.freqstr[:1], (None, None))
    if period is None:
        print(f"Unrecognised frequency: {p} :")
        return None

    # set index to the middle of the period for selection
    if from_:
        if not isinstance(from_, pd.Period):
            from_ = pd.Period(from_, freq=periodic.index.freq)
        frame = frame[frame.index >= from_]
    frame = frame.to_timestamp(how="start")
    frame.index = frame.index + pd.Timedelta(days=adjustment)

    # plot
    THICK_LINE_THRESHOLD = 24
    _, axes = plt.subplots()
    axes.plot(
        frame[frame.columns[0]].index,
        frame[frame.columns[0]].values,
        lw=WIDE_WIDTH if len(frame) <= THICK_LINE_THRESHOLD else NARROW_WIDTH,
        color=COLOR_BLUE,
        label="Annual growth",
    )
    axes.bar(
        frame[frame.columns[1]].index,
        frame[frame.columns[1]].values,
        color=COLOR_RED,
        width=0.7 * adjustment * 2,
        label=f"{period} growth",
    )

    locator = mdates.AutoDateLocator(minticks=4, maxticks=13)
    formatter = mdates.ConciseDateFormatter(locator)
    axes.xaxis.set_major_locator(locator)
    axes.xaxis.set_major_formatter(formatter)
    return axes


def plot_growth_finalise(
    annual: pd.Series,
    periodic: pd.Series,
    from_: str | list | tuple | pd.Timestamp | pd.Period | None = None,
    **kwargs,
) -> None:
    """Plot growth and finalise the plot. Repeat if multiple starting
    times in the from_ argument."""

    if not isinstance(from_, list) and not isinstance(from_, tuple):
        from_ = (from_,)

    kwargs[LEGEND] = LEGEND_SET if LEGEND not in kwargs else kwargs[LEGEND]
    tag_stem = kwargs["tag"] if "tag" in kwargs else ""
    if "ylabel" not in kwargs:
        kwargs["ylabel"] = "Per cent Growth"

    for i, start in enumerate(from_):
        axes = plot_growth(
            annual,
            periodic,
            start,
        )
        if axes:
            kwargs["tag"] = f"{tag_stem}-{i}"
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
        # think about: do we need a ffill() in the next line....
        returnable.append(series.pct_change(periods=periods) * 100)
    return returnable  # [annual, periodic]


def calc_and_plot_growth(
    series: pd.Series,
    from_: str | list | tuple | pd.Timestamp | pd.Period | None = None,
    **kwargs,
) -> None:
    # kludge for when called by abs_data_capture.plot_rows_individually()
    if "ylabel" in kwargs:
        if (
            "percent" not in kwargs["ylabel"].lower()
            or "per cent" not in kwargs["ylabel"].lower()
        ):
            print(f"Warning: removing ylabel of {kwargs['ylabel']}")
            del kwargs["ylabel"]

    growth = calc_growth(series)

    plot_growth_finalise(
        *growth,
        from_,
        **kwargs,
    )


# --- data recalibration

# private
_min_recalibrate = "number"  # all lower case
_max_recalibrate = "decillion"  # all lower case
_keywords = {
    _min_recalibrate.title(): 0,
    "Thousand": 3,
    "Million": 6,
    "Billion": 9,
    "Trillion": 12,
    "Quadrillion": 15,
    "Quintillion": 18,
    "Sextillion": 21,
    "Septillion": 24,
    "Octillion": 27,
    "Nonillion": 30,
    _max_recalibrate.title(): 33,
}
_r_keywords = {v: k for k, v in _keywords.items()}


# private
def _find_calibration(units: str) -> str | None:
    found = None
    for keyword in _keywords:
        if keyword in units or keyword.lower() in units:
            found = keyword
            break
    return found


# private
def _can_recalibrate(n: np.ndarray, units: str, verbose: bool = False) -> bool:
    if not np.issubdtype(n.dtype, np.number):
        print("recalibrate(): Non numeric input data")
        return False
    if _find_calibration(units) is None:
        if verbose:
            print(f"recalibrate(): Units not appropriately calibrated: {units}")
        return False
    if n.max() <= 1000 and n.max() >= 1:
        if verbose:
            print("recalibrate(): No adjustments needed")
        return False
    return True


# private
def _do_recal(n, units, step, operator):
    calibration = _find_calibration(units)
    factor = _keywords[calibration]
    if factor + step not in _r_keywords:
        print(f"Unexpected factor: {factor + step}")
        sys.exit(-1)
    replacement = _r_keywords[factor + step]
    units = units.replace(calibration, replacement)
    units = units.replace(calibration.lower(), replacement)
    n = operator(n, 1000)
    return n, units


# public
def recalibrate(
    data: pd.Series | pd.DataFrame,
    units: str,
    verbose: bool = False,
) -> tuple[pd.Series | pd.DataFrame, str]:
    """Recalibrate a pandas Series or DataFrame."""

    n = data.to_numpy().flatten()
    money = False
    if units.strip() == "$":
        if verbose:
            print("recalibrate() is wrking with money.")
        money = True
        units = "number"

    if _can_recalibrate(n, units, verbose):
        while True:
            maximum = np.nanmax(np.abs(n))
            if maximum > 1000:
                if _max_recalibrate in units.lower():
                    print("recalibrate() is not designed for very big units")
                    break
                n, units = _do_recal(n, units, 3, truediv)
                continue
            if maximum < 1:
                if _min_recalibrate in units.lower():
                    print("recalibrate() is not designed for very small units")
                    break
                n, units = _do_recal(n, units, -3, mul)
                continue
            break

    if money:
        if units == "number":
            units = "$"
        else:
            units = f"{units} $" 

    restore_pandas = pd.DataFrame if len(data.shape) == 2 else pd.Series
    result = restore_pandas(n.reshape(data.shape))
    result.index = data.index
    if len(data.shape) == 2:
        result.columns = data.columns
    if len(data.shape) == 1:
        result.name = data.name
    return result, units


# public
def recalibrate_series(
    data: pd.Series,
    units: str,
    verbose: bool = False,
) -> tuple[pd.Series, str]:
    """Retained for compatibility with earlier code.
    It calls recalibrate()."""

    return recalibrate(data, units, verbose)


# public
def recalibrate_value(value: float, units: str) -> tuple[float, str]:
    """Recalibrate a floating point value."""

    input_ = pd.Series([value])
    output, units = recalibrate(input_, units)
    return output[0], units
