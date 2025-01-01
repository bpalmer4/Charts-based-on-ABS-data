"""A set of functions for plotting timeseries data 
   with matplotlib.
   The intent is to reduce repetitive code, while
   maintaining a consistent look and feel for chart
   outputs. """

# --- imports
# system imports
import re
from pathlib import Path
from typing import Any, Final, cast, TypeVar

# data science imports
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from readabs import recalibrate


# --- constants - default settings
_DataT = TypeVar("_DataT", Series, DataFrame)  # python 3.11+

# no need to set plotstyle elsewhere
plt.style.use("fivethirtyeight")
mpl.rcParams["font.size"] = 12


DEFAULT_FILE_TYPE: Final[str] = "png"
DEFAULT_FIG_SIZE: Final[tuple[float, float]] = (9.0, 4.5)
DEFAULT_DPI: Final[int] = 300
DEFAULT_CHART_DIR: Final[str] = "."

COLOR_AMBER: Final[str] = "darkorange"
COLOR_BLUE: Final[str] = "mediumblue"
COLOR_RED: Final[str] = "#dd0000"
COLOR_GREEN: Final[str] = "mediumseagreen"

NARROW_WIDTH: Final[float] = 0.75
WIDE_WIDTH: Final[float] = 2.0
LEGEND_FONTSIZE: Final[str] = "small"
LEGEND_SET: Final[dict[str, Any]] = {"loc": "best", "fontsize": LEGEND_FONTSIZE}


# --- standard Australian state colors and abbreviations


# private
def _states() -> tuple[dict[str, str], dict[str, str]]:
    """Generate standardised state abbreviations and colors.
    Returns a tuple of two dictionaries. The first dictionary
    maps state names and abbreviations to their standard colors.
    The second dictionary maps state names to their
    standard abbreviations."""

    state_data = {
        "New South Wales": ("NSW", "deepskyblue"),
        "Victoria": ("Vic", "navy"),
        "Queensland": ("Qld", "#c32148"),  # a lighter maroon
        "South Australia": ("SA", "red"),
        "Western Australia": ("WA", "gold"),
        "Tasmania": ("Tas", "seagreen"),  # bottle green is too dark #006A4E
        "Northern Territory": ("NT", "#CC7722"),  # ochre
        "Australian Capital Territory": ("ACT", "blue"),
    }

    colors, abbreviations = {}, {}
    for name, (abbreviation, color) in state_data.items():
        abbreviations[name] = abbreviation
        colors[name] = color
        colors[name.lower()] = color
        colors[abbreviation] = color
        colors[abbreviation.lower()] = color

    return colors, abbreviations


# public
state_colors, state_abbr = _states()


# public
def abbreviate(name: str) -> str:
    """Abbreviate a state name. Returns an abbreviation."""
    return state_abbr.get(name, name)


# --- chart directory stuff


# public
def clear_chart_dir(chart_dir: str) -> None:
    """Remove all graph-image files from the chart_dir."""

    for ext in ("png", "svg"):
        for fs_object in Path(chart_dir).glob(f"*.{ext}"):
            if fs_object.is_file():
                fs_object.unlink()


# private - global chart directory ...
class ChartDirSingleton:
    """Global repository for the class directory.
    Singleton instance implied, but not enforced."""

    def __init__(self, chart_dir: str | None):
        self.set(chart_dir)

    def set(self, chart_dir: str | None) -> None:
        """Setter."""
        self.chart_dir = chart_dir

    def get(self) -> str:
        """Getter."""
        return DEFAULT_CHART_DIR if self.chart_dir is None else self.chart_dir


_chart_dir = ChartDirSingleton(None)  # global chart directory stored here


# public
def set_chart_dir(chart_dir: str | None) -> None:
    """A function to set a global chart directory for finalise_plot(),
    so that it does not need to be included as an argument in each
    call to finalise_plot(). Create the directory if it does not exist."""

    if chart_dir is not None:
        Path(chart_dir).mkdir(parents=True, exist_ok=True)
    _chart_dir.set(chart_dir)


# --- finalise_plot()


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
_oth_kwargs = ("zero_y", "y0", "x0", "dont_save", "dont_close")
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
def _check_kwargs(**kwargs) -> None:
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
                if kwargs[method_name] is not None:
                    print(f"Warning expected dict argument: {method_name}")


# private
def _apply_annotations(axes, **kwargs) -> None:
    """Set figure size and apply chart annotations."""

    fig = axes.figure
    fig_size = DEFAULT_FIG_SIZE if "figsize" not in kwargs else kwargs["figsize"]
    fig.set_size_inches(*fig_size)

    annotations = {
        "rfooter": (0.99, 0.001, "right", "bottom"),
        "lfooter": (0.01, 0.001, "left", "bottom"),
        "rheader": (0.99, 0.999, "right", "top"),
        "lheader": (0.01, 0.999, "left", "top"),
    }

    for annotation in _annotation_kwargs:
        if annotation in kwargs:
            x_pos, y_pos, h_align, v_align = annotations[annotation]
            fig.text(
                x_pos,
                y_pos,
                kwargs[annotation],
                ha=h_align,
                va=v_align,
                fontsize=8,
                fontstyle="italic",
                color="#999999",
            )


# private
def _apply_late_kwargs(axes, **kwargs) -> None:
    """Apply settings found in kwargs, after plotting the data."""
    _apply_splat_kwargs(axes, _splat_kwargs, **kwargs)


def _apply_kwargs(axes, **kwargs) -> None:
    """Apply settings found in kwargs."""

    def check_kwargs(name):
        return name in kwargs and kwargs[name]

    _apply_value_kwargs(axes, _value_kwargs, **kwargs)
    _apply_annotations(axes, **kwargs)

    if check_kwargs("zero_y"):
        bottom, top = axes.get_ylim()
        adj = (top - bottom) * 0.02
        if bottom > -adj:
            axes.set_ylim(bottom=-adj)
        if top < adj:
            axes.set_ylim(top=adj)

    if check_kwargs("y0"):
        low, high = axes.get_ylim()
        if low < 0 < high:
            axes.axhline(y=0, lw=0.66, c="#555555")

    if check_kwargs("x0"):
        low, high = axes.get_xlim()
        if low < 0 < high:
            axes.axvline(x=0, lw=0.66, c="#555555")


# private
def _save_to_file(fig, **kwargs) -> None:
    """Save the figure to file."""

    saving = not kwargs.get("dont_save", False)  # save by default
    if saving:
        chart_dir = kwargs.get("chart_dir", None)
        if chart_dir is None:
            chart_dir = _chart_dir.get()

        title = "" if "title" not in kwargs else kwargs["title"]
        max_title_len = 150  # avoid overly long file names
        shorter = title if len(title) < max_title_len else title[:max_title_len]
        pre_tag = kwargs.get("pre_tag", "")
        tag = kwargs.get("tag", "")
        file_title = re.sub(_remove, "-", shorter).lower()
        file_title = re.sub(_reduce, "-", file_title)
        file_type = kwargs.get("file_type", DEFAULT_FILE_TYPE).lower()  # png or svg
        dpi = kwargs.get("dpi", DEFAULT_DPI)
        fig.savefig(f"{chart_dir}{pre_tag}{file_title}-{tag}.{file_type}", dpi=dpi)


# - public functions for finalise_plot()


# public
def get_possible_kwargs() -> list[str]:
    """Return a list of possible kwargs for finalise_plot()."""
    return list(_ACCEPTABLE_KWARGS)


# public
def finalise_plot(axes, **kwargs) -> None:
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
       - x0 - bool - highlights the x=0 line on the plot
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

    _apply_late_kwargs(axes, **kwargs)

    _save_to_file(fig, **kwargs)

    # show the plot in Jupyter Lab
    if "show" in kwargs and kwargs["show"]:
        plt.show()

    # And close
    closing = True if "dont_close" not in kwargs else not kwargs["dont_close"]
    if closing:
        plt.close()


# --- line_plot()

# constants
STARTS, TAGS = "starts", "tags"
AX = "ax"
STYLE, WIDTH, COLOR = "style", "width", "color"
ALPHA, LEGEND, DROPNA = "alpha", "legend", "dropna"
DRAWSTYLE, MARKER = "drawstyle", "marker"
MARKERSIZE = "markersize"


# private
def _apply_defaults(list_len: int, defaults: dict, kwargs: dict) -> tuple[dict, dict]:
    """Get arguments from kwargs, and apply a default from the
    defaults dict if not there."""

    returnable = {}  # return vehicle
    for option, default in defaults.items():
        if option in kwargs:
            # get the argument, ensure it is in a list
            returnable[option] = kwargs[option]
            if not isinstance(returnable[option], list) and not isinstance(
                returnable[option], tuple
            ):
                returnable[option] = [returnable[option]]  # str -> list[str]
            del kwargs[option]
        else:
            # use the default argument
            returnable[option] = default if isinstance(default, list) else [default]

        # repeat list if not long enough for all lines to be plotted
        if len(returnable[option]) < list_len and list_len > 1:
            multiplier = (list_len // len(returnable[option])) + 1
            returnable[option] = returnable[option] * multiplier

    return returnable, kwargs


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
            e + f"{i:03d}" if i > 0 else e for i, e in enumerate(stags[TAGS])
        ]

    return stags, kwargs


def _get_style_width_color_etc(
    item_count, num_data_points, **kwargs
) -> tuple[dict[str, list], dict]:
    """Get the plot-line attributes arguemnts."""

    if "color" not in kwargs:

        colours: dict[int, str | list[str]] = {
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
            ],
        }
        if item_count > max(colours.keys()):
            # generate a gradient of colours
            c = cm.get_cmap("nipy_spectral")(np.linspace(0, 1, item_count))
            crgb = [
                f"#{int(x*255):02x}{int(y*255):02x}{int(z*255):02x}" for x, y, z, _ in c
            ]
            colours[item_count] = crgb

        k = colours.keys()
        minimum: float | int = min(
            i for i in list(k) + [float("inf")] if i >= item_count
        )
        if np.isinf(minimum):
            n_colours = max(k)
        else:
            n_colours = int(minimum)
        color = colours[n_colours]
    else:
        color = kwargs["color"]

    data_point_thresh = 24
    defaults: dict[str, Any] = {
        STYLE: "-",
        WIDTH: NARROW_WIDTH if num_data_points > data_point_thresh else WIDE_WIDTH,
        COLOR: color,
        ALPHA: 1.0,
        DRAWSTYLE: None,
        MARKER: None,
        MARKERSIZE: 10,
    }
    swce, kwargs = _apply_defaults(item_count, defaults, kwargs)

    swce[LEGEND] = kwargs.get(LEGEND, None)
    if swce[LEGEND] is None and item_count > 1:
        swce[LEGEND] = LEGEND_SET
    if LEGEND in kwargs:
        del kwargs[LEGEND]

    swce[DROPNA] = kwargs.get(DROPNA, False)
    if DROPNA in kwargs:
        del kwargs[DROPNA]

    return swce, kwargs


# public
def line_plot(data: _DataT, **kwargs: Any) -> None:
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
    - ax - plt.Axes | None - axes to plot on (optional)
    - Remaining arguments as for finalise_plot() [but note, the tag
      argument to finalise_plot cannot be used. Use tags instead.]"""

    # sanity checks
    if not isinstance(data, (Series, DataFrame)) or not isinstance(
        data.index, pd.PeriodIndex
    ):
        raise TypeError(
            "The data argument must be a pandas Series or DataFrame with a PeriodIndex"
        )
    if AX in kwargs and kwargs[AX] is not None and STARTS in kwargs:
        print("Caution: only one chart can be plotted with the passed axes")

    # really we are only plotting DataFrames
    df = DataFrame(data)

    # get extra plotting parameters - not passed to finalise_plot()
    item_count = len(df.columns)
    num_data_points = len(df)
    stags, kwargs = _get_multi_starts(**kwargs)  # time horizons
    swce, kwargs = _get_style_width_color_etc(
        item_count, num_data_points, **kwargs
    )  # lines

    # And plot
    for start, tag in zip(stags[STARTS], stags[TAGS]):
        if start and not isinstance(start, pd.Period):
            start = pd.Period(start, freq=cast(pd.PeriodIndex, df.index).freq)
        recent = df[df.index >= start] if start else df
        # note cammot use ax and start/tag in finalise_plot
        axes: None | plt.Axes = kwargs.pop(AX, None)
        for i, column in enumerate(recent):
            if (recent[column].isna()).all():
                continue
            series = (
                recent[column].dropna()
                if DROPNA in swce and swce[DROPNA]
                else recent[column]
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
            kwargs[LEGEND] = swce[LEGEND].copy()
        if axes:
            finalise_plot(axes, tag=tag, **kwargs)


def seas_trend_plot(data: DataFrame, **kwargs) -> None:
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


def get_projection(original: Series, to_period: pd.Period) -> Series:
    """Projection based on data from the start of a series
    to the to_period (inclusive). Returns projection over the whole
    period of the original series."""

    y_regress = original[original.index <= to_period].copy()
    x_regress = np.arange(len(y_regress))
    m, b = np.polyfit(x_regress, y_regress, 1)

    x_complete = np.arange(len(original))
    projection = Series((x_complete * m) + b, index=original.index)

    return projection


def plot_covid_recovery(series: Series, verbose=False, **kwargs) -> None:
    """Plots a series with a PeriodIndex.
    Arguments
     - series to be plotted
     - **kwargs - same as for finalise_plot()."""

    # sanity checks
    if not isinstance(series, Series):
        raise TypeError("The series argument must be a pandas Series")
    if not isinstance(series.index, pd.PeriodIndex):
        raise TypeError("The series must have a pandas PeriodIndex")
    if series.index.freqstr[:1] not in ("Q", "M", "D"):
        raise ValueError("The series index must have a D, M or Q freq")

    # plot COVID counterfactural
    freq = series.index.freqstr  # CHECK
    if "start_r" in kwargs and "end_r" in kwargs:
        # set COVID regression using a bespoke range
        # Note must set both start_r and end_r
        start_regression = pd.Period(kwargs.pop("start_r"), freq=freq)
        start_regression = max(start_regression, series.dropna().index.min())
        end_regression = pd.Period(kwargs.pop("end_r"), freq=freq)
        assert start_regression < end_regression
        if verbose:
            print(
                f"Bespoke pre-COVID regression from {start_regression=} to {end_regression=}"
            )
    else:
        # set COVID regression programatically
        if freq[0] == "Q":
            start_regression = pd.Period("2014Q4", freq=freq)
            end_regression = pd.Period("2019Q4", freq=freq)
        else:
            start_regression = pd.Period("2015-01-31", freq=freq)
            end_regression = pd.Period("2020-01-31", freq=freq)
        if verbose:
            print(
                f"Default pre-COVID regression from {start_regression} to {end_regression}"
            )

    recent = series[series.index >= start_regression].copy()
    recent.name = "Series"
    projection = get_projection(recent, end_regression)
    projection.name = "Pre-COVID projection"
    data_set = DataFrame([projection, recent]).T
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


def _identify_runs(series: Series, threshold: float) -> tuple[Series, Series]:
    """Identify monotonic increasing runs."""
    diffed = series.diff()
    change_points = pd.concat(
        [diffed[diffed.gt(threshold)], diffed[diffed.lt(-threshold)]]
    ).sort_index()
    if series.index[0] not in change_points.index:
        starting_point = Series([0], index=[series.index[0]])
        change_points = pd.concat([change_points, starting_point]).sort_index()
    rising = change_points > 0
    cycles = (rising & ~rising.shift().astype(bool)).cumsum()
    return cycles[rising], change_points


def plot_series_highlighted(series: Series, **kwargs) -> plt.Axes:
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

    rising_stretches, change_points = _identify_runs(series, threshold)

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


def _format_x_axis(axes: plt.Axes, minticks: int, maxticks: int) -> None:
    """Format the x-axis of a matplotlib Axes object."""

    locator = mdates.AutoDateLocator(minticks=minticks, maxticks=maxticks)
    formatter = mdates.ConciseDateFormatter(locator)
    axes.xaxis.set_major_locator(locator)
    axes.xaxis.set_major_formatter(formatter)


def _plot_growth_line_bars(
    frame: DataFrame,
    period: str,
    adjustment: int,
    annotate: int | str = 0,
    annotation_rounding: int = 1,
) -> plt.Axes:
    """Private function: Plot a bar and line percentage growth chart."""

    max_annotation = 30  # maximum number of bars to annotate
    thick_line_threshold = 180  # thin line if too many data points

    _fig, axes = plt.subplots()
    axes.plot(
        frame.index,
        frame["Annual"].to_numpy(),
        lw=(WIDE_WIDTH if len(frame) <= thick_line_threshold else NARROW_WIDTH),
        color=COLOR_BLUE,
        label="Annual growth",
    )
    axes.bar(
        frame.index,
        frame["Periodic"].to_numpy(),
        color=COLOR_RED,
        width=0.7 * adjustment * 2,
        label=f"{period} growth",
    )

    if annotate and len(frame) <= max_annotation:
        annotate_style = {
            "fontsize": annotate,
            "fontname": "Helvetica",
        }
        adjustment = (frame.max().max() - frame.min().min()) * 0.005
        for i, value in enumerate(frame["Periodic"]):
            va = "bottom" if value >= 0 else "top"
            text = axes.text(
                frame.index[i],
                adjustment if value >= 0 else -adjustment,
                f"{value:.{annotation_rounding}f}",
                ha="center",
                va=va,
                **annotate_style,
                fontdict=None,
                color="white",
            )
            text.set_path_effects([pe.withStroke(linewidth=2, foreground=COLOR_RED)])
            axes.text(
                frame.index[-1],
                frame["Annual"].iloc[-1],
                f" {frame["Annual"].iloc[-1]:.{annotation_rounding}f}",
                ha="left",
                va="center",
                **annotate_style,
                fontdict=None,
                color=COLOR_BLUE,
            )

    _format_x_axis(axes, minticks=4, maxticks=16)
    return axes


def plot_growth(
    annual: Series,
    periodic: Series,
    from_: str | pd.Period | None = None,
    annotate: int | str = 0,
    annotation_rounding: int = 1,
) -> None | plt.Axes:
    """Plot a bar and line percentage growth chart.
    Both pandas Series should have a quarterly or monthly
    PeriodIndex. Allow an option to annotate the bars, provided
    the number of bars is less than max_annotation, the value of
    annotate is fontsize (suggest: "x-small")."""

    # sanity checks
    for series in (annual, periodic):
        assert isinstance(series, Series), "initial arguments should be pandas Series"
        assert isinstance(
            series.index, pd.PeriodIndex
        ), "Series index should be a PeriodIndex"

    # put our two series into a datadrame
    frame = DataFrame([annual, periodic], index=["Annual", "Periodic"]).T

    df_period = cast(pd.PeriodIndex, frame.index).freqstr[:1]
    if not df_period or df_period not in "QM":
        print(f"Unrecognised frequency: {df_period} :")
        return None
    period, adjustment = {
        "Q": ("Quarterly", 45),
        "M": ("Monthly", 15),
    }[df_period]

    # set index to the middle of the period for selection
    if from_ is not None:
        frame = frame.loc[lambda x: x.index >= from_]
    frame = frame.to_timestamp(how="start")
    frame.index = frame.index + pd.Timedelta(days=adjustment)

    # plot
    return _plot_growth_line_bars(
        frame, period, adjustment, annotate, annotation_rounding
    )


def plot_growth_finalise(
    annual: Series,
    periodic: Series,
    from_: str | list | tuple | pd.Timestamp | pd.Period | None = None,
    annotate: int = 0,
    **kwargs,
) -> None:
    """Plot growth and finalise the plot. Repeat if multiple starting
    times in the from_ argument."""

    # defaults - and specific settings for this function
    annotation_rounding = kwargs.pop("annotation_rounding", 1)
    kwargs[LEGEND] = kwargs.get(LEGEND, LEGEND_SET)
    kwargs["zero_y"] = kwargs.get("zero_y", True)
    kwargs["ylabel"] = kwargs.get("ylabel", "Per cent Growth")

    # loop through multiple starting points
    tag_stem = kwargs.get("tag", "")
    if not isinstance(from_, list) and not isinstance(from_, tuple):
        from_ = (from_,)
    for i, start in enumerate(from_):
        axes = plot_growth(
            annual,
            periodic,
            start,
            annotate=annotate,
            annotation_rounding=annotation_rounding,
        )
        if axes:
            kwargs["tag"] = f"{tag_stem}-{i}"
            finalise_plot(axes, **kwargs)


def calc_growth(series: Series, ppy: int | None = None) -> tuple[Series, Series]:
    """Calculate annual and periodic growth for a pandas Series,
    with ppy periods per year."""

    if ppy is None:
        ppy = {"Q": 4, "M": 12}[cast(pd.PeriodIndex, series.index).freqstr[:1]]
    annual = series.pct_change(periods=ppy) * 100
    periodic = series.pct_change(periods=1) * 100
    return annual, periodic


def calc_and_plot_growth(
    series: Series,  # but single column DataFrame is allowed
    from_: str | list | tuple | pd.Timestamp | pd.Period | None = None,
    **kwargs,
) -> None:
    """Calculate and plot growth for a series."""

    # kludge for when called by abs_data_capture.plot_rows_individually()
    if "ylabel" in kwargs:
        if (
            "percent" not in kwargs["ylabel"].lower()
            or "per cent" not in kwargs["ylabel"].lower()
        ):
            print(f"Warning: removing ylabel of {kwargs['ylabel']}")
            del kwargs["ylabel"]

    s: Series | DataFrame = series
    # kludge to allow a single column DataFrame to be passed
    if isinstance(s, DataFrame):
        if len(series.columns) == 1:
            s = s[s.columns[0]]
        else:
            raise TypeError("Series expected, not a DataFrame")
    growth = calc_growth(s)

    plot_growth_finalise(
        *growth,
        from_=from_,
        **kwargs,
    )


# --- plot ABS data revisions


# public
def plot_revisions(data: pd.DataFrame, units: str, recent=18, **kwargs) -> None:
    """plot the revisions to ABS data
    Arguments
    data : pd.DataFrame : the data to plot, a column for each data revision
    units : str : the units for the data
    recent : int : the number of recent data points to plot
    kwargs : dict : additional arguments to pass to finalise_plot
        (Note: the ylabel for the plot will be adjusted units)."""

    # adjust units
    repository, units = recalibrate(data[data.columns[::-1]].tail(recent), units)

    # plot the data
    ax = repository.plot()

    # Annotate the last value in each series ...
    for c in repository.columns:
        col: pd.Series = repository.loc[:, c].dropna()
        x, y, s = (
            col.index[-1],
            col.iloc[-1],
            f" {col.iloc[-1]:.{2 if col.iloc[-1] < 100 else 1}f}",
        )
        ax.text(x, y, s, fontsize=10, va="center", ha="left")

    # change the line width for new data
    how_far_back = len(data.columns)
    linewidth = (np.arange(0, how_far_back) / (how_far_back - 1)) + 1
    for line, width in zip(ax.get_lines(), linewidth):
        line.set_linewidth(width)

    finalise_plot(
        ax,
        ylabel=f"{units}",
        **kwargs,
    )
