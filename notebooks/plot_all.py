#!/usr/bin/env python3
"""plot_all.py:
This is a CLI-tool to plot all charts for a specific ABS catalog number."""

# --- imports
# system imports
import sys
from dataclasses import dataclass
import time
from typing import cast, TypeVar

# analytic imports
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
from matplotlib import ticker
from bs4 import BeautifulSoup

# local imports
from plotting import (
    clear_chart_dir,
    line_plot,
    recalibrate,
    set_chart_dir,
    seas_trend_plot,
    finalise_plot,
)
from abs_data_capture import (
    AbsCaptureError,
    AbsDict,
    get_data_links,
    get_abs_data,
    AbsLandingPage,
    get_fs_constants,
    metacol,
    SEAS_ADJ,
    TREND,
)
import common


# --- constants
RECENT_YEARS = 7
EXTRA_MONTHS = 4
TODAY = pd.Timestamp("today")
RECENT = TODAY - pd.DateOffset(years=RECENT_YEARS, months=EXTRA_MONTHS)
PLOT_TIMES = (None, RECENT)

DataT = TypeVar("DataT", pd.Series, pd.DataFrame)  # python 3.11+


# --- Identify available ABS time-series datasets
def build_links_dict() -> dict[str, tuple[str, AbsLandingPage]]:
    """Build a catalogue list using data from the ABS web page.
    Note:
        5655.0: does not have a zip file download facility.
        8165.0: Does not have time-series data."""

    url = "https://www.abs.gov.au/about/data-services/help/abs-time-series-directory"
    html = common.request_get(url)
    table = BeautifulSoup(html, features="lxml").select("table")[-1]

    odd = {"5655.0", "8165.0"}
    link_dict = {}
    for row_num, row in enumerate(table.findAll("tr")):
        if "Ceased" in row.text or row_num < 2:
            continue
        for i, cell in enumerate(row.findAll("td")):
            if i == 0:
                cat_id = cell.text.strip()
                if cat_id in odd:
                    break
            if i == 1:
                landing_page = AbsLandingPage(
                    *cell.find("a").get("href").rsplit("/", 4)[-4:-1]
                )
                link_dict[cat_id] = (cell.text, landing_page)

    return link_dict


LINK_DICT = build_links_dict()


# --- Provide the user with some assistance
def print_known_cat_ids() -> None:
    """Print known catalogue ids."""

    print("Cat#:   Description")
    for cat, (desc, _) in LINK_DICT.items():
        print(f"{cat}: {desc}")


def give_assistance() -> None:
    """Provide some help text"""

    print("To generate plots: plot_all.py [-v -t -a] cat# [cat# ...]")
    print("Where:")
    print("\tcat# is an ABS catalog number")
    print("\t-v is a flag for verbose feedback")
    print("\t-t is a flag for testing without actually plotting")
    print("\t-a is a flag for all catalog numbers.\n")
    print_known_cat_ids()
    print("\nFor this assistance: plot_all.py --help")


# --- plot all the charts for a specific ABS catalogue number
@dataclass(frozen=True)
class Tudds:
    """A vehicle for remembering some key information."""

    table: str
    unit: str
    dtype: str
    data: AbsDict
    source: str


def plot_one_zip_file(
    cat_id: str,
    zip_table_num: int,
    verbose: bool,
    test_mode: bool,
) -> None:
    """For a spcific cat_id and zip_table_num on the relevant ABS
    landing page call get_plot_data() to obtain the excel tables.
    Then call group_and_plot() to methodically work through the
    excel tables in the zipfile and plot every series therein."""

    try:
        data_tuple = get_plot_data(cat_id, zip_table_num, verbose)
    except AbsCaptureError:
        return

    if verbose:
        print(
            f"About to plot series from: catalogue: {cat_id}, zip number: {zip_table_num}"
        )

    if not test_mode:
        group_and_plot(*data_tuple)


def get_plot_data(
    cat_id: str,
    zip_table_num: int,
    verbose: bool,
) -> tuple[AbsDict, pd.DataFrame, str]:
    """Data acquisition and plot initialisation.
    Raises AbsCaptureError if data was not obtained."""

    # --- data acquisition
    if cat_id not in LINK_DICT:
        print(f"Cannot find the ABS catalogue identifier: {cat_id}.")
        raise AbsCaptureError
    _, landing_page = LINK_DICT[cat_id]

    try:
        abs_dict = get_abs_data(landing_page, zip_table_num, verbose)
    except AbsCaptureError as error:
        print(
            "Something went wrong when getting zip number "
            f"{zip_table_num} from {cat_id}: {error}"
        )
        raise error

    source, chart_dir, _, meta = get_fs_constants(abs_dict, landing_page, " - ALL")

    # --- plot initialisation
    set_chart_dir(chart_dir)
    clear_chart_dir(chart_dir)
    plt.style.use("fivethirtyeight")
    mpl.rcParams["font.size"] = 10

    return abs_dict, meta, source


def group_and_plot(
    abs_dict: AbsDict,
    meta: pd.DataFrame,
    source: str,
) -> None:
    """Group like data and dispatch for plotting."""

    for table in meta[metacol.table].unique():
        sub_set1 = meta[meta[metacol.table] == table]
        for unit in sub_set1[metacol.unit].unique():
            sub_set2 = sub_set1[sub_set1[metacol.unit] == unit]
            for dtype in sub_set2[metacol.dtype].unique():
                sub_set3 = sub_set2[sub_set2[metacol.dtype] == dtype]
                plot_subset(sub_set3, Tudds(table, unit, dtype, abs_dict, source))


def plot_subset(subset: pd.DataFrame, tudds: Tudds) -> None:
    """Determine how subsets should be plotted and dispatch"""

    for did in subset[metacol.did].unique():
        selected = subset[subset[metacol.did] == did]
        lower_did = did.lower()
        if len(selected) == 1:
            if (
                "change" in lower_did
                and "percent" in lower_did
                and "year" not in lower_did
                and "annual" not in lower_did
            ):
                x_bar_plot(selected, tudds)
            else:
                x_line_plot(selected, tudds)
        elif (
            SEAS_ADJ in selected[metacol.stype].unique()
            and TREND in selected[metacol.stype].unique()
        ):
            s_seastrend_plot(selected, tudds)
        else:
            for i in range(len(selected.index)):
                x_line_plot(selected.iloc[[i]], tudds)
        # TO DO - think about annual-v-per-period percentage change charts


def title_fix(title: str, max_line_length: int = 80) -> str:
    """Split unusually long titles, into roughly equal parts,
    having regard to the max_line_length in characters.
    Splits occur around white spaces."""

    # Crunch down where the ABS often adds extra padding ...
    title = title.strip().replace(" ;", ";")
    single_space = " "
    title = f"{single_space.join(title.split())} "  # removes all multi-white-spaces

    # split lines roughly in equal parts around spaces.
    flexibility = 5  # chars - some line length flexibility
    max_line_length -= flexibility
    n_folds = ((length := len(title)) // max_line_length) + 1
    if n_folds > 1:
        for fold in range(1, n_folds):
            spaces = [
                pos
                for (pos, char) in enumerate(title)
                if char == single_space and pos < fold * (max_line_length + flexibility)
            ]
            optima = {  # how close is a space to the perfect fold
                abs(int(length * fold / n_folds) - p): i for i, p in enumerate(spaces)
            }
            break_point = spaces[optima[min(optima.keys())]]
            left, right = title[:break_point], title[break_point:]
            title = f"{left.strip()}\n{right.strip()}"

    return title.strip()


def build_title(meta: DataT) -> str:
    """Titles comprise table description and data item descrption."""

    if isinstance(meta, pd.Series):
        tdesc = meta[metacol.tdesc]
        did = meta[metacol.did]
    else:
        row_zero = meta.index[0]
        tdesc = meta.at[row_zero, metacol.tdesc]
        did = meta.at[row_zero, metacol.did]

    return f"{title_fix(tdesc)}\n{title_fix(did)}"


def series_fix(series: pd.Series) -> pd.Series:
    """Remove leading and trailing NaNs from a series.
    Also ensure the series is numeric."""

    series = series.astype(float)  # not sure this is needed.
    first = series.first_valid_index()
    last = series.last_valid_index()
    if first is not None and last is not None:
        return series.loc[first:last]
    return pd.Series()


def s_seastrend_plot(selected: pd.DataFrame, tudds: Tudds):
    """Produce seasonally adjusted / trend plots."""

    # just checking
    if len(selected) > 3 or len(selected) < 2:
        print(selected)

    # plot the original series
    if "Original" in selected[metacol.stype]:
        o = selected[selected[metacol.stype] == "Original"]
        x_line_plot(o, tudds)

    # plot a seasonal/trend
    s_row = selected[selected[metacol.stype] == SEAS_ADJ].squeeze()
    t_row = selected[selected[metacol.stype] == TREND].squeeze()
    s_data = tudds.data[tudds.table][s_row[metacol.id]]
    t_data = tudds.data[tudds.table][t_row[metacol.id]]
    data = pd.DataFrame({SEAS_ADJ: s_data, TREND: t_data})
    data = data.dropna(how="all").astype(float)
    data, unit = recalibrate(data, tudds.unit)

    title = build_title(s_row)
    series_id = f"{s_row[metacol.id]}-{t_row[metacol.id]}"
    seas_trend_plot(
        data=data,
        starts=PLOT_TIMES,
        title=title,
        ylabel=unit,
        rfooter=f"{tudds.source}-{tudds.table}-{series_id}",
        tags=series_id,
        y0=True,
    )


def x_bar_plot(selected: pd.DataFrame, tudds: Tudds) -> None:
    """Produce a single-series bar chart."""

    assert len(selected) == 1
    row = selected.index[0]
    series_id = selected.at[row, metacol.id]
    series_type = selected.at[row, metacol.stype]
    data = series_fix(tudds.data[tudds.table][series_id]).copy()
    move = -15 if cast(pd.PeriodIndex, data.index).freqstr[0] == "M" else -45
    width = -1.5 * move
    data.index = cast(pd.PeriodIndex, data.index).to_timestamp(
        how="end"
    ) + pd.Timedelta(move, unit="d")
    unit = tudds.unit

    data, unit = recalibrate(data, unit)

    if len(data) == 0:
        return

    title = build_title(selected)
    for start in PLOT_TIMES:
        plot_data = data if start is None else data[data.index >= start]
        if plot_data.notna().sum() == 0:
            continue  # noting here to plot ...

        _, ax = plt.subplots()
        ax.bar(plot_data.index, plot_data.values, width=width)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ticker.AutoLocator))
        finalise_plot(
            ax,
            title=title,
            ylabel=unit,
            rfooter=f"{tudds.source}-{tudds.table}-{series_id}",
            lfooter=f"{series_type} series.",
            tag=f"{start}-{series_id}",
        )


def x_line_plot(selected: pd.DataFrame, tudds: Tudds) -> None:
    """Produce a single-series line plot."""

    assert len(selected) == 1
    row = selected.index[0]
    series_id = selected.at[row, metacol.id].strip()
    series_type = selected.at[row, metacol.stype].strip()
    source = f"{tudds.source}-{tudds.table}-{series_id}"

    if series_id not in tudds.data[tudds.table]:
        # This should never happen ...
        print(f"Check: {source}, which was not found in DataFrame.")
        return

    data = series_fix(tudds.data[tudds.table][series_id])
    unit = tudds.unit

    data, unit = recalibrate(data, unit)

    if len(data) == 0:
        return

    title = build_title(selected)
    line_plot(
        data=data,
        starts=PLOT_TIMES,
        title=title,
        ylabel=unit,
        rfooter=source,
        lfooter=f"{series_type} series.",
        tags=series_id,
        y0=True,
    )




def plot_all_zip_files(
    cat_id,
    verbose=False,
    test_mode=False,
) -> None:
    """Find all of the appropriate zip files on the ABS landing
    page for a specific catalog identifier, and then for each
    zip file call plot_one_zip_file() to get plot th."""

    def rule_if(verbose: bool, on: bool = False):
        if verbose:
            print(f"{'vvv' if on else '^^^'}==================== {cat_id if on else ''}")
    rule_if(verbose, on=True)

    zip_suffix = ".zip"
    if cat_id not in LINK_DICT:
        print(f"Cannot find the ABS catalogue identifier: {cat_id}.")
        rule_if(verbose)
        return

    _, landing_page = LINK_DICT[cat_id]
    topic = landing_page.topic.replace("-", " ").title()

    links = get_data_links(landing_page)  # get_data_links is cached
    if zip_suffix not in links:
        print(
            f"Odd: cannot find a zip list for {cat_id}, on topic: {topic}",
        )
        rule_if(verbose)
        return

    zip_links = links[zip_suffix]
    zip_list = [
        i
        for i, link in enumerate(zip_links)
        # manage some of the oddities at ABS ...
        if ("cube" not in link.lower() or cat_id in ("3401.0"))  # exclude data cubes
        and "pivot_table" not in link.lower()  # exclude pivot tables
        # --- following relates to 8701.0
        and "estimated%20dwelling%20stock" not in link.lower()
        # --- following all relate to 8731.0 - building approvals
        and "statistical%20area" not in link.lower()  # small area data cubes
        and "local%20government" not in link.lower()  # small area data cubes
        and "_lga_" not in link.lower()  # small area data cubes
        and "_sa2_" not in link.lower()  # small area data cubes
        and "geopackage" not in link.lower()  # small area data cubes
    ]

    if not zip_list:
        print(f"Odd: no relevant zip file for {cat_id}, on topic: {topic}")
        rule_if(verbose)
        return

    if verbose:
        print(f"About to plot: {cat_id} on {topic}")
        print("Zip files on ABS wesbite to be captured as follows ...")
        for i, a in enumerate(zip_links):
            print(a, i in zip_list)

    for zip_table_num in zip_list:
        if verbose:
            print(f"{zip_table_num}: Working on zip-file: {zip_links[zip_table_num]}")
        plot_one_zip_file(cat_id, zip_table_num, verbose, test_mode)

    rule_if(verbose)


# --- And run ...
def check_flags(argv: list[str]) -> dict[str, bool]:
    """Let's first scan CLI arguments for any run-time flags."""

    flag_set = {
        "h": "help",
        "v": "verbose",
        "t": "test",
        "a": "all",
    }

    flags: dict[str, bool] = {}
    good: dict[str, str] = {}
    for flag, alternate in flag_set.items():
        flags[flag] = False
        good[f"-{flag}"] = flag
        good[f"-{alternate}"] = flag
        good[f"--{alternate}"] = flag

    for arg in argv[1:]:
        if arg[0] != "-":
            continue
        if arg in good:
            flags[good[arg]] = True
        else:
            print(f"Unknown run-time flag: {arg}")
            flags["h"] = True

    return flags


def main(argv: list[str]) -> None:
    """Main function: Execution starts here."""

    if len(argv) == 1:
        give_assistance()
        return

    flags = check_flags(argv)

    if flags["h"]:
        give_assistance()
        return

    if flags["t"]:
        print(
            "In test mode: ABS data will be downloaded, but no charts will be generated."
        )

    if flags["a"]:
        for cat_id in LINK_DICT:
            plot_all_zip_files(cat_id, verbose=flags["v"], test_mode=flags["t"])
        return

    for arg in argv[1:]:
        if arg[0] == "-":
            continue
        plot_all_zip_files(arg, verbose=flags["v"], test_mode=flags["t"])
        time.sleep(2)  # just to give the ABS web site a breather.


if __name__ == "__main__":
    main(sys.argv)
