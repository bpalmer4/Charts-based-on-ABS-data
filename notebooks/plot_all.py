#!/usr/bin/env python3
"""plot_all.py:
This is a CLI-tool to plot all charts for a specific ABS catalog number.
To allow for scanning a large volume of data."""

# --- imports
# system imports
import sys
from dataclasses import dataclass
import time
from typing import cast

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

    print("To generate plots: plot_all.py [-v -t -a] cat#")
    print("Where:\n\tcat# is an ABS catalog number")
    print("\t-v is a flag for verbose feedback")
    print("\t-v is a flag for testing without actually plotting")
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
    data: dict[str, pd.DataFrame]
    source: str


def plot_all_in_zip(cat_id: str, zip_table: int) -> None:
    """For a cat_id, obtain the ABS data, and then methodically work
    through the metadata and plot every series therein."""

    if cat_id not in LINK_DICT:
        print(f"Cannot find the ABS catalogue identifier: {cat_id}.")
        return

    _, landing_page = LINK_DICT[cat_id]
    abs_dict = get_abs_data(landing_page, zip_table)
    source, chart_dir, _, meta = get_fs_constants(abs_dict, landing_page, " - ALL")
    set_chart_dir(chart_dir)
    clear_chart_dir(chart_dir)
    plt.style.use("fivethirtyeight")
    mpl.rcParams["font.size"] = 10

    # lets group up similar data for plotting ...
    for table in meta[metacol.table].unique():
        sub_set1 = meta[meta[metacol.table] == table]
        for unit in sub_set1[metacol.unit].unique():
            sub_set2 = sub_set1[sub_set1[metacol.unit] == unit]
            for dtype in sub_set2[metacol.dtype].unique():
                sub_set3 = sub_set2[sub_set2[metacol.dtype] == dtype]
                plot_subset(sub_set3, Tudds(table, unit, dtype, abs_dict, source))


def plot_subset(subset: pd.DataFrame, tudds: Tudds) -> None:
    """Determine how subsets should be plotted."""

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
        # TO DO - yhink about percentage change charts


def title_fix(title: str) -> str:
    """Split unusually long titles,
    roughly in equal parts."""

    # Cruncg down where the ABS often adds extra padding ...
    title = title.strip().replace(" ;", ";")
    title = title.replace("  ", " ")
    title = title.replace("  ", " ")

    # split lines roughly in equal parts around spaces.
    do_something_over = 80  # characters
    n_folds = ((length := len(title)) // do_something_over) + 1
    if n_folds > 1:
        for fold in range(1, n_folds):
            positions = [pos for pos, char in enumerate(title) if char == " "]
            breaks = {
                abs(int(length * fold / n_folds) - p): i
                for i, p in enumerate(positions)
            }
            break_point = positions[breaks[min(breaks.keys())]] + 1
            left, right = title[:break_point], title[break_point:]
            title = f"{left.strip()}\n{right.strip()}"

    return title


def build_title(tdesc: str, did: str) -> str:
    """Titles comprise table description and data item descrption."""

    tdesc_fix = title_fix(tdesc)
    did_fix = title_fix(did)
    title = f"{tdesc_fix}\n{did_fix}"
    return title


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

    topic = s_row[metacol.tdesc]
    title = build_title(topic, s_row[metacol.did])
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

    title = build_title(selected.at[row, metacol.tdesc], selected.at[row, metacol.did])

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
    series_id = selected.at[row, metacol.id]
    series_type = selected.at[row, metacol.stype]
    data = series_fix(tudds.data[tudds.table][series_id])
    unit = tudds.unit

    data, unit = recalibrate(data, unit)

    if len(data) == 0:
        return

    topic = selected.at[row, metacol.tdesc]
    did = selected.at[row, metacol.did]
    title = build_title(topic, did)

    line_plot(
        data=data,
        starts=PLOT_TIMES,
        title=title,
        ylabel=unit,
        rfooter=f"{tudds.source}-{tudds.table}-{series_id}",
        lfooter=f"{series_type} series.",
        tags=series_id,
        y0=True,
    )


def plotall(
    cat_id,
    verbose=False,
    test_mode=False,
) -> None:
    """Find all of the appropriate zip files and plot them."""

    zip_suffix = ".zip"
    if cat_id not in LINK_DICT:
        print(f"Cannot find the ABS catalogue identifier: {cat_id}.")
        return

    _, landing_page = LINK_DICT[cat_id]
    topic = landing_page.topic.replace("-", " ").title()
    links = get_data_links(landing_page)  # net get_data_links is cached
    if zip_suffix not in links:
        print(
            f"Odd: no zip list for {cat_id}, on topic:",
            f'{landing_page.topic.replace("-", " ").title()}',
        )
        return
    zip_links = links[zip_suffix]
    zip_list = [
        i
        for i, link in enumerate(zip_links)
        # manage some of the oddities at ABS ...
        if ("cube" not in link.lower() or cat_id in ("3401.0"))  # exclude data cubes
        and "pivot_table" not in link.lower()  # exclude pivot tables
        # --- following all relate to 8731.0 - building approvals
        and "statistical%20area" not in link.lower()  # small area data cubes
        and "local%20government" not in link.lower()  # small area data cubes
        and "_lga_" not in link.lower()  # small area data cubes
        and "_sa2_" not in link.lower()  # small area data cubes
        and "geopackage" not in link.lower()  # small area data cubes
    ]
    if not zip_list:
        print(f"Odd: no relevant zip file for {cat_id}, on topic: {topic}")
        return

    if verbose:
        print("===================")
        print(f"About to plot: {cat_id} on {topic}")
        print("Zip files on ABS wesite captured as follows ...")
        for i, a in enumerate(zip_links):
            print(a, i in zip_list)

    if test_mode:
        print("No plots --- in test mode")
    else:
        for zip_table in zip_list:
            if verbose:
                print(f"{zip_table}: Working on zip-file: {zip_links[zip_table]}")
            plot_all_in_zip(cat_id, zip_table)

    if verbose:
        print("===================")


# --- And run ...
def check_flags(argv: list[str]) -> tuple[bool, bool, bool]:
    """Let's first scan for any run-time flags."""

    verbose = False
    test_mode = False
    all_cat_ids = False
    assisted = False

    for arg in argv:

        if arg[0] != "-":
            continue

        if arg in ("--help", "-h"):
            if not assisted:
                give_assistance()
            assisted = True
            continue

        if arg in ("--verbose", "-v"):
            verbose = True
            continue

        if arg in ("--test", "-t"):
            test_mode = True
            continue

        if arg in ("--all", "-a"):
            all_cat_ids = True
            continue

        print(f"Unknown run-time flag: {arg}")

    return verbose, test_mode, all_cat_ids


def main(argv: list[str]) -> None:
    """Main function: Execution starts here."""

    if len(argv) == 1:
        give_assistance()
        return

    verbose, test_mode, all_cat_ids = check_flags(argv)

    if all_cat_ids:
        for cat_id in LINK_DICT:
            plotall(cat_id, verbose=verbose, test_mode=test_mode)
        return

    for arg in argv[1:]:
        if arg[0] == "-":
            continue
        plotall(arg, verbose=verbose, test_mode=test_mode)
        time.sleep(1)  # just to give the web site a breather.


if __name__ == "__main__":
    main(sys.argv)
