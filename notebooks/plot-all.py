#!/usr/bin/env python3
# plot-all.py - plot all charts from a specific ABS catalog number.

# --- imports
# system imports
import sys
from dataclasses import dataclass

# analytic imports
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from bs4 import BeautifulSoup as bs

# local imports
from plotting import clear_chart_dir, line_plot, recalibrate_series, set_chart_dir
from abs_data_capture import (
    AbsLandingPage,
    get_abs_data,
    get_fs_constants,
    metacol,
)
import common


# ---
def build_links_dict() -> dict[str, list[str, AbsLandingPage]]:
    """Build a catalogue list using daya from the ABS web page."""

    url = "https://www.abs.gov.au/about/data-services/help/abs-time-series-directory"
    html = common.request_get(url)
    table = bs(html, features="lxml").select("table")[-1]

    link_dict = {}
    for row_num, row in enumerate(table.findAll("tr")):
        if "Ceased" in row.text or row_num < 2:
            continue
        for i, cell in enumerate(row.findAll("td")):
            if i == 0:
                link_dict[(cat_id := cell.text)] = []
            if i == 1:
                link_dict[cat_id].append(cell.text)
                lp = AbsLandingPage(*cell.find("a").get("href").rsplit("/", 4)[-4:-1])
                link_dict[cat_id].append(lp)

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

    print("To generate plots: plot_all.py cat#")
    print("Where cat# is an ABS catalogue number.\n")
    print_known_cat_ids()
    print("\nFor this assistance: plot_all.py --help")


# --- plot all the charts for a specific ABS catalogue number
@dataclass(frozen=True)
class Tudd:
    table: str
    unit: str
    dtype: str
    data: pd.DataFrame


def plotall(cat_id: str) -> None:
    """For a cat_id, obtain the ABS data, and then methodically work
    through the metadata and plot every series therein."""

    if cat_id not in LINK_DICT:
        print(f"Cannot find the ABS catalogue identifier: {cat_id}.")
        return

    desc, landing_page = LINK_DICT[cat_id]
    print(desc)
    abs_dict = get_abs_data(landing_page)
    _, chart_dir, _, meta = get_fs_constants(abs_dict, landing_page)
    set_chart_dir(chart_dir)
    clear_chart_dir(chart_dir)
    plt.style.use("fivethirtyeight")
    mpl.rcParams["font.size"] = 10

    # lets group up similar data for plotting ...
    for table in meta[metacol.table].unique():
        sub_set1 = meta[meta[metacol.table] == table]
        data = abs_dict[table]
        for unit in sub_set1[metacol.unit].unique():
            sub_set2 = sub_set1[sub_set1[metacol.unit] == unit]
            for dtype in sub_set2[metacol.dtype].unique():
                sub_set3 = sub_set2[sub_set2[metacol.dtype] == dtype]
                plot_subset(sub_set3, Tudd(table, unit, dtype, data))


def plot_subset(subset: pd.DataFrame, tudd: Tudd) -> None:
    """Determine how subsets should be plotted."""

    for did in subset[metacol.did].unique():
        selected = subset[subset[metacol.did] == did]
        if len(selected) == 1:
            x_line_plot(selected, tudd)


def title_fix(title: str) -> str:
    """Split unusually long titles over two lines."""

    title = title.strip().replace(" ;", ";")
    title = title.replace("  ", " ")
    title = title.replace("  ", " ")

    do_something = 80
    if (length := len(title)) > do_something:
        positions = [pos for pos, char in enumerate(title) if char == " "]
        if len(positions) > 0:
            breaks = {abs(int(length / 2) - p): i for i, p in enumerate(positions)}
            break_point = positions[breaks[min(breaks.keys())]] + 1
            left, right = title[:break_point], title[break_point:]
            title = f"{left.strip()}\n{right.strip()}"

    return title


def series_fix(series: pd.Series) -> pd.Series:
    """Remove leading and trailing NaNs from a series."""

    first_idx = series.first_valid_index()
    if first_idx is None:
        return pd.Series()
    last_idx = series.last_valid_index()
    return series.loc[first_idx:last_idx]


def x_line_plot(selected: pd.DataFrame, tudd: Tudd) -> None:
    """Produce a single line plot."""

    assert len(selected) == 1
    row = selected.index[0]
    series_id = selected.at[row, metacol.id]
    data = series_fix(tudd.data[series_id])
    unit = tudd.unit

    # need to think about recalibrate below
    # seems to be problematic in rare circumstances
    # data, unit = recalibrate_series(data, unit)

    if len(data) == 0:
        return
    title = title_fix(selected.at[row, metacol.did])
    tdesc = title_fix(selected.at[row, metacol.tdesc])
    title = f"{tdesc}\n{title}"
    stype = selected.at[row, metacol.stype]

    line_plot(
        data=data,
        title=title,
        ylabel=unit,
        rfooter=f"ABS {selected.at[row, metacol.cat]}-{tudd.table}-{series_id}",
        lfooter=f"{stype} series.",
        tags=f"{selected.at[row, metacol.stype]}-{series_id}",
    )


# --- And run ...
if __name__ == "__main__":
    if len(sys.argv) == 1:
        give_assistance()

    for i, arg in enumerate(sys.argv):
        if i == 0:
            continue

        if i in ("--help", "-h"):
            give_assistance()
            break

        plotall(arg)
