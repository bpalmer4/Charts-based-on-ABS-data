"""Find chart modules, select them by run set and chart name, and run them (`run.py`).

One run set per command, and it names the folder the charts go in: a release run (or
--all) writes each module to CHARTS/<first release name> - <TITLE>/, a topic run to
CHARTS/<topic>/<first release name> - <TITLE>/. A full run first deletes every image in the
folder it is about to fill (for a topic, the whole topic folder). A `--charts` run
deletes nothing, so the charts it does not redraw survive.
"""

import argparse
import difflib
import importlib
import os
import pkgutil
import re
import sys
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import mgplot as mg
from mgplot.settings import IMAGE_EXTENSIONS

from au_econ import paths
from au_econ.run_sets import TOPICS
from au_econ.variables import VARIABLES

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path
    from types import ModuleType

# --- constants
CHART_PACKAGES = ("au_econ.releases", "au_econ.topics")
PACKAGE_PREFIX = "au_econ."
# defining either of these makes a module a chart module, which must then define TOPICS,
# TITLE and CHARTS too; the files inside a large release's subpackage define only CHARTS,
# so they are not picked up
CONTRACT_MARKERS = ("RELEASE", "fetch")
# release names and topics become folder names, so each must be one safe path component
RUN_SET_NAME = re.compile(r"[a-z0-9][a-z0-9._-]*")
# TITLE is the readable part of a folder name; Finder shows ":" and "/" as each other
TITLE_FORBIDDEN = frozenset("/:")
SUGGESTIONS = 3
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_REFUSED = 2

type Failures = list[tuple[str, str]]  # (chart name, exception description)


# --- data containers
@dataclass(frozen=True)
class Chart:
    """A chart function and the names it answers to under --charts."""

    name: str
    function: Callable[[Any], object]
    extras: tuple[str, ...]

    @property
    def names(self) -> frozenset[str]:
        """Return the function's own name plus its extra short names."""
        return frozenset((self.name, *self.extras))


@dataclass(frozen=True)
class ChartModule:
    """A chart module's contract, read and checked at discovery."""

    name: str  # dotted path below au_econ, e.g. releases.abs.labour_force_6202
    release: tuple[str, ...]  # this module's own run-set names; the first names its folder
    topics: tuple[str, ...]  # shared broad words, from run_sets.TOPICS
    title: str  # short readable name, e.g. "Average Weekly Earnings"
    fetch: Callable[[], object]
    charts: tuple[Chart, ...]

    @property
    def run_sets(self) -> tuple[str, ...]:
        """Return every run set that selects this module."""
        return (*self.release, *self.topics)

    @property
    def folder_name(self) -> str:
        """Return the module's chart folder name, e.g. "6302 - Average Weekly Earnings"."""
        return f"{self.release[0]} - {self.title}"


# --- discovery
def _set_cache_environment() -> None:
    """Point readabs and sdmxabs at the package's caches; both read these once, at import."""
    os.environ["READABS_CACHE_DIR"] = str(paths.READABS_CACHE)
    os.environ["SDMXABS_CACHE_DIR"] = str(paths.SDMXABS_CACHE)


def _import_all() -> list[ModuleType]:
    """Import every module under the chart packages (importing a module must not fetch data)."""
    found: list[ModuleType] = []
    for package_name in CHART_PACKAGES:
        package = importlib.import_module(package_name)
        found.extend(
            importlib.import_module(info.name)
            for info in pkgutil.walk_packages(package.__path__, prefix=f"{package_name}.")
        )
    return found


def _read_names(
    name: str, attribute: str, value: object, problems: list[str], *, allow_empty: bool
) -> tuple[str, ...] | None:
    """Read RELEASE (non-empty) or TOPICS (may be empty): a tuple of run-set names."""
    if (
        isinstance(value, tuple)
        and (value or allow_empty)
        and all(isinstance(s, str) and RUN_SET_NAME.fullmatch(s) for s in value)
    ):
        return tuple(str(s) for s in value)
    kind = "a tuple" if allow_empty else "a non-empty tuple"
    problems.append(f"{name}: {attribute} must be {kind} of names matching {RUN_SET_NAME.pattern}")
    return None


def _read_title(name: str, value: object, problems: list[str]) -> str | None:
    """Read TITLE: a short readable name, safe as part of a folder name."""
    if isinstance(value, str) and value and value == value.strip() and not TITLE_FORBIDDEN & set(value):
        return value
    problems.append(f"{name}: TITLE must be a non-empty string without surrounding spaces, '/' or ':'")
    return None


def _read_fetch(name: str, value: object, problems: list[str]) -> Callable[[], object] | None:
    """Read fetch: a function taking no arguments."""
    if callable(value):
        return value
    problems.append(f"{name}: fetch() is missing")
    return None


def _read_chart(name: str, entry: object, problems: list[str]) -> Chart | None:
    """Read one CHARTS entry: (chart function, tuple of extra short names)."""
    match entry:
        case (function, tuple() as extras):
            function_name = getattr(function, "__name__", None)
            if callable(function) and isinstance(function_name, str) and all(isinstance(e, str) for e in extras):
                return Chart(function_name.lower(), function, tuple(str(e).lower() for e in extras))
    problems.append(f"{name}: CHARTS entry {entry!r} is not (chart function, tuple of names)")
    return None


def _read_charts(name: str, value: object, problems: list[str]) -> tuple[Chart, ...] | None:
    """Read CHARTS: a non-empty tuple of entries whose function names do not repeat."""
    if not (isinstance(value, tuple) and value):
        problems.append(f"{name}: CHARTS must be a non-empty tuple of (chart function, extra names) pairs")
        return None
    read = [_read_chart(name, entry, problems) for entry in value]
    charts = tuple(chart for chart in read if chart is not None)
    if len(charts) < len(read):
        return None
    chart_names = [chart.name for chart in charts]
    repeated = sorted({n for n in chart_names if chart_names.count(n) > 1})
    if repeated:
        problems.append(f"{name}: chart names repeat: {', '.join(repeated)}")
        return None
    return charts


def _read_contract(module: ModuleType, problems: list[str]) -> ChartModule | None:
    """Read and check a chart module's RELEASE, TOPICS, TITLE, fetch and CHARTS."""
    name = module.__name__.removeprefix(PACKAGE_PREFIX)
    release = _read_names(name, "RELEASE", getattr(module, "RELEASE", None), problems, allow_empty=False)
    topics = _read_names(name, "TOPICS", getattr(module, "TOPICS", None), problems, allow_empty=True)
    title = _read_title(name, getattr(module, "TITLE", None), problems)
    fetch = _read_fetch(name, getattr(module, "fetch", None), problems)
    charts = _read_charts(name, getattr(module, "CHARTS", None), problems)
    if release is None or topics is None or title is None or fetch is None or charts is None:
        return None
    return ChartModule(name, release, topics, title, fetch, charts)


def _check_run_sets(modules: list[ChartModule], problems: list[str]) -> None:
    """Check release names belong to one module each, and topics come from run_sets.TOPICS."""
    problems.extend(
        f"TOPICS word {topic!r} must match {RUN_SET_NAME.pattern}"
        for topic in sorted(TOPICS)
        if not RUN_SET_NAME.fullmatch(topic)
    )
    owners: dict[str, str] = {}
    for module in modules:
        for release_name in module.release:
            if release_name in owners:
                problems.append(f"release name {release_name!r} used by {owners[release_name]} and {module.name}")
            if release_name in TOPICS:
                problems.append(f"{module.name}: release name {release_name!r} is also a topic")
            owners.setdefault(release_name, module.name)
        unknown = [topic for topic in module.topics if topic not in TOPICS]
        if unknown:
            problems.append(f"{module.name}: TOPICS not in run_sets.TOPICS: {', '.join(unknown)}")


def _check_variables(modules: list[ChartModule], problems: list[str]) -> None:
    """Check the short names in CHARTS come from VARIABLES and none is a chart name."""
    chart_names = {chart.name for module in modules for chart in module.charts}
    for short in sorted(VARIABLES):
        if short != short.lower() or not short.isascii():
            problems.append(f"VARIABLES name {short!r} must be lowercase ASCII")
        if short in chart_names:
            problems.append(f"VARIABLES name {short!r} is also a chart function name")
    for module in modules:
        for chart in module.charts:
            unknown = [e for e in chart.extras if e not in VARIABLES]
            if unknown:
                problems.append(f"{module.name}.{chart.name}: not in VARIABLES: {', '.join(unknown)}")


def _discover() -> tuple[list[ChartModule], list[str]]:
    """Return every chart module, sorted by module path, and any contract problems."""
    problems: list[str] = []
    modules: list[ChartModule] = []
    for module in _import_all():
        if any(hasattr(module, marker) for marker in CONTRACT_MARKERS):
            contract = _read_contract(module, problems)
            if contract is not None:
                modules.append(contract)
    _check_run_sets(modules, problems)
    _check_variables(modules, problems)
    return sorted(modules, key=lambda m: m.name), problems


# --- selection
def _select_modules(name: str, modules: list[ChartModule]) -> list[ChartModule] | None:
    """Return the modules in the named run set, or None if the name is unknown."""
    known = sorted({run_set for module in modules for run_set in module.run_sets})
    if name not in known:
        close = difflib.get_close_matches(name, known, n=SUGGESTIONS)
        hint = f"; did you mean {', '.join(close)}?" if close else ""
        print(f"unknown run set {name!r}{hint}", file=sys.stderr)
        return None
    selected = [m for m in modules if name in m.run_sets]
    print(f"{name} -> {', '.join(m.folder_name for m in selected)}")
    return selected


def _select_charts(
    wanted: list[str], modules: list[ChartModule]
) -> list[tuple[ChartModule, tuple[Chart, ...]]] | None:
    """Return the charts answering to the wanted names, or None if a name matches nothing."""
    available = sorted({n for module in modules for chart in module.charts for n in chart.names})
    unmatched = [w for w in wanted if w not in available]
    if unmatched:
        print(f"no chart in the selected modules answers to: {', '.join(unmatched)}", file=sys.stderr)
        print(f"available: {', '.join(available)}", file=sys.stderr)
        return None
    plan: list[tuple[ChartModule, tuple[Chart, ...]]] = []
    for module in modules:
        charts = tuple(chart for chart in module.charts if chart.names & set(wanted))
        if charts:
            print(f"{module.folder_name}: {', '.join(chart.name for chart in charts)}")
            plan.append((module, charts))
    return plan


def _plan(
    name: str | None, *, wanted: list[str], modules: list[ChartModule]
) -> list[tuple[ChartModule, tuple[Chart, ...]]] | None:
    """Return the modules and charts to run (name None: --all), or None if refused."""
    if name is None:
        print(f"--all -> {len(modules)} modules")
        selected: list[ChartModule] | None = modules
    else:
        selected = _select_modules(name, modules)
    if selected is None:
        return None
    if not wanted:
        return [(module, module.charts) for module in selected]
    return _select_charts(wanted, selected)


# --- execution
def _describe(exc: Exception) -> str:
    """Describe an exception in one line for the summary."""
    return f"{type(exc).__name__}: {exc}"


def _clear_images(folder: Path) -> None:
    """Delete the image files in a folder and all its subfolders (mgplot clears the top only)."""
    for extension in IMAGE_EXTENSIONS:
        for image in folder.rglob(f"*.{extension}"):
            if image.is_file():
                image.unlink()


def _run_chart(chart: Chart, data: object) -> str | None:
    """Run one chart function; return a description of its failure, or None on success."""
    print(f"  {chart.name}")
    try:
        chart.function(data)
    except Exception as exc:
        traceback.print_exc()
        plt.close("all")  # a chart that failed part-way may leave its figure open
        return _describe(exc)
    return None


def _folder(module: ChartModule, topic: str | None) -> Path:
    """Return a module's chart folder, under CHARTS/ or, for a topic run, CHARTS/<topic>/."""
    base = paths.CHARTS_DIR / topic if topic else paths.CHARTS_DIR
    return base / module.folder_name


def _run_module(module: ChartModule, charts: tuple[Chart, ...], folder: Path, *, clear: bool) -> Failures:
    """Run a module's selected charts into a folder, deleting its images first if asked."""
    print(f"\n{folder.relative_to(paths.PROJECT_ROOT)}")
    mg.set_chart_dir(str(folder))
    if clear:
        _clear_images(folder)
    try:
        data = module.fetch()
    except Exception as exc:
        traceback.print_exc()
        return [(chart.name, f"fetch() failed: {_describe(exc)}") for chart in charts]
    failures: Failures = []
    for chart in charts:
        failure = _run_chart(chart, data)
        if failure is not None:
            failures.append((chart.name, failure))
    return failures


def _summarise(results: list[tuple[ChartModule, int, Failures]]) -> int:
    """Print one line per module (and one per failed chart); return the exit code."""
    print("\nSummary")
    if not results:
        print("  nothing to run")
    for module, count, failures in results:
        if not failures:
            print(f"  ok    {module.folder_name} ({count} chart{'' if count == 1 else 's'})")
            continue
        print(f"  FAIL  {module.folder_name}: {len(failures)} of {count} charts failed")
        for chart_name, message in failures:
            print(f"          {chart_name}: {message}")
    return EXIT_FAILED if any(failures for _, _, failures in results) else EXIT_OK


# --- listings
def _print_table(headings: tuple[str, ...], rows: Sequence[tuple[str, ...]]) -> None:
    """Print rows under headings in aligned columns."""
    widths = [max(len(row[i]) for row in (headings, *rows)) for i in range(len(headings))]
    for row in (headings, *rows):
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip())
    if not rows:
        print("(none yet)")


def _list(name: str | None, *, use_all: bool, modules: list[ChartModule]) -> int:
    """List all modules, or the charts in the modules selected by run set; return the exit code."""
    if name is None and not use_all:
        rows = [(m.folder_name, " ".join(m.release), " ".join(m.topics)) for m in modules]
        _print_table(("module", "release", "topics"), rows)
        return EXIT_OK
    selected = modules if name is None else _select_modules(name, modules)
    if selected is None:
        return EXIT_REFUSED
    chart_rows = [(m.folder_name, chart.name, " ".join(chart.extras)) for m in selected for chart in m.charts]
    _print_table(("module", "chart", "short names"), chart_rows)
    return EXIT_OK


# --- command line
def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse and cross-check the command line."""
    parser = argparse.ArgumentParser(prog="run.py", description="Produce the charts for one run set.")
    parser.add_argument(
        "name", nargs="?", help="one run set: a release number (6202), short name (lfs) or topic (wages)"
    )
    parser.add_argument("--all", action="store_true", help="every chart module, each into its release folder")
    parser.add_argument(
        "--charts", nargs="+", metavar="NAME", help="only these charts, without clearing the folder"
    )
    parser.add_argument("--list", action="store_true", help="list modules; with a run set or --all, their charts")
    parser.add_argument("--variables", action="store_true", help="list the shared short economic names")
    parser.add_argument("--topics", action="store_true", help="list the shared broad words (topics)")
    args = parser.parse_args(argv)
    if args.name and args.all:
        parser.error("give a run set or --all, not both")
    if args.charts and not (args.name or args.all):
        parser.error("--charts filters within a run set: give a run set or --all")
    if not (args.name or args.all or args.list or args.variables or args.topics):
        parser.error("give one run set, or --all, --list, --variables or --topics")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Run the chart modules selected on the command line; return the exit code."""
    args = _parse_args(argv)
    if args.variables or args.topics:
        for wanted, heading, shared in ((args.variables, "short name", VARIABLES), (args.topics, "topic", TOPICS)):
            if wanted:
                _print_table((heading, "meaning"), sorted(shared.items()))
        return EXIT_OK
    # files only, never a window: the macOS default backend lays text out a few pixels
    # differently from Agg (which notebooks use), and launchd jobs have no window session
    mpl.use("Agg")
    _set_cache_environment()  # before discovery imports any chart module, and so readabs
    modules, problems = _discover()
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return EXIT_REFUSED
    name = str(args.name).lower() if args.name else None
    if args.list:
        return _list(name, use_all=args.all, modules=modules)
    wanted = list(dict.fromkeys(str(chart).lower() for chart in args.charts or ()))
    plan = _plan(name, wanted=wanted, modules=modules)
    if plan is None:
        return EXIT_REFUSED
    full = not wanted
    topic = name if name in TOPICS else None
    if full and topic:  # the whole topic folder, so a module that left the topic leaves nothing behind
        _clear_images(paths.CHARTS_DIR / topic)
    results = [
        (module, len(charts), _run_module(module, charts, _folder(module, topic), clear=full and not topic))
        for module, charts in plan
    ]
    return _summarise(results)
