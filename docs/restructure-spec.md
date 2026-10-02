# Restructure spec: notebooks to a Python package

Status: agreed; in progress. Done 2026-10-02: steps 1, 3, 3a, the step 5 pilot (6302)
and conversions of 6345, 6427, 6467 and 6401 (CPI measures and expenditure classes),
with the step 4 pieces they need. Next: 6202 Labour Force, then 5206; the inflation
topic module (CPI against other measures, the 6484 splices, Phillips curves, nominal
GDP, misery index) waits for their unemployment, GDP and population getters.
Package: `au_econ` (project and GitHub repository `au-econ`).

## 0. Decisions

| # | Decision | Status |
|---|----------|--------|
| D1 | Package lives in the same repo, under `~/au-econ/src/` | agreed |
| D2 | Names: GitHub repository and `pyproject` project `au-econ`; import package `au_econ` | agreed |
| D3 | Layers: `sources`, `series`, `analysis`, `charting`, `releases`, `topics` | agreed |
| D4 | Charts written to `~/au-econ/CHARTS/` | agreed |
| D5 | Chart folders are named by the run set the command selected: `CHARTS/<module folder>/` for a release run (or `--all`), `CHARTS/<topic>/<module folder>/` for a topic run, where the module folder is `<first release name> - <TITLE>` (e.g. `6302 - Average Weekly Earnings`); modules may add subfolders to group similar charts (section 6) | agreed |
| D6 | Entry point `run.py <run set>`, one run set per command; each module declares its own names (`RELEASE`) and the shared broad words it joins (`TOPICS`); a run set is either kind, and topics overlap | agreed |
| D7 | No `SHOW`; modules only write files | agreed |
| D8 | Narrow and broad run sets, enforced by the runner: a release number (`6202`) and a short release name (`lfs`) go in `RELEASE` and belong to one module only; broad words (`jobs`, `inflation`) go in `TOPICS`, must be listed with their meaning in the shared `run_sets.TOPICS`, and gather related modules, including topic modules | agreed |
| D9 | `sdmxabs` used for the CPI hierarchy codelist only; nothing else | agreed |
| D10 | No notebook fallback: `run.py` runs converted modules only; notebooks run as they do now until converted | agreed |
| D11 | No PyMC (or arviz, jax, numpyro) in this project; Bayesian work belongs in MacroModels | agreed |
| D12 | One entry point (`run.py`), no collection of run shell scripts | agreed |
| D13 | Functions, not classes, unless a class is plainly the obvious model | agreed |
| D14 | Module shape: `fetch()` + chart functions + `CHARTS` tuple; no `main()` (section 4) | agreed |
| D15 | `--charts <name>...` runs selected charts only, without clearing, within the modules selected by run set. Exact matching on names listed beside each chart in `CHARTS`; short economic names (`u`, `pi`) come from one shared list and may cover several measures (section 7) | agreed |
| D16 | No test suite for now: build first, chart production is the test; runner behaviour checked by hand in the pilot | agreed |
| D17 | The old world (`notebooks/`, its helpers, caches, keys and scripts) is frozen: never moved, trimmed or repointed. Everything is recreated in the package, which keeps its own keys and caches at the root. When all of it works, the old world is deleted in one go | agreed |

## 1. Goals and non-goals

Goals
- Logic lives in an installed package that ruff and mypy check directly.
- One function per concept, written once, used by every chart that needs it.
- One command produces the charts for a release or topic: `uv run run.py 6202`.
  `run.py` is the only way to run things: no per-job shell scripts. Scheduled jobs
  (launchd) call `uv run run.py <name>` directly.
- File locations (charts, caches, keys) do not depend on the working directory.
- Migration is a rebuild beside a frozen old world (D17): notebooks and modules
  coexist, the notebooks keep working untouched, and the old world is deleted in
  one go once the new one does everything it did.

Non-goals
- No change to chart content, titles or styling during migration. A converted
  module must reproduce its notebook's charts.
- No changes to the old world (D17): its caches, keys and input data stay where
  they are; the package has its own copies (section 5).
- No cleanup of stale folders (old `.ipynb_checkpoints/`). The micromamba setup
  was removed separately on 2026-10-02.
  Separate piece of work.
- No new charts or analysis bundled into migration steps.
- No Bayesian modelling (PyMC, arviz, jax, numpyro). That lives in MacroModels.

## 2. Repository layout (end state)

```
~/au-econ/
  pyproject.toml            # gains [build-system]; package installed editable by uv sync
  run.py                    # thin CLI wrapper, calls au_econ.runner.main()
  CHARTS/                   # all chart output (D4)
  LOGS/                     # launchd logs (exists)
  KEYS/                     # API keys: fred.api, EIA-API-KEY.txt (gitignored)
  CACHE/                    # http_cache downloads (gitignored)
  .readabs_cache/ .sdmxabs_cache/   # reader caches (gitignored)
  docs/                     # this spec
  src/au_econ/
    __init__.py
    paths.py                # every filesystem location, anchored to the project root
    runner.py               # discovery, run-set and chart selection, execution
    variables.py            # shared short economic names for --charts (u, pi, y, ...)
    run_sets.py             # shared broad words (topics) for run sets (wages, jobs, ...)
    sources/                # one provider each; fetch + cache; no combining
      http_cache.py  abs.py  rba.py  bis.py  fred.py  oecd.py  yahoo.py
      worldbank.py  aip.py  ...   # section 3, "Sources, shared series and caching"
    series/                 # economic concepts; may combine sources
      gdp.py  prices.py  population.py  labour.py  productivity.py  nom.py  rates.py  ...
    analysis/               # transforms: decompose, henderson, political epochs
    charting/               # mgplot helpers, inflation backplane, footer/source helpers
    releases/
      abs/  rba/  ...       # one module per publication
    topics/                 # cross-source chart sets
```

`notebooks/` (the old world) is not part of the end state: it stays untouched until
it is deleted (D17).

Where each old helper's logic is recreated, as modules need it (section 9). The old
helpers themselves stay untouched:

| Old world (`notebooks/`) | Package equivalent |
|---|---|
| `common.py` | `sources/http_cache.py` |
| `abs_structured_capture.py` | `sources/abs.py` (or `sources/abs_structured.py`) |
| `abs_helper.py` | split: fetch part to `sources/abs.py`, chart-dir part removed (runner owns it, section 6), CPI target constants to `charting/` |
| `abs_gdp.py` | `series/gdp.py` |
| `abs_prices.py` | `series/prices.py` |
| `abs_population.py` | `series/population.py` |
| `abs_nom.py` | `series/nom.py` |
| `abs_spliced_series.py` | split: `series/labour.py` (unemployment), `series/productivity.py` |
| `decompose.py`, `henderson.py` | `analysis/` |
| `political.py` | `analysis/` |
| `pymc_helper.py` | not recreated (D11); deleted in step 3a, before D17. It was used only by two `OLD/` notebooks (`Model - Joint NAIRU+r-star`, `Model - Neutral Rate`), which no longer run |
| `abs_plotting.py`, `abs_inflation_backplane.py` | `charting/` |

## 3. Layers and import rules

Imports point down only:

```
releases, topics
      |
   charting      series
      |         /    \
      |   analysis   sources
      |
   (mgplot, readabs, ... third party)
```

- `sources` imports no first-party code except `paths` and `http_cache`.
- `series` may import `sources`, `analysis`.
- `analysis` imports no first-party code.
- `charting` may import `series` only where a chart element needs data
  (the inflation backplane fetches trimmed-mean CPI).
- `releases`, `topics` may import anything below them; never each other.
- A calculation that combines providers (ABS series divided by an RBA series) lives
  in `series/`, in a function named for what the result means.
- Series are selected by description (`find_abs_id` / `select`), never by series ID.
  Existing CLAUDE.md data-handling rules carry over unchanged.

### Sources, shared series and caching

Three rules decide where fetching code goes:

1. **How to get data from a provider lives in `sources/`, one file per provider**:
   URL, API key, caching, parsing. Chart modules never call `requests.get` or
   `pd.read_csv` / `pd.read_excel` on a URL themselves.
2. **A module's `fetch()` only names what it wants** ("RBA table F2", "BIS policy
   rates for these countries") and calls `sources/` to get it. It fetches only the
   module's own data.
3. **A series wanted by two or more modules gets a named getter in `series/`**, and
   chart functions call it directly. For example the cash rate: besides the RBA
   notebooks, seven ABS notebooks call `read_rba_table` / `read_rba_ocr`, so it
   becomes `series/rates.py: get_cash_rate()`.

Providers, as fetched today and as planned:

| Provider | Today | `sources/` file |
|---|---|---|
| ABS | `readabs` (`read_abs_cat`), cached by readabs; `sdmxabs` for the CPI hierarchy only | `abs.py` |
| RBA | `readabs` (`read_rba_table`, `read_rba_ocr`), cached by readabs | `rba.py`: thin wrapper over readabs |
| BIS | `pd.read_csv` straight from a URL | `bis.py`, through `http_cache` |
| FRED | `requests.get` with an API key | `fred.py`: key from `paths.KEYS_DIR`, through `http_cache` |
| OECD | `requests.get` + `pd.read_csv` (SDMX CSV) | `oecd.py`, through `http_cache` |
| Yahoo | `yfinance` | `yahoo.py` |
| World Bank, AIP | `pd.read_excel` of a downloaded file | `worldbank.py`, `aip.py` |
| Mixed (Bonds) | `common.py` `get_file` (14 calls) | already the target pattern; calls move to the relevant provider files |

Whether the BIS, FRED and OECD notebooks cache anything today is not checked; they
call the web directly rather than through the shared cache. Other one-off providers
(ASIC, AFSA, DCCEEW, Home Affairs, DB.nomics, ANGG) get a `sources/` file each when
their notebook is converted.

`fetch()` returns whatever shape suits the module. `AbsRelease` (data dictionary,
metadata, source label, recent date) is the shape for ABS releases, because they
all share it. An RBA module might return a table and its metadata; a FRED module a
DataFrame of the series it asked for. The only requirement is that every chart
function in the module accepts what `fetch()` returns. A topic module that draws
entirely on `series/` getters may have a `fetch()` that returns `None`, and its
chart functions call the getters.

Caching, two levels:
- **On disk, across runs**: the readabs and sdmxabs caches, and `http_cache` for
  every other provider (`paths.CACHE_DIR`).
- **In memory, within a run**: `functools.cache` on the `series/` getters, so ten
  modules asking for the cash rate in one run fetch it once. Getters return copies,
  so a caller mutating its result cannot corrupt the cache (as `abs_population`
  does now).

## 4. Chart modules (`releases/`, `topics/`)

### Shape

```python
"""Labour Force, Australia (6202.0): headline, state and hours charts."""

# --- dependencies
from mgplot import chart_subdir, line_plot_finalise, multi_start

from au_econ.sources.abs import AbsRelease, fetch_release

# --- module contract
RELEASE = ("6202", "lfs")
TOPICS = ("jobs",)
TITLE = "Labour Force"

# --- constants
TABLE = "62020001"
plot_times = 0, -61
STATES_SUBDIR = "States"


# --- data
def fetch() -> AbsRelease:
    """Fetch the release once; every chart function receives it."""
    return fetch_release("6202.0")


# --- charts
def unemployment(release: AbsRelease) -> None:
    """Unemployment rate: national, SA and trend."""
    ...


def states(release: AbsRelease) -> None:
    """Unemployment rate by state."""
    with chart_subdir(STATES_SUBDIR):
        ...


# --- table of contents, in run order
CHARTS = (
    (unemployment, ("u",)),
    (states, ()),
)
```

Fixed section order: docstring, imports, module contract, constants, `fetch()`,
chart functions, `CHARTS` last.

### Contract

- `RELEASE`: non-empty tuple of lowercase strings, at the top of the module: the
  module's own names, a release number and short release name (`6202`, `lfs`).
  The runner refuses to start if two modules share a release name, so `run.py 6202`
  runs the Labour Force module and nothing else (D8). The first release name also
  starts the module's chart folder name (section 6). Release names and topics become
  folder names, so each must match `[a-z0-9][a-z0-9._-]*`.
- `TITLE`: short readable name (`"Average Weekly Earnings"`, `"Labour Force"`). The
  module's chart folder is `<first release name> - <TITLE>`, so people who do not
  remember the codes can find it. No surrounding spaces, `/` or `:` (Finder shows
  `:` as `/`). Release names are unique, so folder names are too.
- `TOPICS`: tuple of lowercase strings (may be empty): the broad words the module
  joins (`jobs`, `inflation`). Each must be in `run_sets.TOPICS`, a shared dict of
  word to meaning that starts small and gains a word, with its meaning, the first
  time a module uses it. This stops one idea being filed under several words
  (`jobs` / `labour` / `employment`). A topic module that uses the CPI joins
  `inflation`, not `6401`. No word may be both a release name and a topic.
  `--list` shows both; `--topics` prints the shared list.
- `fetch()`: no arguments; returns the data every chart function receives, in
  whatever shape suits the module (section 3). Fetches the module's own data only.
  Called once per run, and only if at least one chart is selected. Fetch
  validation lives here or in the source function it calls (existing rule).
- Chart functions: take exactly the value `fetch()` returns, return `None`, write
  chart files. Raise on failure; never swallow exceptions. A chart function's name
  is always one of its `--charts` names (section 7), so it names the subject
  (`unemployment`, `states`, `deflators`), not the action (`plot_states`).
- `CHARTS`: tuple of `(chart function, extra names)` pairs in run order. Extra names
  are short economic names from the shared list (`variables.py`, section 7); the
  function's own name is implicit and not repeated. The runner iterates `CHARTS`;
  there is no `main()`. It is the module's table of contents: reading it tells you
  everything the module produces and what each chart answers to.
- No module-level work beyond constants and definitions: importing a module must
  not fetch data. Discovery imports every module to read its contract, and does so only
  after setting the cache environment variables (section 7).
- No `SHOW`, no `show=` arguments.

### Coding practice

- **Functions, not classes.** A class only where it is plainly the most obvious way
  to model the problem. Data stored on `self` and shared between methods recreates
  the cross-cell-variable problem.
- **Data passes as an argument.** The `fetch()` result replaces the notebook globals
  `abs_dict`, `meta`, `source`, `RECENT`. `AbsRelease` is a small frozen dataclass
  holding those four (a data container, which is the obvious use of a class).
  A chart function sees only what it is given.
- **Data from other releases comes from `series/`.** A chart that needs the CPI or
  population calls the cached getter (`get_cpi()`) itself rather than having
  `fetch()` gather it.
- **Small private helpers** (leading underscore) sit above the chart functions that
  use them. Logic shared by two modules moves to `charting/` or `series/`.
- **Large releases become a subpackage, one file per chart subfolder.** For example
  `releases/abs/national_accounts_5206/` with `deflators.py`, `productivity.py`,
  `savings.py`; each file holds its chart functions; `__init__.py` holds `RELEASE`,
  `TOPICS`, `fetch()` and a `CHARTS` tuple gathering them
  (`CHARTS = (*deflators.CHARTS, *productivity.CHARTS, ...)`). Chart function names
  must be unique across the whole module, since each is a `--charts` name; the
  runner checks. Short economic names may repeat (several charts can answer to `pi`).

Module file naming: `<short_name>_<catalogue>.py`, e.g. `labour_force_6202.py`
(identifiers cannot start with a digit or contain dots).

## 5. Paths (`paths.py`)

One module defines every location, anchored to the project root found from the
file's own position (`Path(__file__).resolve().parents[2]`), never the cwd.

```python
PROJECT_ROOT
CHARTS_DIR      = PROJECT_ROOT / "CHARTS"
LOGS_DIR        = PROJECT_ROOT / "LOGS"
KEYS_DIR        = PROJECT_ROOT / "KEYS"             # fred.api, EIA-API-KEY.txt
CACHE_DIR       = PROJECT_ROOT / "CACHE"            # http_cache downloads
READABS_CACHE   = PROJECT_ROOT / ".readabs_cache"
SDMXABS_CACHE   = PROJECT_ROOT / ".sdmxabs_cache"
```

Nothing points into `notebooks/` (D17), so deleting the old world cannot break the
package. The keys in `KEYS/` are copies of the old world's (made 2026-10-02); all four
locations are gitignored. The caches fill on first use; anything the old caches held
is refetched once. An input-data constant is added when a module first needs an
input file (today only `OLD/` notebooks read `ABS-Data/` and `govt-budget/`).

Third-party caches:
- **readabs** reads `READABS_CACHE_DIR` from the environment once, at import
  (`download_cache.py:22`, default `./.readabs_cache`). `runner.py` sets it to
  `READABS_CACHE` before importing any chart module. Notebooks run from
  `notebooks/` and keep using their own cache there.
- **sdmxabs** is used for exactly one thing: the CPI hierarchy codelist (D9).
  Only `sources/abs.py` may import it. Three notebooks use it today; the
  replacement column says how the package covers each (the notebooks themselves
  are left alone):

  | Notebook | What sdmxabs supplies | Replacement |
  |---|---|---|
  | `ABS-SDMX-Monthly-Labour-Force-6202` | LF, LF_HOURS, LF_UNDER flows: headline, hours, underemployment | Same series are in the 6202.0 spreadsheets that `ABS Monthly Labour Force 6202` already reads. Notebook looks like an SDMX experiment duplicating it; not recreated, after checking no chart is unique to it |
  | `ABS-SDMX-Monthly-Household-Spending-Indicator-5682` | HSI_M / HSI_Q flows; state ERP via `fetch_state_pop` | 5682.0 is already read by `readabs` in two notebooks (`ABS Monthly+Quarterly Household Spending`, `ABS Real Household Spending per Adult`); state ERP from `series.population`. Still to check: that the state-by-category monthly series are in the spreadsheets |
  | `ABS Inflation multi-measure` | The CPI `INDEX` codelist: parent links of group / sub-group / class (cached 14 days in `CACHE/ABS_cpi_hierarchy/`) | None: no spreadsheet equivalent. Stays on `sdmxabs.code_list_for("CPI", "INDEX")`, recreated in `sources/abs.py` |

  sdmxabs reads `SDMXABS_CACHE_DIR` from the environment at import
  (`download_cache.py:19`, default `./.sdmxabs_cache`), the same mechanism as
  readabs. The runner sets it to `SDMXABS_CACHE`, alongside `READABS_CACHE_DIR`.
  sdmxabs is the user's own package, so it can be changed if anything further is needed.

## 6. Charts

Location: `~/au-econ/CHARTS/` (D4), for converted modules. Notebooks are left alone:
they keep writing to `notebooks/CHARTS/`, and nothing there is moved or edited.
`notebooks/CHARTS/` goes when the old world is deleted.

### Folder scheme (D5)

One run set per command, and it names the folder. Each module's own folder is
`<first release name> - <TITLE>`, so `run.py 6302` and `run.py awe` share a folder:

| Command | Charts go to |
|---|---|
| `run.py 6302` / `run.py awe` | `CHARTS/6302 - Average Weekly Earnings/` |
| `run.py wages` | `CHARTS/wages/6302 - Average Weekly Earnings/`, `CHARTS/wages/6345 - Wage Price Index/`, ... (one subfolder per module, so chart file names from different modules cannot collide) |
| `run.py --all` | `CHARTS/<module folder>/` for every module |

Modules declare no folder: it is built from `RELEASE` and `TITLE`, both literals, so
nothing is looked up (no ABS catalogue fetch) when a module is imported. The code
comes first so folders sort by code; the title is there for people who do not
remember the codes. Topic folders are the bare topic word (`wages/`).

The same charts can exist in two places: after `run.py 6302` and `run.py wages`, the
AWE charts are in `CHARTS/6302 - Average Weekly Earnings/` and in
`CHARTS/wages/6302 - Average Weekly Earnings/`, each from its own run. Each folder is
simply the output of the command that names it.

Where a module produces many charts, it groups similar ones in subfolders with
`mgplot.chart_subdir()` (as `5206`, `6432`, `6150` and the inflation notebook do now:
`Deflators/`, `Productivity/`, `ExpenditureClasses/`). Subfolder names are named
constants in the module.

### Clearing

A full run (no `--charts`) deletes image files before drawing, recursively,
including subfolders:
- a release run (or `--all`): each module's own folder, `CHARTS/<module folder>/`;
- a topic run: the whole topic folder, `CHARTS/<topic>/`, so a module that has left
  the topic leaves no stale subfolder behind.

A folder is only ever filled by the command that names it, and that command redraws
everything in it, so clearing cannot delete charts it will not replace. A `--charts`
run clears nothing: selected charts overwrite their own files.

Recursive clearing replaces today's per-subfolder `chart_subdir(..., clear=True)`.
mgplot's `clear_chart_dir()` only clears the top level, so a subfolder a notebook
stops writing to (after a rename, say) keeps stale charts indefinitely. The runner
does the recursive clear itself (image extensions only, as mgplot does); modules
call `chart_subdir(name)` without `clear=`.

Chart footers, title style and colour conventions carry over unchanged.

## 7. `run.py` and the runner

### Usage

```
uv run run.py 6202                # one name
uv run run.py jobs                # a topic: every module that joined it
uv run run.py --list              # table: module, release, topics
uv run run.py --all               # every module
uv run run.py jobs --charts u     # unemployment-rate charts in the jobs modules
uv run run.py lfs --charts u      # the same, in the LFS module only
uv run run.py 5206 --charts deflators productivity
uv run run.py --all --charts pi   # every inflation chart in every module
uv run run.py --variables         # the shared list of short economic names
uv run run.py --topics            # the shared list of broad words
uv run run.py jobs --list         # the charts in the selected modules
```

`run.py` at the project root is a few lines calling `au_econ.runner.main()`.
All logic is in `runner.py` so it is linted and typed.

### Name resolution

- Names are case-insensitive.
- One run set per command (or `--all`); a second name is an error.
- A run set selects every module whose `RELEASE` or `TOPICS` contains it (topics
  overlap by design; release names never do).
- An unknown name is an error, with close-match suggestions (`difflib`); nothing
  runs.
- Before running, the runner prints the modules selected and why
  (`jobs -> labour_force_6202, ...`).

### Chart selection (`--charts`)

- **Scoped by run set.** `--charts` filters within the modules the run set
  selected; it never selects modules itself. `--charts` with no run set is an
  error; `--all --charts u` is the way to search every module.
- **Exact matching**, case-insensitive. A chart answers to its function name and
  to the extra names beside it in `CHARTS`. No prefixes or partial matches:
  `u` never selects `underemployment`.
- **Shared short names.** `src/au_econ/variables.py` holds one dict of short
  economic names and their meanings:

  ```python
  VARIABLES = {
      "u": "Unemployment rate",
      "pi": "Inflation (any measure)",
  }
  ```

  - One name may cover several measures: headline, trimmed mean and weighted
    median charts can all answer to `pi`.
  - ASCII only (`pi`, not `π`), so names type easily at the shell.
  - Every extra name in any module's `CHARTS` must be in `VARIABLES`; the runner
    refuses to start otherwise. This catches typos and stops a name meaning
    different things in different modules.
  - The list starts empty and grows as modules are converted: a name is added
    the first time a chart uses it, with its meaning.
  - A short name must not equal any chart function name, so a name cannot be both.
- A module with no matching chart is skipped entirely: not fetched, folder not touched.
- A name matching nothing in the selected modules is an error, listing the chart
  names available there; nothing runs.
- The selected charts are printed before running, as module selection is.
- Without `--charts`, every chart in `CHARTS` runs.

### Execution

Once, before any chart module is imported: select matplotlib's `Agg` backend, and
set `READABS_CACHE_DIR` and `SDMXABS_CACHE_DIR`.

`Agg` because the runner only writes files (D7) and never needs a window: it is the
renderer Jupyter's inline backend uses, so modules reproduce their notebooks' charts
exactly, and it works in launchd jobs, which have no window session. Under the macOS
default backend the 6302 pilot's charts came out shifted by a few pixels throughout
(found 2026-10-02; the cause inside that backend was not traced). Notebooks that
import `au_econ` keep their own backend: the runner sets it, not the package.

Full topic run only: clear image files in the whole topic folder (section 6).

For each selected module, in a stable order (sorted by module path):
1. point mgplot at the module's folder for this command (section 6),
2. full release run (or `--all`) only: clear image files in the folder and all its
   subfolders. A `--charts` run does **not** clear: it would delete the charts it
   did not redraw. Selected charts overwrite their own files,
3. call `fetch()` once,
4. call each selected chart function with the result, in `CHARTS` order; a failing
   chart is recorded (exception and traceback) and the next chart still runs,
5. record success or failure per chart; continue to the next module.

A failure in `fetch()` fails all of that module's charts.

At the end: a summary per module, listing each failed chart with its exception type
and message (ok modules on one line),
then exit code 0 if all succeeded, 1 otherwise. Output goes to stdout/stderr;
launchd redirects it to `LOGS/` as it does today. No logging framework.

### Unconverted notebooks (D10)

`run.py` knows only converted modules. Every notebook keeps running as it does now
(Jupyter, or `nbconvert`), converted or not, until the old world is deleted. The
launchd job keeps calling `yahoo-commodities-update.sh` until then; switching it to
`uv run run.py yahoo` (the plist sets the working directory) is part of the final
step.

## 8. Packaging and tooling

`pyproject.toml`:
- add `[build-system]` using uv's build backend, with
  the project name changed from `abs` to `au-econ`, from which uv derives the
  import package `au_econ` (no `module-name` setting needed);
  `uv sync` then installs the package editable, importable from any directory and
  from notebook kernels.
- `[tool.ruff] src = ["src", "notebooks", "."]` during migration; `notebooks`
  drops out when the old world is deleted.
- Notebook-only ruff ignores (`E402`, `B018`, `BLE001`, `INP001`, `PLR0913`, `S101`
  for `*.ipynb`) stay scoped to notebooks; package code gets the full rule set.
- mypy runs on `src/` directly; nbqa stays for notebooks.
- The migration adds no shell scripts (D12). Existing scripts: `yahoo-commodities-update.sh`
  is retired in the final step (section 7). The four
  `notebooks/*-all.sh` lint scripts were deleted on 2026-10-02; lint and type checks
  are plain `uv run ruff ...` / `uv run mypy ...` (and `nbqa` for notebooks). The
  rest (`uv-upgrade.sh`, `test-*.sh`) are untouched by this work.
- Dependencies `pymc`, `arviz`, `jax`, `numpyro`, `graphviz` are removed from
  `pyproject.toml` (D11), with `notebooks/pymc_helper.py`. Nothing outside `OLD/`
  imports them. Separate step, so the lock-file change is reviewed on its own.
- No test suite for now (D16): build first; chart production is the test. Runner
  behaviour that chart images cannot show is checked by hand at its first real use
  (step 5): partial runs leave other charts in place, `u` does not select
  `underemployment`, unknown names are refused, the readabs cache in use is
  the root `.readabs_cache` (no new cache appears under `notebooks/` or elsewhere). pytest can be
  added later if something proves fragile.

## 9. Migration plan

Each step is independently reviewable and leaves everything runnable. No step
before the last edits, moves or deletes anything in the old world (D17).

| Step | Change | Touches | Verification |
|---|---|---|---|
| 1 | Package skeleton: `pyproject` build-system, `src/au_econ/__init__.py`, `paths.py`; `.gitignore` gains root `CHARTS/**`, `KEYS/`, `CACHE/`; keys copied into `KEYS/` | 4 files + 2 key copies | `uv sync`; import from root and from `notebooks/`; paths resolve the same from both; `git check-ignore` covers keys and caches |
| 3 | `runner.py` + `run.py` (no real modules yet) | 2 files | ruff, mypy; `--list` and `--variables` run (empty); first real use is step 5 |
| 3a | Drop PyMC stack: remove `pymc`, `arviz`, `jax`, `numpyro`, `graphviz` and `pymc_helper.py` | `pyproject.toml`, `uv.lock`, 1 file | `uv sync`; ruff/mypy clean; no import errors in remaining notebooks |
| 4 | Recreate the helpers' logic in `sources/series/analysis/charting`, one at a time, as the modules being converted need it; no notebook edits | new package files only | ruff, mypy on `src/` |
| 5 | Pilot conversion: recreate the small `6302` notebook (Average Weekly Earnings, 2 charts, needs only `sources/abs.py`) as `releases/abs/average_weekly_earnings_6302.py`; the notebook stays | 1 module + `sources/abs.py` | Image comparison (section 10); hand checks of runner behaviour (section 8) |
| 6+ | Recreate further notebooks, one per step, user's choice of order; merge duplicate functions as each is rebuilt | per step | Image comparison |
| last | Delete the old world: `notebooks/`, the shell scripts it uses, notebook-only `pyproject.toml` settings (nbqa, `*.ipynb` ignores, `src = "notebooks"`); switch launchd to `run.py`; rewrite CLAUDE.md for the new layout | old world, config | `run.py --all` succeeds after the deletion |

## 10. Verifying a conversion

A converted module passes only if it reproduces its notebook's charts:
- Run the notebook and the module the same day (data is fetched live). The notebook
  writes to `notebooks/CHARTS/`, the module to `CHARTS/`, so neither run clears the
  other's output.
- Same set of file names in both chart folders.
- Each pair of PNGs identical pixel-for-pixel (compare decoded pixels, not file
  bytes: PNG metadata differs between runs).
- Both sides drawn with the same backend: run the notebook through Jupyter
  (inline, Agg-based) and the module through `run.py` (Agg). A module drawn any
  other way can differ by a few pixels everywhere for reasons unrelated to the code.
- Any differing chart is investigated, not accepted.
- A deliberate improvement (e.g. `rfooter=source` replacing a literal footer) is a
  second stage: first prove an exact match with the notebook's behaviour, then
  make the change and confirm the differing pixels are confined to where it shows.

## 11. CLAUDE.md changes (applied at the end, step "last")

- Carry over to modules: data handling, charting conventions, no magic numbers,
  no duplicate code, named window constants, fetch validation, no hardcoded IDs.
- Footer rule, stated 2026-10-02: every chart of Australian data (ABS data is
  essentially all Australian) whose title does not contain "Australia" starts its
  lfooter with "Australia. ". Notebook footers that break it are fixed in a
  conversion's second stage, after the exact match (section 10).
- Series-type rule, stated 2026-10-02: where possible, and unless the legend
  already makes it clear, the lfooter says whether the series is Original,
  Seasonally Adjusted or Trend, and, where it applies, Chain Volume Measures or
  Current Prices. Wording: "Original series.", "Seasonally adjusted." or "Trend.",
  from `charting.footers.SERIES_TYPE_NOTES`. A seasonally adjusted against trend
  chart needs no note.
- Source rule, stated 2026-10-02: the rfooter is the source only, without table
  names. Catalogues from one source are comma-separated after one prefix, different
  sources are separated by a semicolon, and there is no closing punctuation (no
  full stop): `ABS: 6345.0, 6401.0; RBA: F1`.
- Recent window, stated 2026-10-02: for quarterly data, five years
  (`charting.windows.quarterly_plot_times`, `0, -21`: twenty quarters of growth
  plus the quarter it grows from). Modules import it rather than define their own.
  For monthly data, a year and a half (`monthly_plot_times`, `0, -19`), so readers
  can easily look back a year; 25 labelled bars (two years) was tried on 2026-10-02
  and proved too cramped.
- Line widths are left to mgplot (2.0 up to 151 points, 1.0 beyond). `width=` is
  used only to give the lines of a multi-line chart different widths, to highlight
  one.
- Drop for modules (they exist only because of cells): imports-at-top-of-cell,
  no cross-cell variables, one responsibility per cell, watermark cell,
  Restart and Run All, `SHOW`.
- New: layer import rules (section 3), module contract (section 4), paths only via
  `paths.py`, verification by image comparison.
- Notebook rules go with the old world.

## 12. Risks

- **Refetching the readabs cache** if `READABS_CACHE_DIR` (or `SDMXABS_CACHE_DIR`)
  is not set before the package is imported. Mitigated by the runner setting both
  first; checked by hand in step 5.
- **Silent wrong module** from overlapping tags. Mitigated by printing the selection
  before running.
- **Clearing charts a run will not replace.** Prevented by construction: a folder is
  only filled by the command that names it, and a full run of that command redraws
  everything in it (section 6).
- **Two copies of a chart** (`CHARTS/6302 - .../` and `CHARTS/wages/6302 - .../`) from runs on
  different days. Accepted: each folder is the output of its own command.
- **Import side effects**: a module that fetches at import would make `--list` slow
  and fragile. The contract forbids it.
