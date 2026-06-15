# Claude Code Project Configuration

## Project Overview
This project contains Python Jupyter notebooks that analyze and visualize economic data from:
- Australian Bureau of Statistics (ABS)
- Reserve Bank of Australia (RBA)
- OECD and BIS

The notebooks fetch the latest data and generate charts for key social and economic statistics.

## Project Structure
- `/notebooks/` - Contains all Jupyter notebooks for data analysis
- `/notebooks/CHARTS/<topic>/` - Output directories for generated charts (set per-notebook via `mg.set_chart_dir()`; nothing writes to a top-level `/charts/`)
- Project uses Python with data analysis libraries

## Development Setup
- Python environment managed with uv
- Virtual environment in `.venv/`
- Designed to work on iPad using the carnets app

## Key Commands
```bash
# Activate virtual environment
source .venv/bin/activate

# Run a notebook
jupyter notebook notebooks/<notebook-name>.ipynb
```

## Coding practice

- Think Before Coding: Don’t assume. Don’t hide confusion. Surface tradeoffs.
- Simplicity First: Minimum code that solves the problem. Nothing speculative.
- Surgical Changes: Touch only what you must. Clean up only your own mess.
- Goal-Driven Execution: Define success criteria. Loop until verified.


## Notebook Hygiene

### Structure and Layout
- **Imports at the top**: All imports in the first code cell(s), grouped and commented:
  stdlib, then third-party (`pandas`, `readabs`), then local (`abs_helper`, `mgplot`).
  Never inline imports inside functions or plotting cells.
- **Pandas display settings after imports**: `pd.options.display.max_rows = 999999` etc.
  in the setup cell.
- **Constants after imports**: Define `SHOW = False`, `plot_times`, `FILE_TYPE`, and other
  configuration in a dedicated cell after imports, before function definitions.
- **Function definitions before use**: Define all plotting/analysis functions before the
  cells that call them.
- **Markdown cells as section headers**: Use markdown to delineate sections
  (Setup, Data Fetch, Plotting, etc.).
- **One responsibility per cell**: Each cell does one thing: fetch data, transform, or plot.
- **Watermark cell at the end**: Use `%watermark` to record Python version, package versions,
  and timestamp.

### Coding Conventions
- Each notebook should be self-contained
- Fetch latest data when run
- Output charts to the notebook's `CHARTS/<topic>/` directory (never a top-level `charts/`)
- Use descriptive names for notebooks indicating the data source and series

### Code Quality
- **Logic in functions, not module level**: Wrap all computation and plotting in functions.
  Call them from clean cells. Module-level code should be limited to imports, constants,
  data fetching, and function calls.
- **No magic numbers**: Use named constants or function parameters, not bare literals.
- **Minimal global state**: Pass data through function arguments and returns rather than
  relying on notebook-scoped variables where practical.
- **No duplicate code across cells**: If you repeat logic, extract it to a function.
- **Consistent variable names across notebooks**: `abs_dict` for the data dictionary,
  `meta` for metadata, `source` for footer attribution, `RECENT` for the latest date,
  `plot_times` for chart time ranges, `table` for table identifiers, `series_id`/`sid`
  for series IDs, `units` for unit strings.

### Data Handling
- **Metadata-driven series selection**: Use `find_abs_id()` with `metacol` selectors
  rather than hardcoding series IDs, which change over time.
- **Recalibrate units before plotting**: Call `ra.recalibrate()` to get human-readable units.
- **`abs_helper.get_abs_data()` called once only**: It resets the chart directory. Use
  `abs_structured_capture` or `ra.read_abs_cat()` for additional data within a notebook.
- **Validate fetched data**: Check for empty DataFrames or unexpected nulls before plotting.
- **COVID year exclusion in decomposition**: Use `ignore_years=(2020, 2021)` when doing
  seasonal decomposition to avoid distortion.

### Charting
- **`SHOW = False` constant**: Define at module level, pass to all plot functions. Enables
  batch execution without chart display.
- **`plot_times` convention**: Define `plot_times = 0, -N` (or `0, RECENT`) for use with
  `multi_start()` to generate full-history and recent-period chart variants.
- **`multi_start()` for paired charts**: Standard pattern to produce both a full-history
  and a recent-period chart for each concept.
- **Every chart needs source attribution**: `rfooter=source` for the data source,
  `lfooter` for geography and series type (e.g., `"Australia. Seasonally Adjusted."`).
- **Consistent title style**: Use colons, not em dashes, in chart titles.

### Reproducibility
- **Restart and Run All before finishing**: Notebooks must execute cleanly top-to-bottom
  with no out-of-order cell dependencies.
- **No leftover debug output**: Remove print statements and temporary cells before committing.

## Notes
- Modified notebooks are currently uncommitted (check git status)
- Main branch is 'main'

## readabs Package Reference

The `readabs` package (source in `~/readabs`) fetches ABS and RBA data. Key patterns:

### Efficient Data Fetching
```python
import readabs as ra
from readabs import metacol as mc

# ALWAYS use single_excel_only to fetch just one table (much faster)
data, meta = ra.read_abs_cat("6401.0", single_excel_only="64010Appendix1a")

# DON'T fetch entire catalog unless needed
# data, meta = ra.read_abs_cat("6401.0")  # Downloads everything - slow!
```

### Finding Series by Description (robust to ID changes)
```python
# Use find_abs_id with metacol for robust series selection
_, series_id, _ = ra.find_abs_id(meta, {
    "64010Appendix1a": mc.table,
    "Index Numbers": mc.did,
    "All groups CPI, seasonally adjusted": mc.did,
})
series = data["64010Appendix1a"][series_id]
```

### metacol Column Names
- `mc.did` - Data Item Description
- `mc.id` - Series ID
- `mc.stype` - Series Type (Original, Seasonally Adjusted)
- `mc.table` - Table name
- `mc.unit` - Unit (Percent, Number, etc.)
- `mc.freq` - Frequency (Monthly, Quarterly)

### abs_helper.get_abs_data() Warning
The `get_abs_data()` function in `abs_helper.py` calls `set_chart_dir()` and `clear_chart_dir()`.
**Do not call it multiple times** - it will reset the chart directory. Use `ra.read_abs_cat()`
with `single_excel_only` instead when fetching additional data within a notebook.

### Key ABS Tables
- CPI Quarterly (Appendix): `64010Appendix1a` in catalog `6401.0`
- CPI Monthly: `640106` in catalog `6401.0`
- WPI: `634501` in catalog `6345.0`
- Labour Force: `62020001` in catalog `6202.0`
- National Accounts: `5206001_Key_Aggregates` in catalog `5206.0`

### Splicing Mixed-Frequency / Multi-Vintage Series (`select` + `splice`)

For a concept spread across frequencies/releases (e.g. monthly CPI back to 2017,
quarterly back to 1948, plus a discontinued indicator covering the gap), use the
four composable splice functions. **Highest priority first**: the first segment
wins on overlap; gaps are left honest (no interpolation unless asked).

| Function | Role |
|----------|------|
| `select_one(data, meta, selector)` | One series from `(data, meta)`; ABS unit kept on `.attrs["unit"]` |
| `select(sources, *, require_same_units=True)` | Iterable of `(data, meta, selector)` → `list[Series]`; **raises on mixed units** unless `require_same_units=False` |
| `splice(segments, *, target=None, rebase=False, agg="mean", output=None, fill=None, name=None)` | Splice ordered series → `(series, report)` |
| `select_and_splice(sources, *, ...same splice kwargs..., require_same_units=True)` | `select` then `splice` (no-transform case) → `(series, unit, report)` |

- A *selector* is the `{search_value: column}` form used by `find_abs_id`
  (`validate_unique=True` → de-dupes on Series ID, raises on real ambiguity).
- Common pattern: a shared `base` selector + per-source frequency override, e.g.
  `base | {"Quarter": mc.freq}` vs `base | {"Month": mc.freq}`.
- **`rebase`** (default `False`): multiplicatively rescales lower-priority
  segments onto the running result's level. ONLY for **ratio-scale / index-like**
  series across reference-period changes (CPI index). WRONG for rates, balances,
  zero-crossing or additive series — leave it off and splice the comparable
  values (e.g. compute Y/Y growth per source, then `splice(..., rebase=False)`).
- `target` = common grid freq (defaults to finest present); `output` = optional
  final resample freq; `agg="mean"` for levels, `"sum"` for flows; `fill` is
  `None`/`"ffill"`/`"interpolate"`.
- The returned `report` DataFrame logs every rebase factor and overlap junction —
  audit it rather than trusting the splice blindly.

```python
# No transform — splice index levels across reference-period changes (rebase=True):
base = {"Index Numbers ;  All groups CPI ;  Australia ;": mc.did, "Index Numbers": mc.unit}
series, unit, report = ra.select_and_splice([
    (cur, cmeta, base | {"Month": mc.freq}),     # new monthly CPI
    (ind, imeta, base | {"Month": mc.freq}),     # discontinued indicator
    (cur, cmeta, base | {"Quarter": mc.freq}),   # long quarterly back to 1948
], output="M", rebase=True)

# With a transform — select, transform each, splice the rates (rebase=False):
m_idx, i_idx, q_idx = ra.select([...])
yoy = lambda s, n: ((s / s.shift(n) - 1) * 100).dropna()
long_yoy, report = ra.splice([yoy(m_idx, 12), yoy(i_idx, 12), yoy(q_idx, 4)], rebase=False)

# Mixed units on purpose (build a rate from counts, splice under the published rate):
unemployed, labour_force, ur_monthly = ra.select([...], require_same_units=False)
ur, report = ra.splice([ur_monthly, unemployed / labour_force * 100], rebase=False)
```

## mgplot Package Reference

The `mgplot` package (source in `~/mgplot`) wraps matplotlib for economic data charting.
**Prefer `*_finalise` functions** for simple single-layer charts.
For composite charts (e.g. fan charts, overlaid fills + lines), layer mgplot functions
with `ax=` chaining, then call `finalise_plot()` to close out. Avoid raw matplotlib
(`ax.plot()`, `ax.fill_between()`, etc.) when an mgplot function exists.

### Architecture
```
# Simple charts: use *_finalise (one-step convenience)
line_plot_finalise(data, **kwargs)
  └─ plot_then_finalise()
       ├─ line_plot(data, **plot_kwargs)    → returns Axes
       └─ finalise_plot(axes, **fp_kwargs)  → styles, saves, closes

# Composite charts: layer mgplot functions, then finalise
ax = fill_between_plot(band_data, color="red", alpha=0.1, label="90% CI")
line_plot(history, ax=ax, color=["navy"], width=2)
finalise_plot(ax, title="...", ylabel="...", show=False)

# finalise_plot() does NOT support plot-level kwargs like annotate, width, color.
```

### Chart Directory Management
```python
import mgplot as mg
mg.set_chart_dir("./CHARTS/MyCharts/")
mg.clear_chart_dir()
```

**Warning:** `abs_helper.get_abs_data()` calls both `set_chart_dir()` and `clear_chart_dir()`.
Don't call it multiple times or you'll reset/clear your chart directory.

### All *_finalise Functions
Each plots data AND saves to file. Pass combined plot + finalise kwargs in one call.

```python
mg.line_plot_finalise(df, ...)           # Line charts
mg.bar_plot_finalise(df, ...)            # Bar charts (grouped or stacked)
mg.growth_plot_finalise(growth_df, ...)  # QoQ bars + TTY line
mg.series_growth_plot_finalise(s, ...)   # Calculates growth from index, then plots
mg.fill_between_plot_finalise(df, ...)   # Shaded area between two columns
mg.postcovid_plot_finalise(s, ...)       # Line with post-COVID projection
mg.revision_plot_finalise(df, ...)       # ABS data revisions
mg.run_plot_finalise(s, ...)             # Highlights runs in a series
mg.seastrend_plot_finalise(df, ...)      # Seasonal + trend overlay
mg.summary_plot_finalise(df, ...)        # Z-score summary (creates 2 plots)
```

### Line Plot Parameters (LineKwargs)
```python
mg.line_plot_finalise(
    data,                # Series or DataFrame
    width=2,             # Line width (float, int, or list per series). NOT lw.
    color=["blue"],      # Colors (str or list per series)
    style="-",           # Line style (str or list)
    alpha=1.0,           # Opacity (float or list)
    marker=None,         # Marker style
    markersize=None,     # Marker size
    drawstyle=None,      # e.g. "steps-post"
    annotate=True,       # Add endpoint value labels
    rounding=1,          # Decimal places for annotations
    fontsize="small",    # Annotation font size
    annotate_color=None, # Annotation color (str, bool, or list)
    plot_from=None,      # Start index (int offset or Period)
    label_series=None,   # Label lines directly instead of legend
    dropna=True,         # Drop NaN values
    # ... plus all Finalise kwargs below
)
```

### Finalise Parameters (FinaliseKwargs)
These work on ALL `*_finalise` functions:
```python
# Titles and labels
title="Chart Title",       # Also used for filename
suptitle="Super Title",    # Above the title
ylabel="Per cent",
xlabel="Year",

# Footers and headers (annotations outside plot area)
rfooter="Source: ABS",     # Right footer
lfooter="Australia. ",     # Left footer
rheader="",                # Right header
lheader="",                # Left header

# Axis limits and ticks
xlim=(0, 100),
ylim=(0, 100),
xticks=[...],
yticks=[...],

# Legend: True, False, None, or dict with any matplotlib legend kwargs
legend=True,
legend={"loc": "upper left", "fontsize": "small", "title": "Quantiles", "ncol": 2},

# Reference lines and bands (single dict or list of dicts)
axhline={"y": 2.5, "color": "red", "linestyle": "--"},
axvline={"x": pd.Period("2020-03"), "color": "grey"},
axhspan={"ymin": 2, "ymax": 3, "color": "lightgreen"},
axvspan={"xmin": ..., "xmax": ...},

# Display and save
y0=True,           # Horizontal line at y=0 if data crosses zero
show=False,        # Display in notebook
tag="mytag",       # Filename becomes: title-mytag.png
pre_tag="prefix",  # Filename becomes: prefix-title.png
file_type="png",   # Output format
dpi=300,           # Resolution
figsize=(8, 6),    # Figure size
dont_save=False,   # Skip saving
dont_close=False,  # Keep figure open
```

### Bar Plot Specific (BarKwargs)
```python
mg.bar_plot_finalise(
    df,
    stacked=False,         # True = stacked, False = grouped side by side
    annotate=True,         # Value labels on bars
    width=0.8,             # Bar width (0-1)
    above=True,            # Annotations above bars
    label_rotation=0,      # X-axis label rotation
    color=["blue", "red"],
)
```

### Multi-Plot Functions
```python
# Same chart at multiple starting points
mg.multi_start(df, function=mg.line_plot_finalise, starts=[0, -20], title="Chart")

# One chart per column
mg.multi_column(df, function=mg.line_plot_finalise, title="Chart")

# Chain any plot function + finalise (used internally by *_finalise)
mg.plot_then_finalise(data, function=mg.line_plot, title="Chart")
```

### Utility Functions
```python
mg.calc_growth(series)           # Returns DataFrame with QoQ and TTY columns
mg.get_color("NSW")              # State color
mg.abbreviate_state("Victoria")  # → "Vic."
mg.contrast("blue")              # Contrasting color for text
```

## Local Helper Modules (in /notebooks/)

### abs_helper.py
Standard notebook setup. **Warning:** `get_abs_data()` resets the chart directory - only call once per notebook. (GDP and population getters used to live here; they now have their own modules - see below. `abs_helper` no longer imports `decompose`/`henderson`, so it does not pull in statsmodels.)

```python
from abs_helper import get_abs_data, collate_summary_data, ANNUAL_CPI_TARGET_RANGE

# Fetches data AND sets up the chart directory (calls set_chart_dir + clear_chart_dir)
abs_dict, meta, source, RECENT = get_abs_data("6401.0")

# Build a summary table for mgplot.summary_plot()  (verbose is keyword-only)
summary = collate_summary_data(to_get, abs_dict, meta)

# CPI target constants for plotting
ANNUAL_CPI_TARGET_RANGE  # {"ymin": 2, "ymax": 3, ...} for axhspan
QUARTERLY_CPI_TARGET     # {"y": 0.617, ...} for axhline
MONTHLY_CPI_TARGET       # {"y": 0.206, ...} for axhline
```

### abs_gdp.py
GDP from the National Accounts (5206.0 key aggregates), cached per kernel session; returns `(series, units)`.

```python
from abs_gdp import get_gdp
gdp, units = get_gdp(gdp_type="CVM", seasonal="SA")   # gdp_type: CP|CVM ; seasonal: SA|T|O
```

### abs_population.py
Single `get_population()` dispatcher for every population concept, plus the smoothing / age-share helpers. Owns the `decompose`/`henderson` (statsmodels) dependency. Leaf fetchers are cached. **Every `get_population(...)` call returns a `(series, units)` tuple** (a defensive copy) - including with `smoothed=`.

```python
from abs_population import get_population, smoothed_monthly_pop_growth

# measure: "ERP" | "civ15" | "adult21" | "implicit"   (state accepts aliases: NSW, Vic, Aus...)
pop, units    = get_population("ERP", state="NSW")        # ERP by state; project=True extends ~2 periods
civ15, units  = get_population("civ15", freq="Q")         # monthly -> quarterly mean (civ15 only)
adults, units = get_population("adult21")                 # quarterly 21+ (civ15 x national 21/15 share)
growth, units = get_population("civ15", smoothed=True)    # smoothed monthly increment; smoothed=True|{how}

# Also exported - these return a bare Series (transforms/ratios), NOT a tuple:
#   smoothed_monthly_pop_growth(level), get_adult_21_share_of_15(), interp_21_share(index)
```

| measure | source | notes |
|---------|--------|-------|
| `ERP` | 3101.0 / 310104 | by state; `project=True` extends ~2 periods |
| `civ15` | 6202.0 / 62020010 | by state; monthly; `freq="Q"`; `smoothed=True\|{}` |
| `adult21` | derived | state civ15 x national 21/15 share (quarterly) |
| `implicit` | 5206.0 | GDP / GDP-per-capita; national only |

### abs_prices.py
Price / numeraire getters. Every series is selected by data-item description (never by series ID); each getter is cached per kernel session and returns `(series, units, stype)` - the series type is reported because it is fixed internally (the caller does not choose it), so callers can label footers.

```python
from abs_prices import get_price_deflator, get_cpi, get_wage_index, get_house_price_index

dfd, units, stype = get_price_deflator("DFD")  # DFD | GNE | HFCE | GDP - published SA IPD index (5206.0)
cpi, units, stype = get_cpi("headline")        # headline (reconstructed to 1948) | headline_sa | trimmed | weighted (6401.0)
wpi, units, stype = get_wage_index("WPI")      # WPI (SA index, 6345.0) | AWOTE ($/week, biannual, 6302.0)
hpi, units, stype = get_house_price_index()    # long-run $ level to 1986 (6432 mean value + discontinued 6416 splice)
report = get_house_price_splice_report()       # the ra.splice() audit for the house-price index
```

- `get_price_deflator`: published Seasonally Adjusted IPDs (the ABS publishes them SA only); DFD (domestic final demand) is the default - the GDP deflator is compromised as a domestic gauge by the terms of trade.
- `get_cpi("headline")`: reconstructed from the published quarterly % change (artefact-free back to 1948); each measure sits on its native ABS reference base (YoY is base-invariant).
- The CPI target constants for plotting remain in `abs_helper`.

### abs_structured_capture.py
For fetching multiple series from different catalogues. **Does NOT reset chart directory** - safe to use for additional data fetching.

```python
from abs_structured_capture import ReqsTuple, ReqsDict, get_abs_data, load_series

# ReqsTuple fields: (cat, table, did, stype, unit, seek_yr_growth, calc_growth, zip_file)
# stype codes: "O"=Original, "S"/"SA"=Seasonally Adjusted, "T"=Trend

# Single series
cpi = ReqsTuple("6401.0", "640106", "All groups CPI, seasonally adjusted", "S", "", True, False, "")
cpi_series = load_series(cpi)

# Multiple series from different catalogues
wanted: ReqsDict = {
    "CPI": ReqsTuple("6401.0", "640106", "All groups CPI, seasonally adjusted", "S", "", True, False, ""),
    "Unemployment": ReqsTuple("6202.0", "62020001", "Unemployment rate ;  Persons ;", "S", "", False, False, ""),
}
data = get_abs_data(wanted)  # Returns dict[str, Series]
```

**Key differences:**
- `abs_helper.get_abs_data(cat)` → returns `(dict, meta, source, recent)`, sets chart dir
- `abs_structured_capture.get_abs_data(wanted)` → returns `dict[str, Series]`, no chart dir changes