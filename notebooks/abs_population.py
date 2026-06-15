"""Population convenience getters across several ABS sources.

Consolidates the population concepts previously scattered across abs_helper.py
and the dwelling-stock / labour-force notebooks behind a single get_population()
dispatcher, mirroring abs_gdp.get_gdp. Four measures are supported:

  ERP       Estimated Resident Population (3101.0, table 310104), by state.
  civ15     Civilian population aged 15+ (6202.0, table 62020010), by state.
  adult21   Population aged 21+ (derived: state civ15 x national 21/15 share),
            quarterly.
  implicit  Population implied by the National Accounts (5206.0 GDP / GDP per
            capita); national only.

Two orthogonal, non-composable options sit on top of the level:
  freq      Downsample the level (currently None or "Q"; civ15 only, since it
            is the sole monthly measure - the others are already quarterly).
  smoothed  De-step a monthly benchmark-interpolated level into a smooth monthly
            *increment* (civ15 only). False | True | {how}, where the dict is
            forwarded to smoothed_monthly_pop_growth. smoothed manages its own
            internal M->Q downsample, so it is not combined with freq.

Series are returned in their native ABS units with the unit string alongside
(ERP/implicit: "Number"; civ15/adult21: "000"); recalibrate at plot time per
the project convention. Leaf fetchers are cached for the life of the kernel
session (readabs handles the on-disk, between-run cache); the public getter
returns a defensive copy so cached data cannot be mutated by a caller. This
module owns the decompose/henderson (statsmodels) dependency so abs_helper
stays light.
"""

# === imports
from __future__ import annotations

from functools import cache

import numpy as np
import pandas as pd
import readabs as ra
from decompose import decompose
from henderson import hma
from readabs import metacol as mc

# === constants
_MEASURES = ("ERP", "civ15", "adult21", "implicit")
_COVID_YEARS = (2020, 2021)
_HENDERSON_TERMS = 13
_PROJECT_PERIODS = 2
_AGE_15 = 15  # civilian working-age threshold (15+)
_AGE_21 = 21  # adult threshold (21+)
_PER_MILLION = 1_000_000  # GDP ($M) / GDP-per-capita ($) -> millions of persons; rescale to persons

# Map common state aliases/abbreviations to the full name the ABS uses in its
# data-item descriptions. Matched case-insensitively; full names map to
# themselves so passing the canonical name is always valid.
_STATE_ALIASES = {
    "australia": "Australia", "aus": "Australia", "au": "Australia",
    "aust": "Australia", "oz": "Australia",
    "new south wales": "New South Wales", "nsw": "New South Wales",
    "victoria": "Victoria", "vic": "Victoria",
    "queensland": "Queensland", "qld": "Queensland",
    "south australia": "South Australia", "sa": "South Australia",
    "western australia": "Western Australia", "wa": "Western Australia",
    "tasmania": "Tasmania", "tas": "Tasmania",
    "northern territory": "Northern Territory", "nt": "Northern Territory",
    "australian capital territory": "Australian Capital Territory",
    "act": "Australian Capital Territory",
}


def _canonical_state(state: str) -> str:
    """Resolve a state alias/abbreviation to the full name the ABS uses.

    Args:
        state: a state/territory name or common abbreviation (e.g. "NSW", "nsw",
            "Vic", "Aus"); matched case-insensitively.

    Returns:
        The full ABS state name (e.g. "New South Wales", "Australia").

    """
    try:
        return _STATE_ALIASES[state.strip().lower()]
    except KeyError:
        known = sorted(set(_STATE_ALIASES.values()))
        raise ValueError(f"Unrecognised state: {state!r}. Known: {known}.") from None


# === leaf fetchers (cached per kernel session; return raw, native-unit series)
@cache
def _pop_erp(state: str) -> tuple[pd.Series, str]:
    """Estimated Resident Population level for a state (3101.0, table 310104).

    Args:
        state: the state/territory name, or "Australia" for the national total.

    Returns:
        The raw ERP series (persons) and its units. Callers must not mutate it.

    """
    cat, table = "3101.0", "310104"
    data, meta = ra.read_abs_cat(cat, single_excel_only=table, verbose=False)
    selector = {
        f";  {state} ;": mc.did,  # too many states have "Australia" in their name
        "Estimated Resident Population ;  Persons ;  ": mc.did,
    }
    _t, sid, units = ra.find_abs_id(meta, selector, verbose=False)
    return data[table][sid], units


@cache
def _pop_civ15(state: str) -> tuple[pd.Series, str]:
    """Civilian population aged 15+ level for a state (6202.0, table 62020010).

    Table 62020010 carries every state plus Australia; the state sits in the
    data-item description ("Persons ;  Australia ;" or "Persons ;  > NSW ;").

    Args:
        state: the state/territory name, or "Australia" for the national total.

    Returns:
        The raw monthly civ15 level ("000") and its units. Not for mutation.

    """
    cat, table = "6202.0", "62020010"
    data, meta = ra.read_abs_cat(cat, single_excel_only=table, verbose=False)
    where = "Australia" if state == "Australia" else f"> {state}"
    selector = {
        table: mc.table,
        "Civilian population aged 15 years and over ;  Persons ;": mc.did,
        f";  {where} ;": mc.did,
        "Original": mc.stype,
    }
    _t, sid, units = ra.find_abs_id(meta, selector, verbose=False)
    return data[table][sid].dropna(), units


@cache
def _pop_adult21(state: str) -> tuple[pd.Series, str]:
    """Quarterly population aged 21+ for a state, on the civilian-15+ basis.

    The ABS has no quarterly single-year-of-age population, so the 21+ count is
    built as the (quarterly mean) civilian population 15+ scaled by the national
    annual 21/15 age-share, interpolated to quarterly. The share drifts slowly,
    so this keeps the 21+ line on the same smooth basis as the 15+ series and
    differing only by the 15-20 cohort. For a state this applies the *national*
    21/15 share to that state's civ15 - a documented approximation, as no state
    single-year-of-age series exists.

    Args:
        state: the state/territory name, or "Australia" for the national total.

    Returns:
        The quarterly 21+ population ("000") and its units. Not for mutation.

    """
    civ15, units = _pop_civ15(state)
    civ15_q = ra.monthly_to_qtly(civ15, f="mean")
    return (interp_21_share(civ15_q.index) * civ15_q).dropna(), units


@cache
def _pop_implicit() -> tuple[pd.Series, str]:
    """Return the population implied by the National Accounts (5206.0); national only.

    Implied population = Gross domestic product / GDP per capita (both chain
    volume measures, from the key aggregates table), scaled to persons.

    Returns:
        The implied quarterly population (persons) and its units ("Number").

    """
    cat, table = "5206.0", "5206001_Key_Aggregates"
    data, meta = ra.read_abs_cat(cat, single_excel_only=table, verbose=False)
    base = {table: mc.table, "Original": mc.stype}
    _t, gdp_id, gdp_unit = ra.find_abs_id(
        meta, base | {"Gross domestic product: Chain volume measures ;": mc.did}, verbose=False
    )
    _t, pc_id, pc_unit = ra.find_abs_id(
        meta, base | {"GDP per capita: Chain volume measures ;": mc.did}, verbose=False
    )
    # The _PER_MILLION rescale assumes GDP in $M over per-capita in $; bail if ABS changes units.
    if (gdp_unit, pc_unit) != ("$ Millions", "$"):
        raise ValueError(
            f"Unexpected 5206.0 units for implicit population: GDP={gdp_unit!r}, "
            f"per-capita={pc_unit!r}; expected ('$ Millions', '$')."
        )
    population = (data[table][gdp_id] / data[table][pc_id] * _PER_MILLION).dropna()
    return population, "Number"


# === age-structure helpers (dimensionless ratios; not (series, units))
@cache
def get_adult_21_share_of_15() -> pd.Series:
    """Annual 21+/15+ population ratio (ERP single year of age, 3101.0, June).

    The 21+ population as a share of the 15+ population: a slowly drifting adult
    age-structure ratio used to scale a civilian-15+ level down to a 21+ estimate
    on the same basis (the two then differ only by the 15-20 cohort).

    Returns:
        The annual (June) ratio as a dimensionless Series.

    """
    age_table = "3101059"
    age_data, age_meta = ra.read_abs_cat("3101.0", single_excel_only=age_table, verbose=False)
    persons = age_meta[age_meta[mc.did].str.contains("Persons")]
    ids_21, ids_15 = [], []
    for _, row in persons.iterrows():
        age_str = row[mc.did].split(";")[2].strip()
        age = 100 if age_str == "100 and over" else int(age_str)
        if age >= _AGE_15:
            ids_15.append(row[mc.id])
        if age >= _AGE_21:
            ids_21.append(row[mc.id])
    pop_21 = age_data[age_table][ids_21].sum(axis=1)  # persons, Y-JUN
    pop_15 = age_data[age_table][ids_15].sum(axis=1)  # persons, Y-JUN
    return (pop_21 / pop_15).dropna()


def interp_21_share(index: pd.PeriodIndex) -> pd.Series:
    """Interpolate the annual 21+/15+ ratio onto a quarterly or monthly index.

    Each June value is anchored at the June quarter (Q2) / June month, linearly
    interpolated between, and held flat past the latest June benchmark.

    Args:
        index: a quarterly (Q-DEC) or monthly (M) PeriodIndex to interpolate onto.

    Returns:
        The interpolated dimensionless ratio, indexed on `index`.

    """
    ratio_annual = get_adult_21_share_of_15()
    out = pd.Series(np.nan, index=index, dtype=float)
    freq = index.freqstr
    for period, value in ratio_annual.items():
        anchor = (
            pd.Period(year=period.year, quarter=2, freq=freq)
            if freq.startswith("Q")
            else pd.Period(year=period.year, month=6, freq=freq)
        )
        if anchor in out.index:
            out[anchor] = value
    return out.interpolate(limit_area="inside").ffill()


# === smoothing transform (bare Series in, bare Series out)
def smoothed_monthly_pop_growth(
    level: pd.Series,
    *,
    ignore_years: tuple[int, int] = _COVID_YEARS,
    henderson_terms: int = _HENDERSON_TERMS,
) -> pd.Series:
    """Smooth a benchmark-stepped monthly population *level* into a smooth monthly *increment*.

    The ABS interpolates quarterly ERP benchmarks onto months, so the raw
    month-on-month change of a population level (e.g. the 6202 civilian
    population aged 15+) is a step function that jumps at each new benchmark.
    Differencing and then smoothing leaves the steps in place; instead this works
    on the quarterly trend:

      1. downsample the level to quarterly means;
      2. take the Trend of a multiplicative seasonal decomposition, ARIMA
         extended so the endpoint uses symmetric (not Musgrave-lagged) Henderson
         weights, with COVID years excluded from the seasonal estimate;
      3. spread each quarterly trend increment evenly over its three months (/3,
         forward-filled) and round the residual steps with an
         ``henderson_terms``-term Henderson moving average;
      4. mask back to the months actually present in ``level`` (dropping the
         fabricated trailing months of any partial quarter).

    Args:
        level: a monthly population Series on a PeriodIndex.
        ignore_years: years excluded from the seasonal estimate (COVID).
        henderson_terms: length of the final Henderson moving average.

    Returns:
        A monthly Series of the smoothed population increment (same units as
        ``level``), indexed on the months present in ``level``.

    """
    q_trend = decompose(
        level.resample("Q").mean(),
        model="multiplicative",
        constant_seasonal=True,
        ignore_years=ignore_years,
        arima_extend=True,
    )["Trend"]
    return (
        hma((q_trend.diff() / 3).resample("M").ffill().dropna(), henderson_terms)
        .reindex(level.index)
        .dropna()
    )


# === projection helper
def _project(pop: pd.Series, periods: int = _PROJECT_PERIODS) -> pd.Series:
    """Extend a population level a few periods at its latest period-on-period rate.

    A rough short-horizon extrapolation; adequate over the ~6 months between an
    ERP release and the present. Operates on a copy (the input may be cached).
    """
    pop = pop.copy()
    rate = pop.iloc[-1] / pop.iloc[-2]
    base = pop.index[-1]
    for i in range(1, periods + 1):
        pop[base + i] = pop[base + i - 1] * rate
    return pop


# === public dispatcher
def _smoothed_increment(
    measure: str,
    state: str,
    freq: str | None,
    *,
    smoothed: bool | dict,
) -> tuple[pd.Series, str]:
    """Return the smoothed monthly civ15 increment (helper for get_population)."""
    if measure != "civ15":
        raise ValueError("smoothed is only supported for measure='civ15'.")
    if freq is not None:
        raise ValueError("smoothed implies a monthly increment; do not combine with freq.")
    opts = smoothed if isinstance(smoothed, dict) else {}
    level, units = _pop_civ15(state)
    return smoothed_monthly_pop_growth(level, **opts), units


def _fetch_level(
    measure: str,
    state: str,
    freq: str | None,
    *,
    project: bool,
) -> tuple[pd.Series, str]:
    """Return a population level (copied, optionally resampled/projected)."""
    if measure == "ERP":
        series, units = _pop_erp(state)
        return (_project(series) if project else series.copy()), units
    if measure == "civ15":
        series, units = _pop_civ15(state)
        series = series.copy()
        if freq == "Q":
            series = ra.monthly_to_qtly(series, f="mean")
        elif freq not in (None, "M"):
            raise ValueError(f"Unsupported freq: {freq!r} (use None or 'Q').")
        return series, units
    if measure == "adult21":
        series, units = _pop_adult21(state)
        return series.copy(), units
    if state != "Australia":  # implicit
        raise ValueError("measure='implicit' is national only; state must be 'Australia'.")
    series, units = _pop_implicit()
    return series.copy(), units


def get_population(
    measure: str = "ERP",
    *,
    state: str = "Australia",
    project: bool = True,
    freq: str | None = None,
    smoothed: bool | dict = False,
) -> tuple[pd.Series, str]:
    """Return an ABS population series and its units, by measure.

    Args:
        measure: one of "ERP", "civ15", "adult21", "implicit" (see module docstring).
        state: state/territory name or abbreviation (e.g. "NSW", "Vic", "Aus"),
            resolved case-insensitively. Honoured by ERP, civ15 and adult21;
            "implicit" is national only.
        project: ERP only - extend the level ~2 periods at the latest rate.
        freq: None or "Q" to downsample the level (civ15 only - the lone monthly
            measure; "Q" uses a quarterly mean).
        smoothed: civ15 only - return the smoothed monthly *increment* instead of
            the level. False | True | dict; a dict is forwarded as keyword
            arguments to smoothed_monthly_pop_growth. Not combinable with freq.

    Returns:
        A tuple of the requested series (a defensive copy) and its native units.

    """
    if measure not in _MEASURES:
        raise ValueError(f"Unknown measure: {measure!r}. Choose from {_MEASURES}.")
    state = _canonical_state(state)
    if smoothed is not False and smoothed is not None:
        return _smoothed_increment(measure, state, freq, smoothed=smoothed)
    if freq is not None and measure != "civ15":
        raise ValueError(f"freq is only supported for measure='civ15' (got {measure!r}).")
    return _fetch_level(measure, state, freq, project=project)
