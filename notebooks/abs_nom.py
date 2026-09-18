"""Net Overseas Migration: the official ABS series and a timely forward proxy.

Split out of ABS Population.ipynb so both that notebook and the building-activity
notebooks can use the same NOM measures rather than each rebuilding them. The
3101.0 fetch is cached for the life of the kernel session (readabs handles the
on-disk cache between runs); the public getters return defensive copies so the
cached objects cannot be mutated by a caller.

- get_nom() - the published Net Overseas Migration series (3101.0, table 310101,
  Original, quarterly) as a 4-quarter rolling sum, i.e. a through-the-year
  migration count in '000.
- get_nom_forward_proxy() - a proxy for NOM built off the monthly Labour Force
  civilian population aged 15+ (6202.0), which lands ~2 weeks after its reference
  month against ~5-6 months for 3101.0. It therefore runs one to two quarters
  ahead of the official series.

The proxy is *not* a substitute for the official count. It rests on the
provisional edge of a benchmark-interpolated series: first-published civ15 months
are typically revised down, so the newest proxy quarters move as the ABS
re-benchmarks. Treat the forward segment as an indication of direction.
"""

# === imports
from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import pandas as pd
import readabs as ra
from readabs import metacol as mc

# Private helpers reused from abs_population, which owns the ERP-by-age
# machinery and the part-quarter completion used below.
from abs_population import (
    _complete_trailing_quarter,
    _erp_age_sum,
    _interp_june,
    get_population,
)

if TYPE_CHECKING:
    from pandas import Series

# === constants
NOM_CAT = "3101.0"
NOM_TABLE = "310101"
NOM_DID = "Net Overseas Migration ;  Australia ;"
DEATHS_DID = "Deaths ;  Australia ;"
NOM_STYPE = "Original"  # 310101 is published Original only
ANNUAL_QTRS = 4  # quarters in a year: the through-the-year window
MONTHS_IN_QUARTER = 3


# === the official series
@cache
def _get_310101() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch 3101.0 table 310101 and its metadata (cached per kernel session)."""
    data, meta = ra.read_abs_cat(NOM_CAT, single_excel_only=NOM_TABLE, verbose=False)
    return data[NOM_TABLE], meta


def _rolling_annual(did: str) -> tuple[Series, str]:
    """A 310101 series selected by description, as a 4-quarter rolling sum."""
    df, meta = _get_310101()
    selector = {NOM_TABLE: mc.table, NOM_STYPE: mc.stype, did: mc.did}
    _t, sid, units = ra.find_abs_id(meta, selector, verbose=False)
    return df[sid].rolling(ANNUAL_QTRS).sum(), units


@cache
def _get_nom() -> tuple[Series, str]:
    """Official NOM as a through-the-year count (cached per kernel session)."""
    nom, units = _rolling_annual(NOM_DID)
    nom = nom.dropna()
    if nom.empty:
        raise ValueError(f"No Net Overseas Migration data found in {NOM_TABLE}")
    return nom, units


def get_nom() -> tuple[Series, str, str]:
    """Return official Net Overseas Migration, through-the-year.

    Returns:
        A tuple of the quarterly NOM series as a 4-quarter rolling sum (a
        defensive copy, in '000 persons per year), its units and its series
        type. The series type is reported rather than chosen because 310101 is
        published Original only.

    """
    nom, units = _get_nom()
    return nom.copy(), units, NOM_STYPE


# === the 6202-based forward proxy
@cache
def _build_nom_forward_proxy() -> tuple[Series, Series, pd.Period, pd.Period]:
    """Build the 6202-based forward proxy for Net Overseas Migration.

    proxy = civ15 year-on-year growth (6202, monthly - the timely lead)
            - 15-year-old ageing-in  +  15+ deaths  +  child migration
          = civ15 growth - 15+ natural increase + child migration

    The +deaths term is essential, not optional: in a headcount change a death
    is indistinguishable from an emigration, so without crediting deaths back
    they are mistaken for people leaving (it sits bundled with ageing-in as the
    ~flat 150k 15+ natural increase). Child migration (0-14 ERP cohort survival,
    3101.0) is observed while ERP-by-age exists, then extended over the forward
    quarters by a ratio of growth derived from the data (not hardcoded), so it
    scales with the migration cycle.

    civ15 arrives monthly, so the newest quarter is usually part-filled. Its
    mean would then sit on its first or second month rather than its midpoint,
    understating growth by a month's worth, so the quarter is completed by
    `_complete_trailing_quarter` before differencing. The quarters beyond
    `last_complete` therefore rest on extrapolated months and are returned
    separately so a chart can mark them.

    Returns (proxy, official_nom, last_official_quarter, last_complete_quarter);
    quarterly TTY, '000.
    """
    # timely engine: civ15 (6202) year-on-year growth
    civ15, _ = get_population("civ15", state="Australia")
    months_in_quarter = civ15.resample("Q").count()
    last_complete = months_in_quarter[months_in_quarter.eq(MONTHS_IN_QUARTER)].index[-1]
    # Complete the trailing quarter, then mask any quarter still part-filled -
    # which drops the part-filled first quarter of the series, keeping the
    # year-ago base of the quarter four steps later honest.
    filled = _complete_trailing_quarter(civ15)
    filled_counts = filled.resample("Q").count()
    level_q = filled.resample("Q").mean().where(filled_counts.eq(MONTHS_IN_QUARTER))
    growth = level_q.diff(ANNUAL_QTRS)

    # single-year-of-age ERP (3101.0): ageing-in (15-year-olds) and the 0-14 pyramid
    def age_ge(a: int) -> Series:
        return _erp_age_sum(a) / 1_000.0  # persons -> '000, annual June

    n15_yr = age_ge(15) - age_ge(16)  # count of 15-year-olds
    childmig_yr = (
        (age_ge(1) - age_ge(16)) - (age_ge(0) - age_ge(15)).shift(1)
    ).dropna()
    n15 = _interp_june(n15_yr, growth.index)
    childmig_obs = _interp_june(childmig_yr, growth.index)

    # NOM and 15+ deaths from 3101.0 (4-quarter rolling)
    deaths, _deaths_units = _rolling_annual(DEATHS_DID)
    nom, _nom_units = _get_nom()

    natural_increase = (n15 - deaths).ffill()  # ageing-in less deaths (~flat 150)
    m15 = growth - natural_increase  # = growth - ageing-in + deaths

    # extend child migration over forward quarters by a derived ratio of growth
    last_age = pd.Period(f"{childmig_yr.index[-1].year}Q2", freq="Q-DEC")
    ratio = (childmig_obs.loc[:last_age] / growth.loc[:last_age]).dropna().median()
    childmig = childmig_obs.copy()
    childmig[childmig.index > last_age] = ratio * growth[childmig.index > last_age]

    proxy = (m15 + childmig).dropna()
    if proxy.empty:
        raise ValueError("NOM forward proxy is empty - check the 6202/3101 inputs")
    return proxy, nom, nom.index[-1], last_complete


def get_nom_forward_proxy() -> tuple[Series, Series, pd.Period, pd.Period]:
    """Return the 6202-based forward proxy for NOM, with the official series.

    Returns:
        A tuple of (proxy, official_nom, last_official_quarter,
        last_complete_quarter). Both series are quarterly through-the-year
        counts in '000 persons per year, and are defensive copies. The proxy
        runs past `last_official_quarter` (that is the point of it); beyond
        `last_complete_quarter` it rests on a part-filled civ15 quarter
        completed by extrapolation, so callers should mark that segment.

    """
    proxy, nom, last_official, last_complete = _build_nom_forward_proxy()
    return proxy.copy(), nom.copy(), last_official, last_complete
