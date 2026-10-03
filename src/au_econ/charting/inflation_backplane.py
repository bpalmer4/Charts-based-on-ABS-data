"""Inflation backplane: shade a quarterly or monthly chart where inflation left the RBA band.

Each period where annualised quarterly trimmed mean inflation sat outside the 2-3% band
gets a vertical span behind the data, red above and blue below, in three steps of
intensity by how far outside the band it sat. Zero-width labelled key spans lead the
shading, so a legend shows every step.

- inflation_backplane(index, ax=None): draw the backplane for the quarters or months in
  index (normally the index of the data just plotted) and return the axes. A month takes
  its quarter's inflation; months after the last published quarter take the monthly
  trimmed mean (see monthly_inflation_tail).
- BACKPLANE_LHEADER: an explanatory lheader for finalise_plot, which owns the header.
- BACKPLANE_MONTHLY_LFOOTER: the inflation note for a monthly chart's lfooter, appended
  after the geography.

The RBA's 2-3% band is a headline-CPI target, so the shading marks where the core measure
sat outside the band, not where the target was missed.
"""

from functools import cache
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import pandas as pd
import readabs as ra
from mgplot import get_setting
from readabs import metacol as mc

from au_econ.series.prices import get_cpi

if TYPE_CHECKING:
    from matplotlib.axes import Axes

INFLATION_LOW = 2.0
INFLATION_HIGH = 3.0
SHADE_STEPS = (1.0, 2.0)  # points outside the band where the shading steps darker
SHADE_ALPHAS = (0.08, 0.16, 0.25)  # one alpha per step, nearest the band first
QUARTERLY = "Q"
MONTHLY = "M"
QUARTERS_PER_YEAR = 4
MONTHS_PER_QUARTER = 3
PERCENT = 100.0

# monthly CPI (6401.0), for the months after the last published quarter
CPI_CATALOGUE = "6401.0"
MONTHLY_CPI_TABLE = "640106"
MONTHLY_TRIMMED_DID = "Index Numbers ;  Trimmed Mean ;  Australia ;"
SEASONALLY_ADJUSTED = "Seasonally Adjusted"

BACKPLANE_LHEADER = (
    f"Shaded where annualised quarterly trimmed mean inflation sat outside "
    f"{INFLATION_LOW:g}-{INFLATION_HIGH:g}%: red above, blue below"
)
BACKPLANE_MONTHLY_LFOOTER = "Inflation: trimmed mean, quarterly; 3mths annualised after latest quarter. "


def quarterly_inflation() -> pd.Series:
    """Return annualised quarterly trimmed mean inflation (SA)."""
    trimmed, _, _ = get_cpi("trimmed")
    quarterly = trimmed.pct_change(fill_method=None).dropna()
    inflation = ((1 + quarterly) ** QUARTERS_PER_YEAR - 1) * PERCENT
    if inflation.empty:
        raise ValueError("No inflation data returned")
    return inflation


@cache
def _monthly_trimmed_index() -> pd.Series:
    """Return the monthly trimmed mean index (SA), from the monthly CPI table (cached; not for mutation)."""
    data, meta = ra.read_abs_cat(CPI_CATALOGUE, single_excel_only=MONTHLY_CPI_TABLE)
    _, series_id, _ = ra.find_abs_id(
        meta,
        {MONTHLY_CPI_TABLE: mc.table, MONTHLY_TRIMMED_DID: mc.did, SEASONALLY_ADJUSTED: mc.stype},
        verbose=False,
    )
    index = data[MONTHLY_CPI_TABLE][series_id].dropna()
    if index.empty:
        raise ValueError(f"No data returned for {MONTHLY_TRIMMED_DID!r}")
    return index


def monthly_inflation_tail() -> pd.Series:
    """Return monthly trimmed mean inflation for the months after the last published quarter.

    Each month is the average index over the three months ending in it, against the three
    months before, annualised: the quarterly calculation on a rolling monthly window. It
    tracks the published quarterly trimmed mean to about 0.3 points (the ABS trims monthly
    and quarterly price changes separately), and each quarterly release replaces it. Empty
    when no month is beyond the last published quarter.
    """
    rolling = _monthly_trimmed_index().rolling(MONTHS_PER_QUARTER).mean()
    growth = (rolling / rolling.shift(MONTHS_PER_QUARTER)) ** QUARTERS_PER_YEAR
    inflation = ((growth - 1) * PERCENT).dropna()
    index = inflation.index
    if not isinstance(index, pd.PeriodIndex):
        raise TypeError("Expected a monthly PeriodIndex")
    return inflation[index.asfreq(QUARTERLY) > quarterly_inflation().index[-1]]


def _step_label(step: int, *, high: bool) -> str:
    """Return the legend label for one shading step, e.g. 'Inflation 3-4%'."""
    edges = (0.0, *SHADE_STEPS)
    near = edges[step]
    far = edges[step + 1] if step + 1 < len(edges) else None
    if high:
        low_edge = INFLATION_HIGH + near
        return f"Inflation {low_edge:g}%+" if far is None else f"Inflation {low_edge:g}-{INFLATION_HIGH + far:g}%"
    high_edge = INFLATION_LOW - near
    return (
        f"Inflation below {high_edge:g}%" if far is None else f"Inflation {INFLATION_LOW - far:g}-{high_edge:g}%"
    )


def _span_style(step: int, *, high: bool) -> dict[str, Any]:
    """Return the ax.axvspan keyword arguments for one shading step."""
    return {"color": "tab:red" if high else "tab:blue", "alpha": SHADE_ALPHAS[step], "zorder": 0, "linewidth": 0}


def inflation_backplane(index: pd.PeriodIndex, ax: Axes | None = None) -> Axes:
    """Shade the periods in index where inflation sat outside the 2-3% band; return the axes.

    index is the quarterly or monthly PeriodIndex to shade, normally that of the data on ax.
    Spans are one period wide, drawn at period ordinals, the x-coordinates mgplot uses for a
    period axis. ax is the axes to draw on; a new one at mgplot's default figsize if None.
    The key spans lead, reds then blues, each nearest the band first; the caller sets the
    legend layout (ncol) in finalise_plot.
    """
    if not isinstance(index, pd.PeriodIndex) or index.empty:
        raise TypeError(f"Expected a non-empty PeriodIndex, got {type(index)}")
    if not index.freqstr.startswith((QUARTERLY, MONTHLY)):
        raise ValueError(f"Expected a quarterly or monthly PeriodIndex, got freq {index.freqstr!r}")
    if ax is None:
        _, ax = plt.subplots(figsize=get_setting("figsize"))

    first = index[0].ordinal
    for high in (True, False):
        for step in range(len(SHADE_ALPHAS)):
            ax.axvspan(first, first, label=_step_label(step, high=high), **_span_style(step, high=high))

    quarterly = quarterly_inflation().reindex(index.asfreq(QUARTERLY))
    inflation = pd.Series(quarterly.to_numpy(), index=index)
    if index.freqstr.startswith(MONTHLY):
        inflation = inflation.fillna(monthly_inflation_tail().reindex(index))
    for period, raw in zip(index, inflation.to_numpy(), strict=True):
        value = float(raw)
        if pd.isna(value) or INFLATION_LOW <= value <= INFLATION_HIGH:
            continue
        high = value > INFLATION_HIGH
        deviation = value - INFLATION_HIGH if high else INFLATION_LOW - value
        step = sum(deviation >= edge for edge in SHADE_STEPS)
        ax.axvspan(period.ordinal, period.ordinal + 1, **_span_style(step, high=high))
    return ax
