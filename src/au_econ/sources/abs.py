"""ABS data: catalogues through readabs, and the CPI expenditure hierarchy through sdmxabs.

sdmxabs is used for the CPI hierarchy only, and only here. Data cubes (workbooks that are
not time-series tables, so readabs does not read them) are found on their release pages
and downloaded through http_cache.
"""

import re
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Unpack
from urllib.parse import urljoin

import readabs as ra
import requests
import sdmxabs as sa

from au_econ.sources.http_cache import get_file

if TYPE_CHECKING:
    from pandas import DataFrame
    from readabs import ReadArgs

RECENT = "2020-12-01"  # default start for recent-period charts

ABS_SITE = "https://www.abs.gov.au"
HEADERS = {"User-Agent": "Mozilla/5.0"}
TIMEOUT = 30  # seconds

# CPI expenditure hierarchy: the SDMX INDEX codelist of the CPI data structure
CPI_STRUCTURE, CPI_DIMENSION = "CPI", "INDEX"
CPI_LEVELS = {0: "aggregate", 1: "group", 2: "sub-group", 3: "class"}
CPI_ROOT = "All groups CPI"


@dataclass(frozen=True)
class AbsRelease:
    """One ABS catalogue: its tables, metadata, source label and a recent start date."""

    data: dict[str, DataFrame]
    meta: DataFrame
    source: str
    recent: str


def fetch_release(cat: str, **kwargs: Unpack[ReadArgs]) -> AbsRelease:
    """Fetch an ABS catalogue (e.g. "6302.0"); raise if nothing comes back.

    Keyword arguments go to readabs.read_abs_cat, e.g. get_zip=False, get_excel=True
    for a catalogue the ABS publishes without a complete zip file.
    """
    data, meta = ra.read_abs_cat(cat, **kwargs)
    if not data or meta.empty:
        raise ValueError(f"ABS {cat}: no data returned")
    return AbsRelease(data=data, meta=meta, source=f"ABS: {cat}", recent=RECENT)


def latest_data_cube_url(release_page: str, cube: str) -> str:
    """Return the URL of a data cube workbook (e.g. "34070DO004") linked from an ABS latest-release page."""
    response = requests.get(release_page, headers=HEADERS, timeout=TIMEOUT)
    response.raise_for_status()
    links = sorted(set(re.findall(rf'href="([^"]*/{cube}_[^"/]*\.xlsx)"', response.text)))
    if len(links) != 1:
        raise ValueError(f"ABS {cube}: expected one workbook link at {release_page}, found {links}")
    return urljoin(ABS_SITE, links[0])


def get_data_cube(url: str) -> bytes:
    """Return an ABS data cube workbook (not a time-series table, so not readabs), cached on disk."""
    return get_file(url, prefix="abs")


@dataclass(frozen=True)
class CpiItem:
    """One item in the CPI expenditure hierarchy."""

    name: str
    level: str  # "aggregate", "group", "sub-group" or "class"
    parent: str | None  # the parent's code


@cache
def _cpi_hierarchy() -> dict[str, CpiItem]:
    """Return the CPI expenditure hierarchy keyed by SDMX code (cached for the run).

    Keyed by code, not name: the ABS reuses a name for a sub-group and its only class
    (e.g. "Tobacco", "Rents").
    """
    codelist = sa.code_list_for(CPI_STRUCTURE, CPI_DIMENSION)
    if not codelist:
        raise ValueError("CPI hierarchy: empty SDMX codelist")

    def depth(code: str) -> int:
        level, current = 0, code
        while "parent" in codelist.get(current, {}):
            level += 1
            current = codelist[current]["parent"]
        return level

    hierarchy: dict[str, CpiItem] = {}
    for code, info in codelist.items():
        level = depth(code)
        parent = info.get("parent")
        hierarchy[str(code)] = CpiItem(
            name=info["name"],
            level=CPI_LEVELS.get(level, f"level-{level}"),
            parent=str(parent) if parent else None,
        )
    return hierarchy


def cpi_names(level: str, root: str = CPI_ROOT) -> list[str]:
    """Return the names at one level of the CPI hierarchy (e.g. "class") that sit under root."""
    hierarchy = _cpi_hierarchy()

    def under_root(code: str) -> bool:
        current = hierarchy[code].parent
        while current is not None and current in hierarchy:
            if hierarchy[current].name == root:
                return True
            current = hierarchy[current].parent
        return False

    return [item.name for code, item in hierarchy.items() if item.level == level and under_root(code)]
