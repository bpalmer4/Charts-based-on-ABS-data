"""ChinaBond: the daily China government bond (CGB) yield curve.

A query is capped at a one-year window (a longer one returns an empty page, not an
error), so history is assembled a calendar year at a time. A completed year cannot
change, so it is cached for good; the current year is cached for RECENT_MAX_AGE. The
reply is a rendered HTML page whose table carries bank and note curves beside the
sovereign one, so rows are filtered on the curve name.
"""

import io

import pandas as pd

from au_econ.sources.http_cache import BROWSER_HEADERS, RECENT_MAX_AGE, get_recent

QUERY_URL = "https://yield.chinabond.com.cn/cbweb-cbrc-web/cbrc/historyQuery"
CURVE_ID = "2c9081e50a2f9606010a3068cae70001"  # the sovereign (CGB) curve
CURVE_NAME = "中债国债收益率曲线"
NAME_COLUMN, DATE_COLUMN = "曲线名称", "日期"
FIRST_YEAR = 2006  # the 10-year point begins on 1 March 2006


def _year(year: int, maturity: str, this_year: int) -> pd.Series:
    """Return one calendar year of the sovereign curve at one maturity (years, e.g. "10")."""
    params = {
        "startDate": f"{year}-01-01",
        "endDate": f"{year}-12-31",
        "gjqx": maturity,
        "qxId": "ycqx",
        "locale": "zh_CN",
        "wrjxCBFlag": "0",
        "ycDefIds": CURVE_ID,
    }
    complete = year < this_year
    content = get_recent(
        QUERY_URL,
        params,
        "chinabond",
        None if complete else RECENT_MAX_AGE,
        headers=BROWSER_HEADERS,
        fallback=True,
    )
    tenor = f"{maturity}年"
    frames = []
    for table in pd.read_html(io.StringIO(content.decode("utf-8"))):
        frame = table.set_axis([str(name) for name in table.iloc[0]], axis=1)[1:]
        if NAME_COLUMN in frame.columns and tenor in frame.columns:
            frames.append(frame[frame[NAME_COLUMN] == CURVE_NAME])
    if not frames:
        if complete:
            raise ValueError(f"ChinaBond returned no {year} data table")
        return pd.Series(dtype=float, index=pd.PeriodIndex([], freq="D"))
    rows = pd.concat(frames)
    return pd.Series(
        pd.to_numeric(rows[tenor], errors="coerce").to_numpy(), index=pd.PeriodIndex(rows[DATE_COLUMN], freq="D")
    ).dropna()


def get_cgb_yield(maturity: str) -> pd.Series:
    """Return the daily sovereign yield at one maturity (years, e.g. "10"), per cent."""
    this_year = pd.Timestamp.today().year
    years = range(FIRST_YEAR, this_year + 1)
    series = pd.concat([_year(year, maturity, this_year) for year in years]).sort_index()
    series = series[~series.index.duplicated(keep="last")]
    if series.empty:
        raise ValueError(f"ChinaBond returned no {maturity}-year observations")
    return series
