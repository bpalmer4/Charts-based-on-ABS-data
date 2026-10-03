"""Oilprice.com: daily crude benchmark prices, from the site's public chart-widget JSON.

Each request needs a CSRF token from the site first. A year of prices comes back per
request (the "period" setting), more than the charts use, so nothing is accumulated
between runs.
"""

import pandas as pd
import requests

BASE_URL = "https://oilprice.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": f"{BASE_URL}/oil-price-charts/",
}
TIMEOUT = 15  # seconds
ONE_YEAR = 5  # the widget's period code for a year of daily prices
STALE_DAYS = 400  # entries this far before the latest are out-of-sequence artefacts

# Oilprice.com blend ids
WTI, BRENT, OMAN, DUBAI = 45, 46, 48, 144


def get_blend(blend_id: int, name: str) -> pd.Series:
    """Return a blend's daily prices, USD per barrel, on a daily PeriodIndex."""
    csrf = requests.get(f"{BASE_URL}/ajax/csrf", headers=HEADERS, timeout=TIMEOUT).json()
    response = requests.post(
        f"{BASE_URL}/freewidgets/json_get_oilprices",
        headers=HEADERS,
        data={"blend_id": blend_id, "period": ONE_YEAR, csrf["name"]: csrf["hash"]},
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    frame = pd.DataFrame(response.json()["prices"])
    if frame.empty:
        raise ValueError(f"Oilprice.com blend {blend_id}: no prices")
    frame["date"] = pd.to_datetime(frame["time"], unit="s").dt.normalize()
    frame["price"] = frame["price"].astype(float)
    frame = frame.sort_values("date").reset_index(drop=True)
    frame = frame[frame["date"] >= frame["date"].iloc[-1] - pd.Timedelta(days=STALE_DAYS)]
    print(f"{name}: {len(frame)} rows")
    series = pd.Series(frame["price"].to_numpy(), index=pd.PeriodIndex(frame["date"], freq="D"), name=name)
    return series[~series.index.duplicated(keep="last")]
