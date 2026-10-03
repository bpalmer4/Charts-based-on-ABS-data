"""OPEC: the OPEC Reference Basket daily price archive, from opec.org.

opec.org sits behind Cloudflare bot protection: plain requests get a 403, so the archive
is fetched with curl_cffi impersonating Chrome's TLS fingerprint. certifi supplies the
certificate bundle (macOS otherwise fails to find the local issuer).
"""

import certifi
import defusedxml.ElementTree as DefusedET
import pandas as pd
from curl_cffi import CurlOpt
from curl_cffi import requests as cffi_requests

BASKET_URL = "https://www.opec.org/basket/basketDayArchives.xml"
NAMESPACE = {"b": "http://tempuri.org/basketDayArchives.xsd"}
IMPERSONATE = "chrome120"
TIMEOUT = 30  # seconds


def get_basket() -> pd.Series:
    """Return the OPEC Reference Basket, USD per barrel, on a daily PeriodIndex."""
    session = cffi_requests.Session(impersonate=IMPERSONATE, curl_options={CurlOpt.CAINFO: certifi.where()})
    response = session.get(BASKET_URL, timeout=TIMEOUT)
    response.raise_for_status()
    root = DefusedET.fromstring(response.text)
    records = [
        (entry.get("data"), float(entry.get("val", "nan"))) for entry in root.findall("b:BasketList", NAMESPACE)
    ]
    if not records:
        raise ValueError("OPEC basket archive has no entries")
    frame = pd.DataFrame(records, columns=["date", "price"])
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.sort_values("date").reset_index(drop=True)
    print(f"OPEC Basket: {len(frame)} rows  ({frame['date'].min().date()} to {frame['date'].max().date()})")
    series = pd.Series(
        frame["price"].to_numpy(), index=pd.PeriodIndex(frame["date"], freq="D"), name="OPEC Basket"
    )
    return series[~series.index.duplicated(keep="last")]
