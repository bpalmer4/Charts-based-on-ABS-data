"""CME Group: daily futures settlement curves, from the public settlements JSON on cmegroup.com.

The settlements pages are Akamai-protected, so requests go through a curl_cffi session
impersonating Chrome's TLS fingerprint, which first visits a settlements page for its
cookies. Products are identified by CME's numeric product ids.
"""

import certifi
import pandas as pd
from curl_cffi import CurlOpt
from curl_cffi import requests as cffi_requests

SETTLEMENTS_BASE = "https://www.cmegroup.com/CmeWS/mvc/Settlements/Futures"
REFERER = "https://www.cmegroup.com/markets/energy/refined-products/singapore-gasoil-swap-futures.settlements.html"
IMPERSONATE = "chrome120"
TIMEOUT = 30  # seconds
PAGE_SIZE = 500
NO_SETTLE = ("", "-")


def get_settlement_curve(product_id: int) -> tuple[pd.Series, pd.Timestamp]:
    """Return a product's latest settlement curve (contract month to price) and its trade date."""
    session = cffi_requests.Session(impersonate=IMPERSONATE, curl_options={CurlOpt.CAINFO: certifi.where()})
    session.get(REFERER, timeout=TIMEOUT)
    headers = {"Referer": REFERER, "Accept": "application/json"}

    dates = session.get(f"{SETTLEMENTS_BASE}/TradeDate/{product_id}", headers=headers, timeout=TIMEOUT)
    dates.raise_for_status()
    trade_dates = dates.json()
    if not trade_dates:
        raise ValueError(f"CME product {product_id}: no trade dates available")
    trade_date = trade_dates[0][0]  # [MM/DD/YYYY, report type] pairs, newest first

    settlements = session.get(
        f"{SETTLEMENTS_BASE}/Settlements/{product_id}/FUT",
        params={"strategy": "DEFAULT", "tradeDate": trade_date, "pageSize": PAGE_SIZE},
        headers=headers,
        timeout=TIMEOUT,
    )
    settlements.raise_for_status()

    records: dict[pd.Period, float] = {}
    for row in settlements.json().get("settlements", []):
        settle = (row.get("settle") or "").replace(",", "")
        if settle in NO_SETTLE:
            continue
        try:
            # months come as e.g. "APR 26"; pd.Period("APR 26", "M") would silently parse to year 1
            period = pd.Period(pd.to_datetime(row.get("month", ""), format="%b %y"), freq="M")
            price = float(settle)
        except ValueError, KeyError:
            continue
        records[period] = price
    return pd.Series(records).sort_index(), pd.Timestamp(trade_date)
