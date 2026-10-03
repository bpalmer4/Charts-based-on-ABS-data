"""AFSA: personal insolvency statistics, as CSV files from the data.gov.au catalogue (CKAN).

AFSA keeps data.gov.au in step with its own website. Each file is found by its exact
resource name in the catalogue, then downloaded through http_cache; AFSA reuses the same
file names each release, and the server's Last-Modified decides when to download again.
When the catalogue cannot be reached, the cached copy of the file is used, with a warning.
The CSV files are used because the XLSX versions are unreliable.
"""

import io

import pandas as pd
import requests

from au_econ import paths
from au_econ.sources.http_cache import get_file

PACKAGE_URL = "https://data.gov.au/data/api/3/action/package_show"
PREFIX = "afsa"
TIMEOUT = 60  # seconds
DOWNLOAD_TIMEOUT = 300  # the monthly file is large
MONTHLY = ("monthly-personal-insolvency-statistics", "Monthly personal insolvency time series")
QUARTERLY = ("quarterly_personal_insolvency_statistics", "Quarterly personal insolvency statistics per quarter")


def _resource_url(package: str, name: str) -> str:
    """Return the URL of the one CSV resource in a data.gov.au package with exactly this name."""
    response = requests.get(PACKAGE_URL, params={"id": package}, timeout=TIMEOUT)
    response.raise_for_status()
    urls = [
        str(resource["url"])
        for resource in response.json()["result"]["resources"]
        if resource.get("format", "").upper() == "CSV" and resource.get("name") == name
    ]
    if len(urls) != 1:
        raise ValueError(f"data.gov.au {package}: expected one CSV named {name!r}, found {len(urls)}")
    return urls[0]


def _cached(package: str, error: Exception) -> bytes:
    """Return the cached copy of a package's file after a failed catalogue lookup, saying so."""
    cached = sorted(paths.CACHE_DIR.glob(f"{PREFIX}-{package}--*.csv"), key=lambda file: file.stat().st_mtime)
    if not cached:
        raise RuntimeError(f"data.gov.au unreachable and no cached {package} file") from error
    print(f"WARNING: could not check data.gov.au ({type(error).__name__}); using cached {cached[-1].name}")
    return cached[-1].read_bytes()


def get_csv(dataset: tuple[str, str]) -> pd.DataFrame:
    """Return one AFSA CSV file (MONTHLY or QUARTERLY) as a frame."""
    package, name = dataset
    try:
        url = _resource_url(package, name)
    except (requests.RequestException, KeyError, ValueError) as error:
        content = _cached(package, error)
    else:
        content = get_file(url, prefix=f"{PREFIX}-{package}", timeout=DOWNLOAD_TIMEOUT, fallback=True)
    return pd.read_csv(io.BytesIO(content))
