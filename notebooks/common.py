""" Common data capture functions."""

# --- imports
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


class HttpError(Exception):
    """Indicate we had a problem retrieving data from HTTP."""


class CacheError(Exception):
    """Indicate we had a problem retrieving data from the cache."""


def check_response(url: str, response: requests.Response) -> None:
    """Raise an Exception if we could not retrieve URL."""
    code = response.status_code
    if code != 200 or response.headers is None:
        raise HttpError(f"Problem accessing: {url}.")


def request_get(url: str) -> bytes:
    """Use requests to get the contents of the specified URL."""
    gotten = requests.get(url, allow_redirects=True, timeout=20)  # timeout in seconds
    check_response(url, gotten)
    contents = gotten.content  # bytes
    return contents


def save_to_cache(file: Path, contents: bytes) -> None:
    """Save bytes to the file-system."""

    if file.exists():
        print("Removing old cache file.")
        file.unlink()

    file.open(mode="w", buffering=-1, encoding=None, errors=None, newline=None)
    print(f"Saving to cache: {file}")
    file.write_bytes(contents)


def get_file(url: str, dir_path: Path) -> bytes:
    """Get file from URL or local file-system cache."""

    def get_fpath(url: str) -> Path:
        """Convert URL string into a cache file name,
        then return as a Path object."""

        bad_cache_pattern = r'[~"#%&*:<>?\\{|}]+'
        file_name = re.sub(
            bad_cache_pattern,
            "",
            f"{hashlib.md5(url.encode('utf-8')).hexdigest()}--{url.split(r'/')[-1]}",
        )
        return Path(dir_path / f"{file_name}")

    # sanity checks
    if not dir_path.is_dir():
        raise ValueError("Cache path is not a directory")

    # get URL modification time in UTC
    response = requests.head(
        url, allow_redirects=True, timeout=20
    )  # only get the header
    check_response(url, response)
    source_mtime = pd.to_datetime(response.headers["Last-Modified"], utc=True)
    print(f"Source modification date: {source_mtime}")

    # get cache modification time in UTC
    target_mtime: datetime | None = None
    file_path = get_fpath(url)
    if file_path.exists() and file_path.is_file():
        target_mtime = pd.to_datetime(
            datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc), utc=True
        )

    # get and save URL source
    one_day_old = pd.Timestamp.utcnow() - pd.DateOffset(days=1)
    if (
        target_mtime is None            # cache is empty
        or source_mtime > target_mtime  # URL is definitely fresher than cache
        or one_day_old > target_mtime   # prophylaxis: our cache might just be stale
    ):
        print("About to download and cache the latest data.")
        file_bytes = request_get(url)
        if file_bytes is not None:
            save_to_cache(file_path, file_bytes)
            return file_bytes

    # return the data that has been cached
    file_path = get_fpath(url)
    print(f"Retrieving data from the cache file: {file_path}")
    if not file_path.exists() or not file_path.is_file():
        raise CacheError("Cached file not available?")
    file_bytes = file_path.read_bytes()
    return file_bytes


### --- preliminary testing
DO_TEST = False

if __name__ == "__main__" and DO_TEST:
    URL1 = (  # ABS
        "https://www.abs.gov.au/statistics/labour/employment-and-unemployment/"
        "labour-force-australia/nov-2023/6202001.xlsx"
    )
    URL2 = (  # RBA
        'https://www.rba.gov.au/statistics/tables/xls/a02hist.xls'
    )

    TEST_CACHE_DIR = "./TEST_CACHE/"
    path = Path(TEST_CACHE_DIR)
    for url_ in (URL1, URL2):
        path.mkdir(parents=True, exist_ok=True)
        content = get_file(url_, path)
        print(len(content))
