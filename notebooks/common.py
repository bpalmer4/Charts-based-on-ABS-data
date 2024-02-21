""" Common data capture functions."""

# --- imports
from hashlib import md5
import re
from datetime import datetime, timezone
from os import utime
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


def retrieve_from_cache(file: Path) -> bytes:
    """Retrieve bytes from file-system."""
    if not file.exists() or not file.is_file():
        raise CacheError("Cached file not available?")
    return file.read_bytes()


def get_file(url: str, cache_dir: Path) -> bytes:
    """Get file from URL or local file-system cache, depending on freshness."""

    def get_fpath() -> Path:
        """Convert URL string into a cache file name,
        then return as a Path object."""
        bad_cache_pattern = r'[~"#%&*:<>?\\{|}]+'  # remove these chars from name
        hash_name = md5(url.encode("utf-8")).hexdigest()
        tail_name = url.split("/")[-1]
        file_name = re.sub(bad_cache_pattern, "", f"{hash_name}--{tail_name}")
        return Path(cache_dir / file_name)

    # sanity checks
    if not cache_dir.is_dir():
        raise ValueError("Cache path is not a directory")

    # get URL modification time in UTC
    response = requests.head(url, allow_redirects=True, timeout=20)
    check_response(url, response)
    source_time = response.headers.get("Last-Modified", None)
    source_mtime = (
        None if source_time is None else pd.to_datetime(source_time, utc=True)
    )

    # get cache modification time in UTC
    target_mtime: datetime | None = None
    file_path = get_fpath()
    if file_path.exists() and file_path.is_file():
        target_mtime = pd.to_datetime(
            datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc), utc=True
        )

    # get and save URL source data
    if target_mtime is None or (  # cache is empty
        source_mtime is not None
        and source_mtime > target_mtime  # URL is fresher than cache
    ):
        print("About to download and cache the latest data.")
        url_bytes = request_get(url)  # will raise exception if it fails
        save_to_cache(file_path, url_bytes)
        # - change file mod time to reflect mtime at URL
        if source_mtime is not None:
            unixtime = source_mtime.value / 1_000_000_000  # convert to seconds
            utime(file_path, (unixtime, unixtime))
        return url_bytes

    # return the data that has been cached previously
    print(f"Retrieving data from cache.")
    return retrieve_from_cache(file_path)


# --- preliminary testing:
DO_TEST = False

if __name__ == "__main__" and DO_TEST:
    # prepare for the test cases
    URL1 = (  # ABS
        "https://www.abs.gov.au/statistics/labour/employment-and-unemployment/"
        "labour-force-australia/nov-2023/6202001.xlsx"
    )
    URL2 = "https://www.rba.gov.au/statistics/tables/xls/a02hist.xls"  # RBA
    TEST_CACHE_DIR = "./TEST_CACHE/"
    path = Path(TEST_CACHE_DIR)
    path.mkdir(parents=True, exist_ok=True)

    # do the testing
    for url_ in (URL1, URL2):
        content = get_file(url_, path)
        print(len(content))
