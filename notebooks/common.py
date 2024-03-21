"""Common data capture functions.
Data capture over the intrwebs with a cache facility."""

# --- imports
from hashlib import md5
import re
from datetime import datetime, timezone
from os import utime
from pathlib import Path

import pandas as pd
import requests


class HttpError(Exception):
    """We have a problem retrieving data from HTTP."""


class CacheError(Exception):
    """We have a problem retrieving data from the cache."""


def check_response(url: str, response: requests.Response) -> None:
    """Raise an Exception if we could not retrieve URL."""
    code = response.status_code
    if code != 200 or response.headers is None:
        raise HttpError(f"Problem {code} accessing: {url}.")


def request_get(url: str) -> bytes:
    """Use requests to get the contents of the specified URL."""
    gotten = requests.get(url, allow_redirects=True, timeout=20)  # timeout in seconds
    check_response(url, gotten)
    contents = gotten.content  # bytes
    return contents


def save_to_cache(file: Path, contents: bytes, verbose: bool) -> None:
    """Save bytes to the file-system."""
    if file.exists():
        if verbose:
            print("Removing old cache file.")
        file.unlink()
    if verbose:
        print(f"Saving to cache: {file}")
    file.open(mode="w", buffering=-1, encoding=None, errors=None, newline=None)
    file.write_bytes(contents)


def retrieve_from_cache(file: Path) -> bytes:
    """Retrieve bytes from file-system."""
    if not file.exists() or not file.is_file():
        raise CacheError(f"Cached file not available: {file.name}")
    return file.read_bytes()


def get_file(
    url: str,
    cache_dir: Path,
    cache_name_prefix: str = "cache",
    verbose: bool = False,
) -> bytes:
    """Get file from URL or local file-system cache, depending on freshness.
    Note: we create the cache_dir if it does not exist.
    Returns: the contents of the file as bytes."""

    def get_fpath() -> Path:
        """Convert URL string into a cache file name,
        then return as a Path object."""
        bad_cache_pattern = r'[~"#%&*:<>?\\{|}]+'  # chars to remove from name
        hash_name = md5(url.encode("utf-8")).hexdigest()
        tail_name = url.split("/")[-1].split("?")[0]
        file_name = re.sub(
            bad_cache_pattern, "", f"{cache_name_prefix}--{hash_name}--{tail_name}"
        )
        return Path(cache_dir / file_name)

    # create and check cache_dir is a directory
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not cache_dir.is_dir():
        raise CacheError(f"Cache path is not a directory: {cache_dir.name}")

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
    if target_mtime is None or (  # cache is empty, or
        source_mtime is not None
        and source_mtime > target_mtime  # URL is fresher than cache
    ):
        if verbose:
            print("About to download and cache the latest data.")
        url_bytes = request_get(url)  # will raise exception if it fails
        save_to_cache(file_path, url_bytes, verbose)
        # - change file mod time to reflect mtime at URL
        if source_mtime is not None:
            unixtime = source_mtime.value / 1_000_000_000  # convert to seconds
            utime(file_path, (unixtime, unixtime))
        return url_bytes

    # return the data that has been cached previously
    if verbose:
        print("Retrieving data from cache.")
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
    
    # do the testing
    for url_ in (URL1, URL2):
        content = get_file(url_, path, verbose=True)
        print(len(content))
