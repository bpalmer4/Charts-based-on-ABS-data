"""Web downloads with an on-disk cache (paths.CACHE_DIR), for providers without a reader package.

Two rules, chosen by the caller:
- get_file: a cached file is used until the server reports, through its Last-Modified
  header, that the source is newer. A server that sends no Last-Modified is downloaded
  once and then served from the cache.
- get_recent: a cached file is used while it is younger than a given age (or forever,
  with no age), for servers whose Last-Modified cannot be trusted.

Both take request headers (some servers refuse a request without a browser User-Agent),
and a fallback option: when a download fails and a cached copy exists, the cached copy is
returned with a printed warning instead of raising. Messages name the URL but never its
params, so params may carry an API key.
"""

import re
import time
from datetime import timedelta
from email.utils import parsedate_to_datetime
from hashlib import md5
from os import utime
from typing import TYPE_CHECKING

import requests

from au_econ import paths

if TYPE_CHECKING:
    from pathlib import Path

RECENT_MAX_AGE = timedelta(minutes=90)  # default age limit for get_recent callers
BROWSER_HEADERS = {  # for servers that refuse a request without a browser User-Agent
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}

TIMEOUT = 20  # seconds
OK = 200
UNSAFE_FILE_CHARACTERS = r'[~"#%&*:<>?\\{|}]+'


class HttpError(Exception):
    """A URL could not be retrieved."""


def _check(url: str, response: requests.Response) -> None:
    """Raise if the request failed; the message names the URL but never its parameters."""
    if response.status_code != OK:
        raise HttpError(f"Problem {response.status_code} accessing: {url}")


def _cache_file(url: str, params: dict[str, str] | None, prefix: str) -> tuple[str, Path]:
    """Return the full URL (with params) and the cache file that holds it."""
    request = requests.Request("GET", url, params=params).prepare()
    full_url = request.url or url
    tail = url.rsplit("/", maxsplit=1)[-1].split("?", maxsplit=1)[0]
    digest = md5(full_url.encode(), usedforsecurity=False).hexdigest()  # names the file; not security
    name = re.sub(UNSAFE_FILE_CHARACTERS, "", f"{prefix}--{digest}--{tail}")
    paths.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return full_url, paths.CACHE_DIR / name


def _cached_instead(url: str, file: Path, error: Exception) -> bytes:
    """Return the cached copy after a failed download, saying so."""
    print(f"WARNING: download failed ({type(error).__name__}); using the cached copy of {url}")
    return file.read_bytes()


def get_recent(
    url: str,
    params: dict[str, str] | None,
    prefix: str,
    max_age: timedelta | None,
    timeout: int = TIMEOUT,
    *,
    headers: dict[str, str] | None = None,
    fallback: bool = False,
) -> bytes:
    """Return the contents of url (with query params), from the cache if saved less than max_age ago.

    max_age None keeps a cached file forever (for data that cannot change).
    """
    full_url, file = _cache_file(url, params, prefix)
    if file.is_file() and (max_age is None or time.time() - file.stat().st_mtime < max_age.total_seconds()):
        return file.read_bytes()
    try:
        response = requests.get(full_url, headers=headers, allow_redirects=True, timeout=timeout)
        _check(url, response)
    except (requests.RequestException, HttpError) as error:
        if fallback and file.is_file():
            return _cached_instead(url, file, error)
        raise
    file.write_bytes(response.content)
    return response.content


def get_file(
    url: str,
    params: dict[str, str] | None = None,
    prefix: str = "cache",
    timeout: int = TIMEOUT,
    *,
    headers: dict[str, str] | None = None,
    fallback: bool = False,
) -> bytes:
    """Return the contents of url (with query params), from the cache when it is fresh."""
    full_url, file = _cache_file(url, params, prefix)
    try:
        head = requests.head(full_url, headers=headers, allow_redirects=True, timeout=TIMEOUT)
        _check(url, head)
        modified = head.headers.get("Last-Modified")
        source_time = None if modified is None else parsedate_to_datetime(modified).timestamp()
        if file.is_file() and (source_time is None or source_time <= file.stat().st_mtime):
            return file.read_bytes()
        response = requests.get(full_url, headers=headers, allow_redirects=True, timeout=timeout)
        _check(url, response)
    except (requests.RequestException, HttpError) as error:
        if fallback and file.is_file():
            return _cached_instead(url, file, error)
        raise
    file.write_bytes(response.content)
    if source_time is not None:
        utime(file, (source_time, source_time))
    return response.content
