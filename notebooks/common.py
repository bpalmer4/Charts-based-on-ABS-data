""" Common data capture functions."""

# --- imports
from pathlib import Path
import re
import requests


# --- functions
_bad_cache_pattern = re.compile('[~"#%&*:<>?/\\{|}\\.]+')
def cachefy_name(name:str) -> str:
    """Remove characters from a prospe1ctive filename
       that may cause problems."""

    new_name = re.sub(_bad_cache_pattern, '_', name)
    return new_name


def request_get(url:str) -> bytes | None:
    """Use requests to get the contents of the specified URL."""

    gotten = requests.get(url, allow_redirects=True,
                          timeout=20) # timeout in seconds
    code = gotten.status_code
    if code != 200:
        print(f'Could not get web page ({url}), error code: {code}')
        return None
    contents = gotten.content # bytes
    return contents


def save_to_cache(path:Path, contents:bytes) -> None:
    """Save bytes to the file-system."""

    path.open(mode='w', buffering=-1, encoding=None, errors=None, newline=None)
    path.write_bytes(contents)
