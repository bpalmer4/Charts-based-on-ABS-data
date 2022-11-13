# common.py

import re
import requests
from typing import Optional
from pathlib import Path


_bad_cache_oattern = re.compile('[~"#%&*:<>?/\\{|}\.]+')
def cachefy_name(name: str) -> str:
    new_name = re.sub(_bad_cache_oattern, '_', name)
    return new_name


def request_get(url: str) -> Optional[bytes]:
    gotten = requests.get(url, allow_redirects=True)
    code = gotten.status_code
    if code != 200:
        print(f'Could not get web page ({url}), error code: {code}')
        return None
    contents = gotten.content # bytes
    return contents


def save_to_cache(path:Path, contents:bytes):
    path.open(mode='w', buffering=-1, encoding=None, errors=None, newline=None)
    path.write_bytes(contents)
