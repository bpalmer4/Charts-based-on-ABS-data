"""Every filesystem location, anchored to the project root rather than the working directory.

Nothing here points into notebooks/: the package keeps its own keys and caches, so the
notebooks can be deleted without breaking it.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHARTS_DIR = PROJECT_ROOT / "CHARTS"
LOGS_DIR = PROJECT_ROOT / "LOGS"
KEYS_DIR = PROJECT_ROOT / "KEYS"  # fred.api, EIA-API-KEY.txt (gitignored)
CACHE_DIR = PROJECT_ROOT / "CACHE"  # http_cache downloads (gitignored)
READABS_CACHE = PROJECT_ROOT / ".readabs_cache"
SDMXABS_CACHE = PROJECT_ROOT / ".sdmxabs_cache"
