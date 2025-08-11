import os
import pickle
import time
import hashlib
from typing import Any, Optional


CACHE_DIR = os.path.join("cache", "results")
os.makedirs(CACHE_DIR, exist_ok=True)


def _key_to_path(key: str) -> str:
    """Convert an arbitrary cache key to a safe file path."""
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{digest}.pkl")


def get_cached_result(key: str, ttl_seconds: int = 900) -> Optional[dict]:
    """
    Read a cached result if it exists and is within the TTL.

    Args:
        key: Unique cache key
        ttl_seconds: Time-to-live in seconds (default 15 minutes)

    Returns:
        Cached dict if fresh; otherwise None
    """
    path = _key_to_path(key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        ts = payload.get("_ts", 0)
        if (time.time() - ts) <= ttl_seconds:
            return payload.get("data")
    except Exception:
        return None
    return None


def set_cached_result(key: str, data: dict) -> None:
    """Persist a result to cache with a timestamp."""
    path = _key_to_path(key)
    try:
        with open(path, "wb") as f:
            pickle.dump({"_ts": time.time(), "data": data}, f)
    except Exception:
        # Best-effort caching only
        pass


def invalidate_cached_result(key: str) -> bool:
    """Remove a cached result if present. Returns True if removed."""
    path = _key_to_path(key)
    try:
        if os.path.exists(path):
            os.remove(path)
            return True
    except Exception:
        return False
    return False

