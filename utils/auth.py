from __future__ import annotations

import os
import time
from typing import Optional

from fastapi import Header, HTTPException, Request


_RATE_BUCKET: dict[str, list[float]] = {}


def _cfg_token() -> Optional[str]:
    tok = os.getenv("API_TOKEN")
    return tok.strip() if tok else None


def _rate_key(req: Request, token: Optional[str]) -> str:
    # Prefer token key; fallback to client host
    if token:
        return f"tok:{token}"
    client = getattr(req, "client", None)
    host = getattr(client, "host", "unknown") if client else "unknown"
    return f"ip:{host}"


def _check_rate_limit(key: str, limit_per_minute: int) -> None:
    now = time.time()
    window_start = now - 60.0
    buf = _RATE_BUCKET.setdefault(key, [])
    # Drop old
    _RATE_BUCKET[key] = [t for t in buf if t >= window_start]
    if len(_RATE_BUCKET[key]) >= max(1, limit_per_minute):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please slow down.")
    _RATE_BUCKET[key].append(now)


async def auth_guard(request: Request, authorization: Optional[str] = Header(None)) -> None:
    """Simple bearer-token auth + per-minute rate limiting.

    If API_TOKEN is unset, auth is disabled (open mode).
    Rate limiting still applies (by IP) with default 60 req/min.
    """
    cfg_token = _cfg_token()
    limit = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))

    # Extract bearer token from header if present
    provided: Optional[str] = None
    if authorization and authorization.lower().startswith("bearer "):
        provided = authorization.split(" ", 1)[1].strip()

    # Rate limit first (use provided token or IP)
    key = _rate_key(request, provided)
    _check_rate_limit(key, limit)

    # Auth check only if configured
    if cfg_token is None:
        return
    if not provided or provided != cfg_token:
        raise HTTPException(status_code=401, detail="Unauthorized")


