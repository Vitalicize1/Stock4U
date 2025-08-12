from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional


def get_logger(name: str = "stock4u") -> logging.Logger:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        ch = logging.StreamHandler()
        ch.setLevel(level)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


# --------- Simple JSONL metrics sink ---------
_METRICS_FILE = os.getenv("METRICS_FILE", os.path.join("cache", "metrics", "metrics.jsonl"))


def _write_jsonl(path: str, record: Dict) -> None:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # Best effort
        pass


def log_metric(name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
    rec = {
        "ts": int(time.time() * 1000),
        "event": "metric",
        "name": name,
        "value": value,
        "tags": tags or {},
    }
    _write_jsonl(_METRICS_FILE, rec)


def increment(name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
    log_metric(name, float(value), tags)


@contextmanager
def time_block(name: str, tags: Optional[Dict[str, str]] = None):
    start = time.time()
    try:
        yield
    finally:
        elapsed_ms = int((time.time() - start) * 1000)
        log_metric(name, float(elapsed_ms), tags)


# --------- File-based alerts ---------
_ALERTS_FILE = os.getenv("ALERTS_FILE", os.path.join("cache", "metrics", "alerts.log"))
_ALERT_ENABLED = os.getenv("ALERT_ENABLED", "true").lower() == "true"


def emit_alert(title: str, body: str, level: str = "warning", kv: Optional[Dict[str, str]] = None) -> None:
    if not _ALERT_ENABLED:
        return
    rec = {
        "ts": int(time.time() * 1000),
        "level": level,
        "title": title,
        "body": body,
        "kv": kv or {},
    }
    try:
        # ensure directory exists even if METRICS_FILE is customized in tests
        Path(_ALERTS_FILE).parent.mkdir(parents=True, exist_ok=True)
        _write_jsonl(_ALERTS_FILE, rec)
    finally:
        # Echo to stderr for visibility
        try:
            print(f"ALERT[{level.upper()}]: {title} - {body}", file=sys.stderr)
        except Exception:
            pass



