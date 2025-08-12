from __future__ import annotations

import os
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

from langgraph_flow import run_prediction


DEFAULT_TICKERS: List[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "BRK-B", "JPM",
    "V", "UNH", "XOM", "HD", "LLY", "MA", "PG", "ORCL", "COST", "KO",
]


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _overall_confidence(result: dict) -> Optional[float]:
    try:
        pr = result.get("prediction_result", {})
        prediction = pr.get("prediction", pr)
        cm = result.get("confidence_metrics") or pr.get("confidence_metrics") or {}
        conf = cm.get("overall_confidence")
        if conf is None:
            conf = prediction.get("confidence")
        return float(conf) if conf is not None else None
    except Exception:
        return None


def _direction(result: dict) -> Optional[str]:
    try:
        pr = result.get("prediction_result", {})
        prediction = pr.get("prediction", pr)
        d = prediction.get("direction")
        return str(d) if d is not None else None
    except Exception:
        return None


def compute_top_picks(
    tickers: Iterable[str],
    timeframe: str = "1d",
    top_n: int = 3,
    max_scan: Optional[int] = None,
    rotate_daily: bool = True,
    low_api_mode: bool = False,
    fast_ta_mode: bool = False,
) -> List[dict]:
    universe = [str(t).strip().upper() for t in tickers if str(t).strip()]
    if not universe:
        universe = list(DEFAULT_TICKERS)
    # Optional rotation for large lists
    if max_scan and len(universe) > max_scan:
        if rotate_daily:
            key = datetime.utcnow().strftime("%Y%m%d")
            offset = sum(ord(c) for c in key) % len(universe)
            universe = universe[offset:] + universe[:offset]
        universe = universe[:max_scan]

    rows: List[Tuple[str, str, float]] = []
    for t in universe:
        try:
            result = run_prediction(
                t,
                timeframe,
                low_api_mode=low_api_mode,
                fast_ta_mode=fast_ta_mode,
                use_ml_model=False,
            )
            conf = _overall_confidence(result)
            direction = _direction(result) or "Unknown"
            if conf is not None:
                rows.append((t, direction, float(conf)))
        except Exception:
            # Skip errors to keep batch running
            continue

    rows.sort(key=lambda r: r[2], reverse=True)
    picks = [
        {"ticker": t, "direction": d, "confidence": c, "timeframe": timeframe}
        for (t, d, c) in rows[: top_n]
    ]
    return picks


def _write_payload(path: Path, picks: List[dict]) -> dict:
    payload = {"generated_at": _now_iso(), "picks": picks}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_daily_picks_job() -> dict:
    """Run a scheduled job to compute and cache daily top picks.

    Reads configuration from environment variables:
      - DAILY_PICKS_TICKERS: comma-separated universe (optional)
      - DAILY_PICKS_TIMEFRAME: default "1d"
      - DAILY_PICKS_TOP_N: default 3
      - DAILY_PICKS_MAX_SCAN: default 200
      - DAILY_PICKS_ROTATE: default 1 (enable rotation)
      - DAILY_PICKS_LOW_API: default 0
      - DAILY_PICKS_FAST_TA: default 0
      - DAILY_PICKS_PATH: default cache/daily_picks.json
    """
    tickers_env = os.getenv("DAILY_PICKS_TICKERS", "")
    tickers = [t.strip().upper() for t in tickers_env.split(",") if t.strip()] if tickers_env else DEFAULT_TICKERS
    timeframe = os.getenv("DAILY_PICKS_TIMEFRAME", "1d")
    top_n = int(os.getenv("DAILY_PICKS_TOP_N", "3"))
    max_scan = int(os.getenv("DAILY_PICKS_MAX_SCAN", "200"))
    rotate = os.getenv("DAILY_PICKS_ROTATE", "1") == "1"
    low_api_mode = os.getenv("DAILY_PICKS_LOW_API", "0") == "1"
    fast_ta_mode = os.getenv("DAILY_PICKS_FAST_TA", "0") == "1"
    out_path = Path(os.getenv("DAILY_PICKS_PATH", "cache/daily_picks.json"))

    picks = compute_top_picks(
        tickers=tickers,
        timeframe=timeframe,
        top_n=top_n,
        max_scan=max_scan,
        rotate_daily=rotate,
        low_api_mode=low_api_mode,
        fast_ta_mode=fast_ta_mode,
    )
    return _write_payload(out_path, picks)


