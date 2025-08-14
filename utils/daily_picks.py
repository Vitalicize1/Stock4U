from __future__ import annotations

import os
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from langgraph_flow import run_prediction


# Curated list of top stocks for daily picks
DAILY_PICKS_UNIVERSE: List[str] = [
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
    tickers: List[str] = None,
    timeframe: str = "1d",
    top_n: int = 3,
    low_api_mode: bool = True,  # Default to low API mode for background jobs
    fast_ta_mode: bool = True,  # Default to fast TA mode for background jobs
) -> List[dict]:
    """Compute top picks from the universe (optimized for background execution)."""
    universe = tickers or DAILY_PICKS_UNIVERSE
    
    rows: List[tuple[str, str, float]] = []
    for ticker in universe:
        try:
            result = run_prediction(
                ticker,
                timeframe,
                low_api_mode=low_api_mode,
                fast_ta_mode=fast_ta_mode,
                use_ml_model=False,
            )
            conf = _overall_confidence(result)
            direction = _direction(result) or "Unknown"
            if conf is not None:
                rows.append((ticker, direction, float(conf)))
        except Exception:
            # Skip errors to keep batch running
            continue

    # Sort by confidence (highest first) and take top N
    rows.sort(key=lambda r: r[2], reverse=True)
    picks = [
        {"ticker": t, "direction": d, "confidence": c, "timeframe": timeframe}
        for (t, d, c) in rows[:top_n]
    ]
    return picks


def _write_payload(path: Path, picks: List[dict]) -> dict:
    """Write picks to cache file."""
    payload = {"generated_at": _now_iso(), "picks": picks}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_daily_picks_job() -> dict:
    """Run a scheduled job to compute and cache daily top picks (background optimized).
    
    Environment variables:
    - DAILY_PICKS_TIMEFRAME: default "1d"
    - DAILY_PICKS_TOP_N: default 3
    - DAILY_PICKS_LOW_API: default 1 (optimized for background)
    - DAILY_PICKS_FAST_TA: default 1 (optimized for background)
    - DAILY_PICKS_PATH: default cache/daily_picks.json
    """
    timeframe = os.getenv("DAILY_PICKS_TIMEFRAME", "1d")
    top_n = int(os.getenv("DAILY_PICKS_TOP_N", "3"))
    low_api_mode = os.getenv("DAILY_PICKS_LOW_API", "1") == "1"  # Default to True for background
    fast_ta_mode = os.getenv("DAILY_PICKS_FAST_TA", "1") == "1"  # Default to True for background
    out_path = Path(os.getenv("DAILY_PICKS_PATH", "cache/daily_picks.json"))

    print(f"🔄 Computing daily picks (top {top_n}) in background...")
    picks = compute_top_picks(
        timeframe=timeframe,
        top_n=top_n,
        low_api_mode=low_api_mode,
        fast_ta_mode=fast_ta_mode,
    )
    
    payload = _write_payload(out_path, picks)
    print(f"✅ Daily picks written: {out_path}")
    
    # Log the daily picks for tracking
    try:
        from utils.prediction_logger import log_daily_picks
        log_daily_picks(picks)
        print(f"📊 Daily picks logged for accuracy tracking")
    except Exception as e:
        print(f"Warning: Could not log daily picks: {e}")
    
    return payload


if __name__ == "__main__":
    run_daily_picks_job()
