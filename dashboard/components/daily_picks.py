from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, List, Optional

import streamlit as st

from langgraph_flow import run_prediction


DEFAULT_TICKERS: List[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "BRK-B", "JPM",
    "V", "UNH", "XOM", "HD", "LLY", "MA", "PG", "ORCL", "COST", "KO",
]


@dataclass
class Pick:
    ticker: str
    direction: str
    confidence: float
    timeframe: str


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


def _load_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        txt = path.read_text(encoding="utf-8").strip()
        return json.loads(txt) if txt else {}
    except Exception:
        return {}


def _save_cache(path: Path, payload: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass


def _is_stale(payload: dict, max_age_hours: int = 24) -> bool:
    try:
        ts = payload.get("generated_at")
        if not ts:
            return True
        if ts.endswith("Z"):
            ts = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts)
        return datetime.utcnow() >= (dt.replace(tzinfo=None) + timedelta(hours=max_age_hours))
    except Exception:
        return True


def _build_payload(picks: List[Pick]) -> dict:
    return {
        "generated_at": _now_iso(),
        "picks": [
            {
                "ticker": p.ticker,
                "direction": p.direction,
                "confidence": p.confidence,
                "timeframe": p.timeframe,
            }
            for p in picks
        ],
    }


def _normalize_universe_text(text: Optional[str]) -> List[str]:
    if not text:
        return []
    raw = [x.strip().upper() for x in text.replace("\n", ",").split(",")]
    return [t for t in raw if t]


def _generate_picks(
    tickers: Iterable[str],
    timeframe: str = "1d",
    top_n: int = 3,
    low_api_mode: bool = False,
    fast_ta_mode: bool = False,
) -> List[Pick]:
    rows: List[Pick] = []
    for t in tickers:
        try:
            with st.spinner(f"Evaluating {t}..."):
                result = run_prediction(t, timeframe, low_api_mode=low_api_mode, fast_ta_mode=fast_ta_mode, use_ml_model=False)
            conf = _overall_confidence(result)
            direction = _direction(result) or "Unknown"
            if conf is not None:
                rows.append(Pick(ticker=t, direction=direction, confidence=float(conf), timeframe=timeframe))
        except Exception as e:
            st.info(f"Skipping {t}: {e}")
    # Sort by confidence desc and keep top_n
    rows.sort(key=lambda r: (r.confidence if r.confidence is not None else 0.0), reverse=True)
    return rows[: top_n]


def display_daily_picks(
    tickers: Optional[Iterable[str]] = None,
    timeframe: str = "1d",
    top_n: int = 3,
    cache_path: str = "cache/daily_picks.json",
    auto_refresh_once: bool = True,
    custom_tickers_text: Optional[str] = None,
    max_scan: int = 200,
    rotate_daily: bool = True,
) -> None:
    st.subheader("Daily Top Picks")
    path = Path(cache_path)
    data = _load_cache(path)

    # Build universe
    universe: List[str] = []
    if tickers:
        universe.extend([str(t).strip().upper() for t in tickers if str(t).strip()])
    if custom_tickers_text:
        universe.extend(_normalize_universe_text(custom_tickers_text))
    if not universe:
        universe = list(DEFAULT_TICKERS)
    universe = sorted(list(dict.fromkeys(universe)))  # unique, stable order

    # Deterministic rotation when scanning very large universes
    scan_list = list(universe)
    if max_scan and len(scan_list) > max_scan:
        if rotate_daily:
            # Use YYYYMMDD hash to rotate
            day_key = datetime.utcnow().strftime("%Y%m%d")
            offset = sum(ord(c) for c in day_key) % len(scan_list)
            scan_list = scan_list[offset:] + scan_list[:offset]
        scan_list = scan_list[:max_scan]

    # Auto refresh once per session if stale
    if auto_refresh_once and _is_stale(data) and not st.session_state.get("_daily_picks_refreshed", False):
        with st.spinner("Generating today's picks..."):
            picks = _generate_picks(scan_list, timeframe=timeframe)
        payload = _build_payload(picks)
        _save_cache(path, payload)
        data = payload
        st.session_state["_daily_picks_refreshed"] = True

    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("Refresh Picks"):
            with st.spinner("Refreshing picks..."):
                picks = _generate_picks(scan_list, timeframe=timeframe)
            payload = _build_payload(picks)
            _save_cache(path, payload)
            data = payload
    with col_b:
        st.caption("Picks use the daily timeframe across your ticker list and are cached daily.")

    # Render cards
    picks = (data or {}).get("picks") or []
    if not picks:
        st.info("No picks yet. Click Refresh Picks to generate today's recommendations.")
        return

    grid = st.columns(top_n)
    for idx, p in enumerate(picks[:top_n]):
        with grid[idx]:
            st.metric(label=f"{p.get('ticker', '')}", value=str(p.get("direction", "Unknown")), delta=f"{float(p.get('confidence', 0)):.1f}%")


