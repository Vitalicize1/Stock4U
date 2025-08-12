from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Tuple

import streamlit as st

# Import local workflow runner lazily inside compute to reduce import-time cost.


DEFAULT_TICKERS: List[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "AMD", "NFLX", "AVGO", "JPM", "XOM",
]


def _extract_direction_and_confidence(result: Dict[str, Any]) -> Tuple[str | None, float | None]:
    try:
        prediction_result = result.get("prediction_result", {})
        prediction = prediction_result.get("prediction", prediction_result)
        cm = result.get("confidence_metrics") or prediction_result.get("confidence_metrics") or {}
        overall_conf = cm.get("overall_confidence")
        if overall_conf is None:
            overall_conf = prediction.get("confidence")
        direction = prediction.get("direction")
        # Normalize direction text
        direction = str(direction) if direction is not None else None
        if direction:
            direction = direction.upper()
        return direction, float(overall_conf) if overall_conf is not None else None
    except Exception:
        return None, None


@st.cache_data(ttl=3600, show_spinner=False)
def compute_top_picks(
    tickers: List[str], timeframe: str = "1d", k: int = 3,
    *, low_api_mode: bool = False, fast_ta_mode: bool = True, use_ml_model: bool = False,
    _seed: int | None = None,
) -> Dict[str, Any]:
    """Run predictions across tickers and return top-k by blended confidence.

    Cached for an hour to avoid quota/cost. Pass a changing `_seed` to bypass cache.
    """
    from langgraph_flow import run_prediction  # local import to avoid heavy import at app start

    results: List[Dict[str, Any]] = []
    for t in tickers:
        try:
            res = run_prediction(t, timeframe, low_api_mode=low_api_mode, fast_ta_mode=fast_ta_mode, use_ml_model=use_ml_model)
            direction, confidence = _extract_direction_and_confidence(res)
            if direction in {"UP", "DOWN"} and isinstance(confidence, (int, float)):
                results.append({
                    "ticker": t,
                    "direction": direction,
                    "confidence": float(confidence),
                    "result": res,
                })
        except Exception:
            continue

    # Sort by confidence desc and take top-k
    results.sort(key=lambda r: r.get("confidence", 0.0), reverse=True)
    picks = results[: max(0, k)]
    return {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "timeframe": timeframe,
        "count": len(picks),
        "picks": picks,
    }


def display_top_picks(
    tickers: List[str] | None = None,
    *, timeframe: str = "1d", k: int = 3,
    low_api_mode: bool = False, fast_ta_mode: bool = True, use_ml_model: bool = False,
) -> None:
    """Render Top Picks Today section in the main dashboard.

    Uses cached computation; includes a refresh button to bypass cache on demand.
    """
    tickers = tickers or DEFAULT_TICKERS

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Top Picks Today")
        st.caption(f"Evaluated {len(tickers)} tickers · timeframe {timeframe}")
    with right:
        refresh = st.button("Refresh Picks", use_container_width=True)

    seed = int(datetime.utcnow().timestamp()) if refresh else 0
    data = compute_top_picks(
        tickers, timeframe, k,
        low_api_mode=low_api_mode, fast_ta_mode=fast_ta_mode, use_ml_model=use_ml_model,
        _seed=seed,
    )

    picks = data.get("picks", [])
    if not picks:
        st.info("No high-confidence picks available right now.")
        return

    cols = st.columns(min(len(picks), 3))
    for idx, pick in enumerate(picks):
        c = cols[idx % len(cols)]
        with c:
            st.metric(
                label=f"{pick['ticker']} — Direction",
                value=str(pick.get("direction", "-")),
                delta=f"Confidence {pick.get('confidence', 0):.1f}%",
            )
            # Optional: show a tiny summary line
            try:
                pr = pick["result"].get("prediction_result", {})
                prediction = pr.get("prediction", pr)
                reason = prediction.get("reasoning")
                if isinstance(reason, str) and len(reason) > 0:
                    preview = reason.strip().split("\n")[0]
                    if len(preview) > 140:
                        preview = preview[:137] + "..."
                    st.caption(preview)
            except Exception:
                pass


