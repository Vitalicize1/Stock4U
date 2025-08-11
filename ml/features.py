from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np


def _as_float(value: Any, default: float = 0.0) -> float:
    """Best-effort conversion to float.

    - If value is a pandas Series/Index, numpy array or list/tuple, use the last element
    - If value is not directly convertible, fall back to default
    """
    if value is None:
        return default
    # Pandas Series / Index duck-typing
    try:
        if hasattr(value, "iloc") and callable(getattr(value, "iloc")):
            try:
                value = value.iloc[-1]
            except Exception:
                pass
    except Exception:
        pass
    # Numpy array / list / tuple
    if isinstance(value, (list, tuple, np.ndarray)):
        try:
            if len(value) > 0:
                value = value[-1]
        except Exception:
            return default
    try:
        return float(value)
    except Exception:
        return default


def build_features_from_state(state: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, float]]:
    """Extract a simple feature vector from the existing workflow state.

    Uses technical indicators, sentiment, and simple recent price change signals when available.
    Returns (X, features_dict) where X is shape (n_features,).
    """
    tech = state.get("enhanced_technical_analysis") or state.get("technical_analysis", {})
    senti = state.get("sentiment_integration", {})
    price_data = (state.get("data", {}) or {}).get("price_data", {})

    indicators = (tech.get("indicators") or {}) if isinstance(tech, dict) else {}
    trend = (tech.get("trend_analysis") or {}) if isinstance(tech, dict) else {}
    trading = (tech.get("trading_signals") or {}) if isinstance(tech, dict) else {}

    integrated = (senti.get("integrated_analysis") or {}) if isinstance(senti, dict) else {}

    # Core features with safe defaults
    f: Dict[str, float] = {
        "rsi": _as_float(indicators.get("rsi"), 50.0),
        "macd": _as_float(indicators.get("macd"), 0.0),
        "macd_hist": _as_float(indicators.get("macd_histogram"), 0.0),
        "sma_20": _as_float(indicators.get("sma_20"), 0.0),
        "sma_50": _as_float(indicators.get("sma_50"), 0.0),
        "sma_200": _as_float(indicators.get("sma_200"), 0.0),
        "trend_strength": _as_float(trend.get("trend_strength"), 0.0),
        "adx_strength": _as_float(trend.get("adx_strength"), 0.0),
        "signal_strength": _as_float(trading.get("signal_strength"), 0.0),
        "stoch_k": _as_float(indicators.get("stoch_k"), 50.0),
        "stoch_d": _as_float(indicators.get("stoch_d"), 50.0),
        "bb_upper": _as_float(indicators.get("bb_upper"), 0.0),
        "bb_lower": _as_float(indicators.get("bb_lower"), 0.0),
        "bb_middle": _as_float(indicators.get("bb_middle"), 0.0),
        "integrated_score": _as_float(integrated.get("integrated_score"), 50.0),
        # Sentiment features from sentiment_analysis if present
        "sentiment_score": _as_float(((state.get("sentiment_analysis") or {}).get("overall_sentiment") or {}).get("sentiment_score"), 0.0),
    }

    # Simple price momentum proxy when available
    try:
        last_close = _as_float(price_data.get("current_price"), 0.0)
        prev_close = _as_float(price_data.get("previous_close"), 0.0)
        f["daily_return_pct"] = (last_close - prev_close) / prev_close * 100.0 if prev_close else 0.0
        # Distances to MAs (percent)
        for sma_key in ("sma_20", "sma_50", "sma_200"):
            sma_val = f.get(sma_key)
            if sma_val and sma_val != 0:
                f[f"dist_{sma_key}"] = (last_close - sma_val) / sma_val * 100.0
            else:
                f[f"dist_{sma_key}"] = 0.0
        # Bollinger %b
        bb_u, bb_l = f.get("bb_upper", 0.0), f.get("bb_lower", 0.0)
        if bb_u and bb_l and bb_u != bb_l:
            f["bb_percent_b"] = (last_close - bb_l) / (bb_u - bb_l)
        else:
            f["bb_percent_b"] = 0.5
    except Exception:
        f["daily_return_pct"] = f.get("daily_return_pct", 0.0) if isinstance(f.get("daily_return_pct"), (int, float)) else 0.0
        for sma_key in ("sma_20", "sma_50", "sma_200"):
            f[f"dist_{sma_key}"] = f.get(f"dist_{sma_key}", 0.0)
        f["bb_percent_b"] = f.get("bb_percent_b", 0.5)

    X = np.array(list(f.values()), dtype=float)
    return X, f


