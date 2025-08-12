"""
Portfolio sizing engine (MVP).

Transforms per-ticker prediction outputs into target portfolio weights and
trade intents under simple, configurable constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class PortfolioConfig:
    """Configuration for portfolio sizing and constraints.

    Attributes:
        max_weight_per_ticker: Max absolute target weight per name (e.g., 0.10 for 10%).
        max_gross_exposure: Max sum of absolute weights across names (e.g., 1.0 for 100%).
        max_cash_utilization_per_day: Max fraction of available cash that can be deployed in new buys.
        sizing_gain: Scales confidence into weight; 0.10 means at most ±10% if fully confident.
        allow_shorts: If False, negative weights are clamped to 0.
    """

    max_weight_per_ticker: float = 0.10
    max_gross_exposure: float = 1.00
    max_cash_utilization_per_day: float = 0.20
    sizing_gain: float = 0.10
    allow_shorts: bool = False


def _extract_direction_and_confidence(prediction_result: Dict, confidence_metrics: Optional[Dict]) -> tuple[str, float]:
    direction = str(prediction_result.get("direction", "NEUTRAL")).upper()
    # Prefer overall_confidence from confidence_metrics, else prediction_result.confidence, else 50
    if isinstance(confidence_metrics, dict):
        overall = confidence_metrics.get("overall_confidence")
        if overall is not None:
            try:
                return direction, float(overall)
            except Exception:
                pass
    conf = prediction_result.get("confidence", 50)
    try:
        return direction, float(conf)
    except Exception:
        return direction, 50.0


def compute_target_weights(
    per_ticker_inputs: Dict[str, Dict[str, Dict]],
    config: PortfolioConfig,
) -> Dict[str, float]:
    """Compute target weights for each ticker.

    per_ticker_inputs schema:
        {
          "AAPL": {"prediction_result": {...}, "confidence_metrics": {...}},
          ...
        }
    Returns:
        Dict of ticker -> target weight in [-max_weight, max_weight] (or [0, max] if long-only).
    """
    raw_weights: Dict[str, float] = {}
    for ticker, inputs in per_ticker_inputs.items():
        pred = inputs.get("prediction_result", {}) or {}
        confm = inputs.get("confidence_metrics", {}) or {}
        direction, confidence = _extract_direction_and_confidence(pred, confm)

        # Map confidence [0..100] and direction to signed weight suggestion
        # Center at 50 → 0; 100 → +1; 0 → -1, then scale by sizing_gain
        strength = max(-1.0, min(1.0, (confidence - 50.0) / 50.0))
        signed = strength
        if direction in ("DOWN", "SELL", "STRONG_SELL"):
            signed = -abs(strength)
        elif direction in ("UP", "BUY", "STRONG_BUY"):
            signed = abs(strength)
        else:
            signed = 0.0

        weight = config.sizing_gain * signed

        # Enforce per-name and long-only constraints
        if not config.allow_shorts:
            weight = max(0.0, weight)
        if abs(weight) > config.max_weight_per_ticker:
            weight = max(-config.max_weight_per_ticker, min(config.max_weight_per_ticker, weight))

        raw_weights[ticker] = float(weight)

    # Normalize to respect gross exposure cap
    gross = sum(abs(w) for w in raw_weights.values())
    if gross > config.max_gross_exposure and gross > 0:
        scale = config.max_gross_exposure / gross
        return {t: w * scale for t, w in raw_weights.items()}

    return raw_weights


def allowable_buy_notional(available_cash: float, config: PortfolioConfig) -> float:
    """Max notional for new long exposure today based on cash and config."""
    if available_cash <= 0:
        return 0.0
    return float(available_cash) * float(config.max_cash_utilization_per_day)


