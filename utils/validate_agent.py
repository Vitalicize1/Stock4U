from __future__ import annotations

"""
Agent Validation CLI and helpers.

Validates the structure and value ranges of `prediction_result`,
`confidence_metrics`, and `recommendation`. Optionally runs a quick
prediction per ticker (offline by default) and writes a report to
`cache/metrics/validation/`.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime


ALLOWED_DIRECTIONS = {"UP", "DOWN", "NEUTRAL", "BUY", "SELL", "STRONG_BUY", "STRONG_SELL"}
ALLOWED_ACTIONS = {"BUY", "SELL", "BUY_WEAK", "SELL_WEAK", "HOLD"}
ALLOWED_POSITION_SIZES = {"small", "normal", "large"}


def _num(v: Any, default: float | None = None) -> float | None:
    try:
        return float(v)
    except Exception:
        return default


def validate_prediction_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    """Validate shapes and ranges of a single prediction payload.

    Accepts either a flat dict with keys like direction/confidence or a
    nested shape where the prediction is under `prediction_result` or
    `prediction_result.prediction`.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # Normalize to accessors
    root = result or {}
    pred_block = root.get("prediction_result") if isinstance(root, dict) else None
    if isinstance(pred_block, dict) and "prediction" in pred_block and "direction" not in pred_block:
        pred = pred_block.get("prediction", {})
    else:
        pred = pred_block if isinstance(pred_block, dict) else root

    cm = (root.get("confidence_metrics") or (pred_block or {}).get("confidence_metrics") or {}) if isinstance(root, dict) else {}
    rec = (root.get("recommendation") or (pred_block or {}).get("recommendation") or {}) if isinstance(root, dict) else {}

    # Direction
    direction = str(pred.get("direction", "")).upper()
    if not direction:
        errors.append("missing prediction.direction")
    elif direction not in ALLOWED_DIRECTIONS:
        warnings.append(f"unexpected direction '{direction}'")

    # Confidence
    conf = _num(pred.get("confidence"))
    overall_conf = _num(cm.get("overall_confidence"))
    conf_val = overall_conf if overall_conf is not None else conf
    if conf_val is None:
        warnings.append("missing confidence (overall_confidence or confidence)")
    else:
        if conf_val < 0 or conf_val > 100:
            errors.append(f"confidence out of range: {conf_val}")

    # Recommendation
    action = str(rec.get("action", "")).upper() if isinstance(rec, dict) else ""
    if not action:
        warnings.append("missing recommendation.action")
    elif action not in ALLOWED_ACTIONS:
        warnings.append(f"unexpected recommendation.action '{action}'")

    pos_size = str(rec.get("position_size", "normal")).lower() if isinstance(rec, dict) else "normal"
    if pos_size not in ALLOWED_POSITION_SIZES:
        warnings.append(f"unexpected position_size '{pos_size}'")

    # Optional sanity: if very high confidence and neutral direction/action
    if conf_val is not None and conf_val >= 85 and (direction in {"NEUTRAL"} or action == "HOLD"):
        warnings.append("very high confidence but neutral recommendation")

    # Recommendation consistency checks
    if action in {"BUY", "BUY_WEAK"}:
        if direction in {"DOWN", "SELL", "STRONG_SELL"}:
            errors.append("action BUY but direction indicates DOWN")
        # Confidence thresholds: BUY>60, BUY_WEAK>40
        if conf_val is not None:
            if action == "BUY" and conf_val < 60:
                warnings.append("BUY action with confidence < 60")
            if action == "BUY_WEAK" and conf_val < 40:
                warnings.append("BUY_WEAK action with confidence < 40")
    if action in {"SELL", "SELL_WEAK"}:
        if direction in {"UP", "BUY", "STRONG_BUY"}:
            errors.append("action SELL but direction indicates UP")
        if conf_val is not None:
            if action == "SELL" and conf_val < 60:
                warnings.append("SELL action with confidence < 60")
            if action == "SELL_WEAK" and conf_val < 40:
                warnings.append("SELL_WEAK action with confidence < 40")

    # Key/risk factors type checks
    key_factors = pred.get("key_factors", []) if isinstance(pred, dict) else []
    risk_factors = pred.get("risk_factors", []) if isinstance(pred, dict) else []
    if key_factors and not isinstance(key_factors, list):
        warnings.append("key_factors is not a list")
    else:
        for k in key_factors[:5]:  # sample a few
            if not isinstance(k, (str, int, float)):
                warnings.append("key_factors contains non-serializable item")
                break

    if risk_factors and not isinstance(risk_factors, list):
        warnings.append("risk_factors is not a list")
    else:
        for k in risk_factors[:5]:
            if not isinstance(k, (str, int, float)):
                warnings.append("risk_factors contains non-serializable item")
                break

    # Price range sanity
    pr = pred.get("price_range", {}) if isinstance(pred, dict) else {}
    low = _num(pr.get("low")) if isinstance(pr, dict) else None
    high = _num(pr.get("high")) if isinstance(pr, dict) else None
    if low is not None and high is not None and low > high:
        errors.append("price_range.low > price_range.high")

    # Confidence metrics (if present) should be 0..100
    if isinstance(cm, dict):
        oc = _num(cm.get("overall_confidence"))
        if oc is not None and (oc < 0 or oc > 100):
            errors.append("confidence_metrics.overall_confidence out of range")
        for k in ("technical_confidence", "sentiment_confidence", "llm_confidence"):
            v = _num(cm.get(k))
            if v is not None and (v < 0 or v > 100):
                warnings.append(f"confidence_metrics.{k} out of range")

    # Optional: technical indicators sanity if present
    ta = None
    if isinstance(result, dict):
        ta = result.get("technical_analysis") or (result.get("final_prediction", {}) or {}).get("technical_analysis")
    if isinstance(ta, dict):
        import math
        indicators = ta.get("indicators") if isinstance(ta.get("indicators"), dict) else {}
        for name, val in indicators.items():
            key = str(name).lower()
            # RSI 0..100
            if "rsi" in key:
                v = _num(val if not isinstance(val, dict) else val.get("value"))
                if v is not None and (v < 0 or v > 100):
                    warnings.append("indicator RSI out of range (0-100)")
            # SMA/EMA numeric sanity
            if "sma" in key or "ema" in key:
                v = _num(val if not isinstance(val, dict) else val.get("value"))
                if v is None:
                    warnings.append(f"indicator {name} not numeric")
            # MACD components numeric and finite
            if "macd" in key and isinstance(val, dict):
                for comp in ("macd", "signal", "histogram"):
                    v = _num(val.get(comp))
                    if v is None or not math.isfinite(float(v)):
                        warnings.append("indicator MACD has non-finite component")
                        break

    badge = "pass" if not errors and not warnings else ("warn" if not errors else "fail")
    return {
        "errors": errors,
        "warnings": warnings,
        "badge": badge,
        "summary": {
            "direction": direction or None,
            "confidence": conf_val,
            "action": action or None,
            "position_size": pos_size,
        },
    }


@dataclass
class ValidateConfig:
    tickers: List[str]
    offline: bool = True
    use_ml_model: bool = False
    outdir: str = str(Path("cache") / "metrics" / "validation")


def validate_live(cfg: ValidateConfig) -> Dict[str, Any]:
    """Run a single prediction for each ticker and validate the payloads."""
    try:
        from agents.prediction_agent import PredictionAgent
    except Exception as e:
        return {"status": "error", "error": f"import failed: {e}"}

    agent = PredictionAgent()
    out: Dict[str, Any] = {"status": "success", "tickers": cfg.tickers, "results": []}
    for t in cfg.tickers:
        # Minimal state so the agent can run in offline mode quickly
        state = {
            "ticker": t,
            "timeframe": "1d",
            "data": {"price_data": {"current_price": 0, "previous_close": 0}},
            "technical_analysis": {"technical_score": 50},
        }
        if cfg.offline:
            state["low_api_mode"] = True
        if cfg.use_ml_model:
            state["use_ml_model"] = True
        try:
            pred = agent.make_prediction(state)
        except Exception as e:
            out["results"].append({"ticker": t, "status": "error", "error": str(e)})
            continue
        report = validate_prediction_payload(pred)
        out["results"].append({"ticker": t, "status": "success", **report})

    # Write report
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(cfg.outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    fp = out_path / f"validation_{ts}.json"
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    # Aggregate pass/fail counts
    passes = sum(1 for r in out["results"] if r.get("badge") == "pass")
    warns = sum(1 for r in out["results"] if r.get("badge") == "warn")
    fails = sum(1 for r in out["results"] if r.get("badge") == "fail")
    out["summary_counts"] = {"pass": passes, "warn": warns, "fail": fails}

    out["path"] = str(fp)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Validate agent outputs")
    p.add_argument("--tickers", default="AAPL,MSFT,GOOGL", help="Comma-separated tickers to validate")
    p.add_argument("--online", action="store_true", help="Allow LLM calls (default offline)")
    p.add_argument("--ml", action="store_true", help="Use ML model when available")
    p.add_argument("--outdir", default=str(Path("cache") / "metrics" / "validation"))
    args = p.parse_args()

    cfg = ValidateConfig(
        tickers=[t.strip().upper() for t in args.tickers.split(",") if t.strip()],
        offline=(not args.online),
        use_ml_model=bool(args.ml),
        outdir=args.outdir,
    )
    res = validate_live(cfg)
    if res.get("status") == "success":
        print(f"Validation report written: {res.get('path')}")
    else:
        print(f"Validation failed: {res.get('error')}")


if __name__ == "__main__":
    main()


