# agents/tools/evaluator_optimizer_tools.py
"""
Deterministic tools for evaluating prediction quality and generating optimization feedback.
These utilities avoid LLMs and return structured, UI-friendly payloads.
"""

from typing import Dict, Any, List
from datetime import datetime
from langchain_core.tools import tool


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    try:
        v = float(value)
    except Exception:
        v = 0.0
    return max(low, min(high, v))


@tool
def assess_prediction_quality_tool(prediction: Dict[str, Any], confidence_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Score prediction direction/confidence and reasoning completeness."""
    try:
        dir_ok = 1 if prediction.get("direction") in {"UP", "DOWN", "NEUTRAL"} else 0
        conf = float(confidence_metrics.get("overall_confidence", prediction.get("confidence", 50)))
        reasoning = prediction.get("reasoning") or ""
        reasoning_score = 100.0 if len(reasoning) >= 200 else 70.0 if len(reasoning) >= 80 else 40.0 if reasoning else 20.0
        score = _clamp(0.4 * conf + 0.2 * reasoning_score + 0.4 * (dir_ok * 100))
        return {
            "status": "success",
            "prediction_quality": {
                "score": score,
                "direction_present": bool(dir_ok),
                "reasoning_quality": "good" if reasoning_score >= 70 else ("fair" if reasoning_score >= 40 else "poor"),
                "confidence_used": conf,
            },
        }
    except Exception as e:
        return {"status": "error", "error": f"assess_prediction_quality_tool failed: {e}"}


@tool
def assess_technical_consistency_tool(technical_analysis: Dict[str, Any], prediction: Dict[str, Any]) -> Dict[str, Any]:
    """Check if prediction direction aligns with technical signals."""
    try:
        if "trading_signals" in (technical_analysis or {}):
            overall = (technical_analysis.get("trading_signals", {}) or {}).get("overall_recommendation", "HOLD")
        else:
            sigs = (technical_analysis or {}).get("technical_signals", [])
            overall = "HOLD" if not sigs else sigs[0]
        pred_dir = (prediction or {}).get("direction", "NEUTRAL")
        aligned = (pred_dir == "UP" and "BUY" in overall) or (pred_dir == "DOWN" and "SELL" in overall) or (pred_dir == "NEUTRAL" and "HOLD" in overall)
        consistency = 85.0 if aligned else 55.0 if "HOLD" in overall or pred_dir == "NEUTRAL" else 35.0
        return {
            "status": "success",
            "technical_consistency": {
                "aligned": aligned,
                "score": consistency,
                "technical_signal": overall,
                "prediction_direction": pred_dir,
            },
        }
    except Exception as e:
        return {"status": "error", "error": f"assess_technical_consistency_tool failed: {e}"}


@tool
def assess_risk_adequacy_tool(prediction_result: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate if risk assessment exists and warnings are appropriate."""
    try:
        risk = (prediction_result or {}).get("risk_assessment", {}) if isinstance(prediction_result, dict) else {}
        level = str(risk.get("overall_risk_level", "medium")).lower()
        parts = [risk.get("market_risk", "unknown"), risk.get("volatility_risk", "unknown"), risk.get("liquidity_risk", "unknown"), risk.get("sector_risk", "unknown"), risk.get("sentiment_risk", "unknown")]
        completeness = sum(1 for p in parts if p != "unknown") / max(1, len(parts))
        warnings = risk.get("risk_warnings", []) or []
        score = _clamp(60 + 20 * completeness + (10 if warnings else 0) - (10 if level == "high" else 0))
        return {
            "status": "success",
            "risk_assessment": {
                "score": score,
                "overall_risk_level": level,
                "warnings_present": bool(warnings),
                "completeness": completeness,
            },
        }
    except Exception as e:
        return {"status": "error", "error": f"assess_risk_adequacy_tool failed: {e}"}


@tool
def assess_recommendation_strength_tool(recommendation: Dict[str, Any], confidence_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Score recommendation strength given confidence and action."""
    try:
        action = (recommendation or {}).get("action", "HOLD")
        conf = float(confidence_metrics.get("overall_confidence", 50))
        base = 60 if action in {"BUY", "SELL"} else 50
        score = _clamp(base + (conf - 50) * 0.6)
        return {
            "status": "success",
            "recommendation_strength": {
                "score": score,
                "action": action,
                "confidence": conf,
            },
        }
    except Exception as e:
        return {"status": "error", "error": f"assess_recommendation_strength_tool failed: {e}"}


@tool
def calculate_evaluation_score_tool(prediction_quality: Dict[str, Any], technical_consistency: Dict[str, Any], risk_assessment: Dict[str, Any], recommendation_strength: Dict[str, Any]) -> Dict[str, Any]:
    """Combine sub-scores into an overall evaluation score."""
    try:
        pq = float((prediction_quality or {}).get("score", 50))
        tc = float((technical_consistency or {}).get("score", 50))
        ra = float((risk_assessment or {}).get("score", 50))
        rs = float((recommendation_strength or {}).get("score", 50))
        overall = _clamp(0.35 * pq + 0.25 * tc + 0.2 * ra + 0.2 * rs)
        return {"status": "success", "overall_score": overall}
    except Exception as e:
        return {"status": "error", "error": f"calculate_evaluation_score_tool failed: {e}"}


@tool
def generate_optimization_suggestions_tool(prediction_quality: Dict[str, Any], technical_consistency: Dict[str, Any], risk_assessment: Dict[str, Any], recommendation_strength: Dict[str, Any]) -> Dict[str, Any]:
    """Produce actionable suggestions based on weaknesses."""
    try:
        suggestions: List[str] = []
        if (prediction_quality or {}).get("score", 0) < 60:
            suggestions.append("Improve reasoning depth and ensure clear directional call.")
        if not (technical_consistency or {}).get("aligned", False):
            suggestions.append("Align prediction with dominant technical signal or justify divergence.")
        if (risk_assessment or {}).get("completeness", 0) < 0.8:
            suggestions.append("Enhance risk breakdown; ensure market/volatility/liquidity/sector/sentiment set.")
        if (recommendation_strength or {}).get("score", 0) < 60:
            suggestions.append("Adjust action to reflect confidence or reduce position sizing.")
        return {"status": "success", "optimization_suggestions": suggestions}
    except Exception as e:
        return {"status": "error", "error": f"generate_optimization_suggestions_tool failed: {e}"}


@tool
def evaluate_overall_tool(prediction_result: Dict[str, Any], technical_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    One-call evaluator: computes all sub-scores and summary suitable for UI.
    """
    try:
        # Normalize shape
        prediction = prediction_result if (isinstance(prediction_result, dict) and "direction" in prediction_result) else (prediction_result or {}).get("prediction", {})
        confidence_metrics = (prediction_result or {}).get("confidence_metrics", {}) if isinstance(prediction_result, dict) else {}
        recommendation = (prediction_result or {}).get("recommendation", {}) if isinstance(prediction_result, dict) else {}

        pq = assess_prediction_quality_tool.invoke({"prediction": prediction, "confidence_metrics": confidence_metrics}).get("prediction_quality", {})
        tc = assess_technical_consistency_tool.invoke({"technical_analysis": technical_analysis, "prediction": prediction}).get("technical_consistency", {})
        ra = assess_risk_adequacy_tool.invoke({"prediction_result": prediction_result}).get("risk_assessment", {})
        rs = assess_recommendation_strength_tool.invoke({"recommendation": recommendation, "confidence_metrics": confidence_metrics}).get("recommendation_strength", {})
        overall = calculate_evaluation_score_tool.invoke({
            "prediction_quality": pq, "technical_consistency": tc, "risk_assessment": ra, "recommendation_strength": rs
        }).get("overall_score", 0.0)
        suggestions = generate_optimization_suggestions_tool.invoke({
            "prediction_quality": pq, "technical_consistency": tc, "risk_assessment": ra, "recommendation_strength": rs
        }).get("optimization_suggestions", [])

        return {
            "status": "success",
            "evaluation_results": {
                "evaluation_timestamp": datetime.now().isoformat(),
                "prediction_quality": pq,
                "technical_consistency": tc,
                "risk_assessment": ra,
                "recommendation_strength": rs,
                "overall_score": overall,
                "optimization_suggestions": suggestions,
                "evaluator_version": "tools-1.0",
                "evaluation_criteria": [
                    "prediction_quality", "technical_consistency", "risk_assessment", "recommendation_strength"
                ],
            },
        }
    except Exception as e:
        return {"status": "error", "error": f"evaluate_overall_tool failed: {e}"}


