# agents/tools/elicitation_tools.py
"""
Deterministic tools for the elicitation/finalization stage.
These functions assemble a consistent `final_summary` payload and a formatted
user-facing message without relying on an LLM.
"""

from typing import Dict, Any, List
from datetime import datetime
from langchain_core.tools import tool


def _normalize_prediction(prediction_result: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(prediction_result, dict):
        return {"direction": "NEUTRAL", "confidence": 50}
    if "direction" in prediction_result:
        return prediction_result
    return prediction_result.get("prediction", {}) or {"direction": "NEUTRAL", "confidence": 50}


def _normalize_technical(technical_analysis: Dict[str, Any]) -> Dict[str, Any]:
    technical_analysis = technical_analysis or {}
    if "indicators" in technical_analysis or "trading_signals" in technical_analysis:
        # already enhanced-like
        return technical_analysis
    # basic -> wrap
    trend = technical_analysis.get("trend_analysis", {})
    sr = technical_analysis.get("support_resistance", {})
    signals = technical_analysis.get("technical_signals", [])
    return {
        "technical_score": technical_analysis.get("technical_score", 0),
        "indicators": {},
        "trend_analysis": {
            "trends": {
                "short_term": trend.get("short_term_trend", "Unknown"),
                "medium_term": trend.get("medium_term_trend", "Unknown"),
                "long_term": trend.get("long_term_trend", "Unknown"),
            },
            "trend_strength": trend.get("trend_strength", 0.0),
        },
        "trading_signals": {
            "signals": signals,
            "overall_recommendation": (signals[0] if signals else "HOLD"),
        },
        "support_resistance": {
            "nearest_support": sr.get("support_level"),
            "nearest_resistance": sr.get("resistance_level"),
        },
        "patterns": [],
    }


@tool
def assemble_final_summary_tool(ticker: str, prediction_result: Dict[str, Any],
                                technical_analysis: Dict[str, Any],
                                evaluation_results: Dict[str, Any],
                                sentiment_integration: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Builds a standardized final_summary dict for the dashboard/chatbot."""
    try:
        prediction_norm = _normalize_prediction(prediction_result)
        technical_norm = _normalize_technical(technical_analysis)
        confidence_metrics = (prediction_result or {}).get("confidence_metrics", {})

        final_recommendation = {
            "action": (prediction_result or {}).get("recommendation", {}).get("action", "HOLD"),
            "position_size": (prediction_result or {}).get("recommendation", {}).get("position_size", "normal"),
            "timeframe": (prediction_result or {}).get("recommendation", {}).get("timeframe", "1_day"),
        }

        final_summary = {
            "elicitation_timestamp": datetime.now().isoformat(),
            "ticker": ticker,
            "prediction_summary": {
                "direction": prediction_norm.get("direction", "NEUTRAL"),
                "confidence": (confidence_metrics or {}).get("overall_confidence", prediction_norm.get("confidence", 50)),
                "price_target": prediction_norm.get("price_target"),
                "reasoning": prediction_norm.get("reasoning", "No reasoning provided"),
                "key_factors": prediction_norm.get("key_factors", []),
            },
            "technical_summary": technical_norm,
            "evaluation_summary": {
                "overall_score": (evaluation_results or {}).get("overall_score", 0),
                "prediction_quality": (evaluation_results or {}).get("prediction_quality", {}),
                "technical_consistency": (evaluation_results or {}).get("technical_consistency", {}),
                "risk_assessment": (evaluation_results or {}).get("risk_assessment", {}),
                "recommendation_strength": (evaluation_results or {}).get("recommendation_strength", {}),
            },
            "final_recommendation": final_recommendation,
        }

        if sentiment_integration:
            final_summary["evaluation_summary"]["sentiment_integration"] = sentiment_integration

        return {"status": "success", "final_summary": final_summary}
    except Exception as e:
        return {"status": "error", "error": f"assemble_final_summary_tool failed: {e}"}


@tool
def format_final_summary_text_tool(ticker: str, final_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Return a concise markdown string summarizing the result for UI/chat."""
    try:
        ps = final_summary.get("prediction_summary", {})
        ts = final_summary.get("technical_summary", {})
        es = final_summary.get("evaluation_summary", {})
        rec = final_summary.get("final_recommendation", {})
        direction = str(ps.get("direction", "NEUTRAL")).upper()
        conf = ps.get("confidence", 50)
        overall = ts.get("trading_signals", {}).get("overall_recommendation", "HOLD")
        score = es.get("overall_score", 0)
        action = rec.get("action", "HOLD")
        text = (
            f"📊 {ticker} — Direction: {direction}, Confidence: {conf:.1f}%\n"
            f"🔧 Technical: {overall}, Eval Score: {score:.1f}/100\n"
            f"💡 Recommendation: {action}"
        )
        return {"status": "success", "text": text}
    except Exception as e:
        return {"status": "error", "error": f"format_final_summary_text_tool failed: {e}"}


@tool
def elicit_confirmation_tool(ticker: str, prediction_result: Dict[str, Any],
                             technical_analysis: Dict[str, Any],
                             evaluation_results: Dict[str, Any],
                             sentiment_integration: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """One-call tool to build final_summary and a short formatted message."""
    try:
        fs = assemble_final_summary_tool.invoke({
            "ticker": ticker,
            "prediction_result": prediction_result,
            "technical_analysis": technical_analysis,
            "evaluation_results": evaluation_results,
            "sentiment_integration": sentiment_integration or {},
        })
        final_summary = (fs or {}).get("final_summary", {})
        text = format_final_summary_text_tool.invoke({
            "ticker": ticker,
            "final_summary": final_summary,
        }).get("text", "")
        return {"status": "success", "final_summary": final_summary, "message": text}
    except Exception as e:
        return {"status": "error", "error": f"elicit_confirmation_tool failed: {e}"}


