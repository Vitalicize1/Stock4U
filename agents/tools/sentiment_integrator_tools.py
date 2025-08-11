# agents/tools/sentiment_integrator_tools.py
"""
Deterministic tools for integrating sentiment with technical analysis.
These are non‑LLM utilities that the sentiment integrator agent can call
to compute an integrated score, adjust signals, and generate insights.
"""

from typing import Dict, Any, List
from datetime import datetime
from langchain_core.tools import tool


def _normalize_sentiment_score(score: float) -> float:
    try:
        s = float(score)
    except Exception:
        s = 0.0
    return max(-1.0, min(1.0, s))


def _get_technical_score(technical_analysis: Dict[str, Any]) -> float:
    if not isinstance(technical_analysis, dict):
        return 50.0
    # Enhanced structure with trading_signals
    if "trading_signals" in technical_analysis:
        signals = technical_analysis.get("trading_signals", {}) or {}
        signal_strength = float(signals.get("signal_strength", 0))
        # map roughly to 0-100
        return max(0.0, min(100.0, 50.0 + signal_strength * 10.0))
    # Basic structure with explicit score
    return float(technical_analysis.get("technical_score", 50.0))


def _label_from_score(score: float) -> str:
    if score > 0.3:
        return "very_positive"
    if score > 0.1:
        return "positive"
    if score < -0.3:
        return "very_negative"
    if score < -0.1:
        return "negative"
    return "neutral"

def _market_context_score(market_data: Dict[str, Any]) -> float:
    if not isinstance(market_data, dict):
        return 50.0
    market_trend = (market_data.get("market_trend") or market_data.get("market_sentiment", {}) or {}).get(
        "overall_trend", market_data.get("market_trend", "neutral")
    )
    if market_trend == "bullish":
        return 70.0
    if market_trend == "bearish":
        return 30.0
    return 50.0


@tool
def integrate_scores(technical_analysis: Dict[str, Any], sentiment_analysis: Dict[str, Any],
                     market_data: Dict[str, Any] | None = None,
                     technical_weight: float = 0.6, sentiment_weight: float = 0.3,
                     market_weight: float = 0.1) -> Dict[str, Any]:
    """
    Compute an integrated score combining technical (0-100), sentiment (-1..1 mapped to 0-100),
    and market context (0-100). Returns breakdown suitable for UI.
    """
    try:
        t_score = _get_technical_score(technical_analysis)
        s_score = _normalize_sentiment_score((sentiment_analysis.get("overall_sentiment", {}) or {}).get("sentiment_score", 0.0))
        s_conf = float((sentiment_analysis.get("overall_sentiment", {}) or {}).get("confidence", 0.0) or 0.0)
        s_abs = abs(s_score)
        s_norm = (s_score + 1.0) * 50.0
        m_score = _market_context_score(market_data or {})

        # Dynamic weighting: boost sentiment weight when confidence and absolute sentiment are high
        # Base: tech 0.6, sent 0.3, market 0.1 → Shift up to +0.2 into sentiment when strong & confident
        dynamic_sent_boost = min(0.2, 0.2 * s_abs * (0.5 + 0.5 * min(1.0, s_conf)))
        dyn_sent_w = max(0.25, sentiment_weight + dynamic_sent_boost)
        # Pull proportionally from technical weight
        dyn_tech_w = max(0.4, technical_weight - dynamic_sent_boost)
        dyn_market_w = market_weight  # keep market context stable
        # Normalize to sum 1.0
        w_sum = dyn_tech_w + dyn_sent_w + dyn_market_w
        dyn_tech_w, dyn_sent_w, dyn_market_w = [w / w_sum for w in (dyn_tech_w, dyn_sent_w, dyn_market_w)]

        integrated = (t_score * dyn_tech_w) + (s_norm * dyn_sent_w) + (m_score * dyn_market_w)
        return {
            "status": "success",
            "integrated_analysis": {
                "integrated_score": float(integrated),
                "technical_contribution": float(t_score * dyn_tech_w),
                "sentiment_contribution": float(s_norm * dyn_sent_w),
                "market_contribution": float(m_score * dyn_market_w),
                "integration_breakdown": {
                    "technical_weight": dyn_tech_w,
                    "sentiment_weight": dyn_sent_w,
                    "market_weight": dyn_market_w,
                },
            },
        }
    except Exception as e:
        return {"status": "error", "error": f"integrate_scores failed: {e}"}


@tool
def adjust_signals_with_sentiment(technical_analysis: Dict[str, Any], sentiment_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adjust technical signals based on sentiment score; returns adjusted signals and recommendation.
    """
    try:
        # Extract technical signals in a structure-agnostic way
        if "trading_signals" in (technical_analysis or {}):
            ts = (technical_analysis or {}).get("trading_signals", {}) or {}
            original_signals = ts.get("signals", [])
            overall_recommendation = ts.get("overall_recommendation", "HOLD")
        else:
            original_signals = (technical_analysis or {}).get("technical_signals", [])
            overall_recommendation = "HOLD" if not original_signals else original_signals[0]

        s_score = _normalize_sentiment_score((sentiment_analysis.get("overall_sentiment", {}) or {}).get("sentiment_score", 0.0))
        adjusted_signals: List[Any] = []
        sentiment_adjustment = 0

        if s_score > 0.3:
            sentiment_adjustment = 1
            for s in original_signals:
                if isinstance(s, dict) and s.get("type") == "BUY":
                    s = {**s, "strength": "strong", "sentiment_boost": True}
                adjusted_signals.append(s)
        elif s_score < -0.3:
            sentiment_adjustment = -1
            for s in original_signals:
                if isinstance(s, dict) and s.get("type") == "SELL":
                    s = {**s, "strength": "strong", "sentiment_boost": True}
                adjusted_signals.append(s)
        else:
            adjusted_signals = original_signals

        if sentiment_adjustment > 0 and "BUY" in overall_recommendation:
            adjusted_recommendation = "STRONG_BUY" if overall_recommendation == "BUY" else overall_recommendation
        elif sentiment_adjustment < 0 and "SELL" in overall_recommendation:
            adjusted_recommendation = "STRONG_SELL" if overall_recommendation == "SELL" else overall_recommendation
        else:
            adjusted_recommendation = overall_recommendation

        return {
            "status": "success",
            "adjusted_technical_signals": {
                "original_signals": original_signals,
                "adjusted_signals": adjusted_signals,
                "original_recommendation": overall_recommendation,
                "adjusted_recommendation": adjusted_recommendation,
                "sentiment_adjustment": sentiment_adjustment,
            },
        }
    except Exception as e:
        return {"status": "error", "error": f"adjust_signals_with_sentiment failed: {e}"}


@tool
def calculate_alignment_and_confidence(technical_analysis: Dict[str, Any], sentiment_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate alignment between technical and sentiment and produce adjusted confidence.
    """
    try:
        # Technical confidence proxy
        if "trading_signals" in (technical_analysis or {}):
            technical_confidence = float((technical_analysis.get("trading_signals", {}) or {}).get("signal_strength", 0))
        else:
            technical_confidence = float((technical_analysis or {}).get("technical_score", 50))

        s_score = _normalize_sentiment_score((sentiment_analysis.get("overall_sentiment", {}) or {}).get("sentiment_score", 0.0))
        sentiment_confidence = float((sentiment_analysis.get("overall_sentiment", {}) or {}).get("confidence", 0.5))

        sentiment_direction = "BUY" if s_score > 0.1 else "SELL" if s_score < -0.1 else "HOLD"
        # Determine technical direction
        if "trading_signals" in (technical_analysis or {}):
            overall_recommendation = (technical_analysis.get("trading_signals", {}) or {}).get("overall_recommendation", "HOLD")
        else:
            ts = (technical_analysis or {}).get("technical_signals", [])
            overall_recommendation = "HOLD" if not ts else ts[0]

        if "BUY" in overall_recommendation and sentiment_direction == "BUY":
            alignment = "aligned"
        elif "SELL" in overall_recommendation and sentiment_direction == "SELL":
            alignment = "aligned"
        elif "HOLD" in overall_recommendation and sentiment_direction == "HOLD":
            alignment = "neutral"
        else:
            alignment = "conflicting"

        if alignment == "aligned":
            adjusted_confidence = min(100.0, technical_confidence * 1.2)
        elif alignment == "conflicting":
            adjusted_confidence = technical_confidence * 0.8
        else:
            adjusted_confidence = technical_confidence

        return {
            "status": "success",
            "sentiment_adjusted_confidence": {
                "technical_confidence": technical_confidence,
                "sentiment_confidence": sentiment_confidence,
                "adjusted_confidence": adjusted_confidence,
                "alignment": alignment,
                "confidence_boost": adjusted_confidence - technical_confidence,
            },
        }
    except Exception as e:
        return {"status": "error", "error": f"calculate_alignment_and_confidence failed: {e}"}


@tool
def generate_sentiment_insights_tool(sentiment_analysis: Dict[str, Any], technical_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Generate qualitative insights and recommendations from sentiment context."""
    try:
        raw = (sentiment_analysis.get("overall_sentiment", {}) or {}).get("sentiment_score", 0.0)
        s = _normalize_sentiment_score(raw)
        label = _label_from_score(s)
        if s > 0.5:
            impact = "strong_positive"
        elif s > 0.1:
            impact = "moderate_positive"
        elif s < -0.5:
            impact = "strong_negative"
        elif s < -0.1:
            impact = "moderate_negative"
        else:
            impact = "neutral"

        recommendations: List[str] = []
        if s > 0.3:
            recommendations = [
                "Consider increasing position size due to positive sentiment",
                "Monitor for potential momentum continuation",
                "Watch for sentiment-driven breakouts",
            ]
        elif s < -0.3:
            recommendations = [
                "Consider reducing position size due to negative sentiment",
                "Monitor for potential sentiment-driven reversals",
                "Watch for support levels as sentiment improves",
            ]
        else:
            recommendations = [
                "Sentiment is neutral - focus on technical analysis",
                "Monitor for sentiment shifts",
                "Maintain current position sizing",
            ]

        return {
            "status": "success",
            "sentiment_insights": {
                "sentiment_score": float(s),
                "sentiment_label": label,
                "impact_assessment": impact,
                "recommendations": recommendations,
                "key_insights": [
                    f"Market sentiment is { (sentiment_analysis.get('overall_sentiment', {}) or {}).get('sentiment_label', 'neutral') }",
                    f"Sentiment impact: {impact}",
                    f"Recommendations: {len(recommendations)} actionable items",
                ],
            },
        }
    except Exception as e:
        return {"status": "error", "error": f"generate_sentiment_insights_tool failed: {e}"}


@tool
def integrate_sentiment_tool(technical_analysis: Dict[str, Any], sentiment_analysis: Dict[str, Any],
                             market_data: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Orchestrates score integration, signal adjustment, confidence, and insights
    into a single payload under key 'sentiment_integration'.
    """
    try:
        integrated = integrate_scores.invoke({
            "technical_analysis": technical_analysis,
            "sentiment_analysis": sentiment_analysis,
            "market_data": market_data or {},
        })
        adjusted = adjust_signals_with_sentiment.invoke({
            "technical_analysis": technical_analysis,
            "sentiment_analysis": sentiment_analysis,
        })
        conf = calculate_alignment_and_confidence.invoke({
            "technical_analysis": technical_analysis,
            "sentiment_analysis": sentiment_analysis,
        })
        insights = generate_sentiment_insights_tool.invoke({
            "sentiment_analysis": sentiment_analysis,
            "technical_analysis": technical_analysis,
        })

        result = {
            "integrated_analysis": (integrated.get("integrated_analysis") if isinstance(integrated, dict) else {}) or {},
            "adjusted_technical_signals": (adjusted.get("adjusted_technical_signals") if isinstance(adjusted, dict) else {}) or {},
            "sentiment_adjusted_confidence": (conf.get("sentiment_adjusted_confidence") if isinstance(conf, dict) else {}) or {},
            "sentiment_insights": (insights.get("sentiment_insights") if isinstance(insights, dict) else {}) or {},
        }

        return {"status": "success", "sentiment_integration": result, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"status": "error", "error": f"integrate_sentiment_tool failed: {e}"}


