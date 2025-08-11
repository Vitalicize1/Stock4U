"""
Prediction Agent Tools

Toolized helpers for the prediction stage:
- generate_llm_prediction_tool: delegate to best available LLM and return a prediction
- generate_rule_based_prediction_tool: fallback prediction when LLM unavailable
- calculate_confidence_metrics_tool: compute confidence metrics from analysis
- generate_recommendation_tool: derive a recommendation from prediction + confidence
"""

from typing import Dict, Any
from datetime import datetime
from langchain_core.tools import tool


@tool
def generate_llm_prediction_tool(analysis_summary: str) -> Dict[str, Any]:
    """
    Generate a prediction using the best available LLM via delegation.

    Args:
        analysis_summary: Comprehensive analysis summary string

    Returns:
        Dict with keys: prediction_result (dict)
    """
    try:
        # Use orchestrator's delegation tool to select provider and call prediction
        from agents.tools.orchestrator_tools import delegate_prediction_agent

        delegation_result = delegate_prediction_agent.invoke({
            "analysis_summary": analysis_summary,
            "available_llms": ["groq", "gemini"]
        })

        prediction_result = delegation_result.get("prediction_result")
        if not prediction_result:
            return {
                "status": "error",
                "error": "Delegation returned no prediction result",
                "prediction_result": None,
                "selected_agent": delegation_result.get("selected_agent")
            }

        return {
            "status": "success",
            "prediction_result": prediction_result,
            "selected_agent": delegation_result.get("selected_agent"),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"LLM prediction failed: {str(e)}",
            "prediction_result": None
        }


@tool
def generate_rule_based_prediction_tool(analysis_summary: str) -> Dict[str, Any]:
    """
    Rule-based fallback prediction when LLM is unavailable.

    Args:
        analysis_summary: Comprehensive analysis summary string (unused; kept for symmetry)

    Returns:
        Dict with rule-based prediction under key prediction_result
    """
    prediction = {
        "direction": "neutral",
        "confidence": 50.0,
        "price_target": None,
        "price_range": {
            "low": 0,
            "high": 0
        },
        "reasoning": "Rule-based analysis indicates mixed signals with balanced technical and sentiment factors.",
        "key_factors": [
            "Technical indicators show neutral momentum",
            "Sentiment analysis provides additional context",
            "Price is near support/resistance levels",
            "Market trend is neutral"
        ],
        "risk_factors": [
            "Market volatility could impact short-term movement",
            "Earnings announcements or news events could change outlook",
            "Sector-specific factors may influence performance",
            "Sentiment shifts could alter technical patterns"
        ],
        "sentiment_influence": "Sentiment integrated with technical indicators for a balanced assessment"
    }

    return {
        "status": "success",
        "prediction_result": prediction,
        "timestamp": datetime.now().isoformat()
    }


@tool
def calculate_confidence_metrics_tool(technical_analysis: Dict[str, Any],
                                      sentiment_integration: Dict[str, Any],
                                      prediction_result: Dict[str, Any],
                                      sentiment_analysis: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Calculate confidence using multiple signals so it doesn't default to 50.

    Heuristics used when fields are missing:
    - technical_confidence: prefer technical_score; else derive from signals/trend/ADX
    - sentiment_confidence: prefer integrated_score; else derive from overall_sentiment.sentiment_score
    - llm_confidence: use prediction_result.confidence if present; else bias by direction
    Weights: 0.4 technical, 0.3 sentiment, 0.3 llm
    """
    try:
        # Technical contribution
        technical_score = technical_analysis.get("technical_score")
        if technical_score is None:
            # Derive from trading signals or trends
            trading = technical_analysis.get("trading_signals", {})
            signal_strength = trading.get("signal_strength")
            if signal_strength is None:
                # derive from counts if available
                buy = len([s for s in trading.get("signals", []) if s.get("type") == "BUY"])
                sell = len([s for s in trading.get("signals", []) if s.get("type") == "SELL"])
                net = buy - sell
                signal_strength = max(-5, min(5, net))  # clamp
            trend_strength = technical_analysis.get("trend_analysis", {}).get("trend_strength") or \
                             technical_analysis.get("trend_analysis", {}).get("trend_strength", 0)
            adx = technical_analysis.get("trend_analysis", {}).get("adx_strength", 0)
            # Normalize to 0-100
            s_norm = 50 + (signal_strength or 0) * 10
            t_norm = float(trend_strength or 0)
            a_norm = float(adx or 0)
            technical_score = max(0, min(100, 0.5 * s_norm + 0.3 * t_norm + 0.2 * a_norm))

        # Sentiment contribution
        integrated_score = sentiment_integration.get("integrated_analysis", {}).get("integrated_score")
        if integrated_score is None:
            sa = sentiment_analysis or {}
            overall = sa.get("overall_sentiment", {})
            sscore = overall.get("sentiment_score")  # expected -1..1 or 0..1
            if sscore is None:
                # derive from labels
                label = overall.get("sentiment_label", "neutral")
                sscore = {"positive": 0.5, "neutral": 0.0, "negative": -0.5}.get(str(label).lower(), 0.0)
            # Map roughly from [-1,1] -> [0,100]
            if sscore is not None:
                try:
                    sfloat = float(sscore)
                    if -1.0 <= sfloat <= 1.0:
                        integrated_score = (sfloat + 1.0) * 50.0
                    else:
                        # assume 0..1
                        integrated_score = sfloat * 100.0
                except Exception:
                    integrated_score = 50.0
            else:
                integrated_score = 50.0

        # LLM contribution
        llm_confidence = prediction_result.get("confidence")
        if llm_confidence is None:
            # Bias by direction
            direction = str(prediction_result.get("direction", "NEUTRAL")).upper()
            llm_confidence = 45.0 if direction in ("UP", "DOWN") else 50.0

        # Blend
        weighted_confidence = (
            0.4 * float(technical_score) + 0.3 * float(integrated_score) + 0.3 * float(llm_confidence)
        )

        # Level
        if weighted_confidence >= 85:
            confidence_level = "very_high"
        elif weighted_confidence >= 65:
            confidence_level = "high"
        elif weighted_confidence >= 45:
            confidence_level = "medium"
        elif weighted_confidence >= 25:
            confidence_level = "low"
        else:
            confidence_level = "very_low"

        return {
            "status": "success",
            "confidence_metrics": {
                "overall_confidence": round(weighted_confidence, 1),
                "confidence_level": confidence_level,
                "technical_confidence": float(technical_score),
                "integrated_confidence": float(integrated_score),
                "llm_confidence": float(llm_confidence),
                "signal_strength": "strong" if weighted_confidence > 70 else "weak"
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Confidence metric calculation failed: {str(e)}",
            "confidence_metrics": {
                "overall_confidence": 50,
                "confidence_level": "medium",
                "technical_confidence": 50,
                "integrated_confidence": 50,
                "llm_confidence": 50,
                "signal_strength": "weak"
            }
        }


@tool
def generate_recommendation_tool(prediction_result: Dict[str, Any],
                                 confidence_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate recommendation from prediction + confidence metrics.
    """
    try:
        direction = prediction_result.get("direction", "neutral")
        confidence = confidence_metrics.get("overall_confidence", 50)

        recommendation = {
            "action": "HOLD",
            "position_size": "normal",
            "timeframe": "1_day",
            "stop_loss": None,
            "take_profit": None
        }

        if direction == "UP" and confidence > 60:
            recommendation["action"] = "BUY"
        elif direction == "DOWN" and confidence > 60:
            recommendation["action"] = "SELL"
        elif direction == "UP" and confidence > 40:
            recommendation["action"] = "BUY_WEAK"
        elif direction == "DOWN" and confidence > 40:
            recommendation["action"] = "SELL_WEAK"

        if confidence > 80:
            recommendation["position_size"] = "large"
        elif confidence < 40:
            recommendation["position_size"] = "small"

        return {"status": "success", "recommendation": recommendation}
    except Exception as e:
        return {
            "status": "error",
            "error": f"Recommendation generation failed: {str(e)}",
            "recommendation": {
                "action": "HOLD",
                "position_size": "normal",
                "timeframe": "1_day",
                "stop_loss": None,
                "take_profit": None
            }
        }


