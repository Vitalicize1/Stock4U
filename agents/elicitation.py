# agents/elicitation.py
from typing import Dict, Any
from datetime import datetime
from agents.tools.elicitation_tools import elicit_confirmation_tool

def elicit_confirmation(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Elicitation agent that provides final confirmation and summary of prediction results.
    
    Args:
        state: Current state containing all prediction results and evaluation
        
    Returns:
        Final state with confirmation and summary
    """
    try:
        ticker = state.get("ticker", "UNKNOWN")
        prediction_result = state.get("prediction_result", {})
        evaluation_results = state.get("evaluation_results", {})
        
        # Handle both basic and enhanced technical analysis structures
        enhanced_technical_analysis = state.get("enhanced_technical_analysis", {})
        technical_analysis = state.get("technical_analysis", {})
        
        # Use enhanced analysis if available, otherwise fall back to basic
        if enhanced_technical_analysis:
            technical_analysis = enhanced_technical_analysis
            print(f"📊 Using enhanced technical analysis for final summary")
        elif technical_analysis:
            print(f"📊 Using basic technical analysis for final summary")
        else:
            print(f"⚠️ No technical analysis available for final summary")
            technical_analysis = {}
        
        # Create final summary (prefer deterministic tool)
        # Support both prediction structures: flat dict or wrapped under "prediction"
        normalized_prediction_result = prediction_result
        if isinstance(prediction_result, dict) and "prediction" in prediction_result and "direction" not in prediction_result:
            normalized_prediction_result = prediction_result

        try:
            tool_res = elicit_confirmation_tool.invoke({
                "ticker": state.get("ticker", "UNKNOWN"),
                "prediction_result": normalized_prediction_result,
                "technical_analysis": technical_analysis,
                "evaluation_results": evaluation_results,
                "sentiment_integration": state.get("sentiment_integration", {}),
            })
        except Exception:
            tool_res = None

        if isinstance(tool_res, dict) and tool_res.get("status") == "success":
            final_summary = tool_res.get("final_summary", {})
            final_summary["confidence_level"] = _determine_confidence_level(normalized_prediction_result, evaluation_results)
            final_summary["risk_warnings"] = _generate_risk_warnings(normalized_prediction_result)
            final_summary["next_steps"] = _suggest_next_steps(normalized_prediction_result, evaluation_results)
        else:
            final_summary = {
                "elicitation_timestamp": datetime.now().isoformat(),
                "prediction_summary": _create_prediction_summary(normalized_prediction_result),
                "technical_summary": _create_technical_summary(technical_analysis, state.get("current_price")),
                "evaluation_summary": _create_evaluation_summary(evaluation_results),
                "final_recommendation": _create_final_recommendation(normalized_prediction_result, evaluation_results),
                "confidence_level": _determine_confidence_level(normalized_prediction_result, evaluation_results),
                "risk_warnings": _generate_risk_warnings(normalized_prediction_result),
                "next_steps": _suggest_next_steps(normalized_prediction_result, evaluation_results)
            }
        
        # Add workflow completion metadata
        final_summary.update({
            "workflow_completed": True,
            "total_processing_time": _calculate_processing_time(state),
            "pipeline_stages_completed": [
                "orchestrator",
                "data_collector", 
                "technical_analyzer",
                "sentiment_analyzer",
                "sentiment_integrator",
                "prediction_agent",
                "evaluator_optimizer",
                "elicitation"
            ],
            "final_status": "completed"
        })
        
        print(f"✅ Prediction analysis completed for {ticker}")
        print(f"🎯 Final Recommendation: {final_summary['final_recommendation']['action']}")
        print(f"📊 Confidence: {final_summary['confidence_level']}")
        
        return {
            "status": "success",
            "final_summary": final_summary,
            "workflow_status": "completed"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Elicitation failed: {str(e)}",
            "workflow_status": "failed"
        }

def _create_prediction_summary(prediction_result: Dict[str, Any]) -> Dict[str, Any]:
    """Create a summary of the prediction results."""
    
    # If prediction_result is already the prediction dict (has direction), use it directly
    prediction = prediction_result if "direction" in prediction_result else prediction_result.get("prediction", {})
    confidence_metrics = prediction_result.get("confidence_metrics", {}) if isinstance(prediction_result, dict) else {}
    
    return {
        "direction": prediction.get("direction", "neutral"),
        "confidence": confidence_metrics.get("overall_confidence", 50),
        "price_target": prediction.get("price_target"),
        "reasoning": prediction.get("reasoning", "No reasoning provided"),
        "key_factors": prediction.get("key_factors", []),
        "risk_factors": prediction.get("risk_factors", [])
    }

def _create_technical_summary(technical_analysis: Dict[str, Any], current_price: float | None = None) -> Dict[str, Any]:
    """Create a summary of technical analysis."""
    
    # Handle enhanced technical analysis structure
    if "indicators" in technical_analysis:
        # Enhanced structure
        indicators = technical_analysis.get("indicators", {})
        trading_signals = technical_analysis.get("trading_signals", {})
        trend_analysis = technical_analysis.get("trend_analysis", {})
        support_resistance = technical_analysis.get("support_resistance", {})
        patterns = technical_analysis.get("patterns", [])
        
        return {
            "technical_score": technical_analysis.get("technical_score", 0),
            "indicators": indicators,
            "trend_analysis": trend_analysis,
            "technical_signals": trading_signals.get("signals", []),
            "trading_signals": trading_signals,
            "support_resistance": support_resistance,
            "patterns": patterns,
            "momentum_analysis": {
                "rsi": indicators.get("rsi", 50),
                "macd": indicators.get("macd", 0),
                "stochastic": indicators.get("stoch_k", 50)
            }
        }
    else:
        # Basic structure -> normalize to enhanced-like shape expected by UI
        basic_trend = technical_analysis.get("trend_analysis", {})
        basic_sr = technical_analysis.get("support_resistance", {})
        signals_list = technical_analysis.get("technical_signals", [])

        # Map trend fields to enhanced structure
        trend_analysis = {
            "trends": {
                "short_term": basic_trend.get("short_term_trend", "Unknown"),
                "medium_term": basic_trend.get("medium_term_trend", "Unknown"),
                "long_term": basic_trend.get("long_term_trend", "Unknown"),
            },
            "trend_strength": basic_trend.get("trend_strength", 0.0),
        }

        # Map support/resistance fields
        support_resistance = {
            "current_price": current_price,
            "nearest_support": basic_sr.get("support_level"),
            "nearest_resistance": basic_sr.get("resistance_level"),
        }

        # Build trading signals block from list
        overall = "HOLD"
        if signals_list:
            # Prefer first element or a stronger one if present
            if any(s == "STRONG_BUY" for s in signals_list):
                overall = "STRONG_BUY"
            elif any(s == "BUY" for s in signals_list):
                overall = "BUY"
            elif any(s == "STRONG_SELL" for s in signals_list):
                overall = "STRONG_SELL"
            elif any(s == "SELL" for s in signals_list):
                overall = "SELL"

        trading_signals = {
            "signals": signals_list,
            "overall_recommendation": overall,
        }

        return {
            "technical_score": technical_analysis.get("technical_score", 0),
            "indicators": {},
            "trend_analysis": trend_analysis,
            "technical_signals": signals_list,
            "trading_signals": trading_signals,
            "support_resistance": support_resistance,
            "momentum_analysis": technical_analysis.get("momentum_analysis", {}),
        }

def _create_evaluation_summary(evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
    """Create a summary of evaluation results."""
    
    return {
        "overall_score": evaluation_results.get("overall_score", 0),
        "prediction_quality": evaluation_results.get("prediction_quality", {}),
        "technical_consistency": evaluation_results.get("technical_consistency", {}),
        "risk_assessment": evaluation_results.get("risk_assessment", {}),
        "recommendation_strength": evaluation_results.get("recommendation_strength", {}),
        "optimization_suggestions": evaluation_results.get("optimization_suggestions", [])
    }

def _create_final_recommendation(prediction_result: Dict[str, Any], 
                               evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
    """Create the final trading recommendation."""
    
    recommendation = prediction_result.get("recommendation", {})
    evaluation_score = evaluation_results.get("overall_score", 50)
    
    # Adjust recommendation based on evaluation score
    final_recommendation = {
        "action": recommendation.get("action", "HOLD"),
        "position_size": recommendation.get("position_size", "normal"),
        "timeframe": recommendation.get("timeframe", "1_day"),
        "confidence": prediction_result.get("confidence_metrics", {}).get("overall_confidence", 50),
        "evaluation_score": evaluation_score,
        "recommendation_strength": "strong" if evaluation_score > 70 else "moderate" if evaluation_score > 50 else "weak"
    }
    
    # Adjust action based on evaluation score
    if evaluation_score < 40:
        final_recommendation["action"] = "HOLD"
        final_recommendation["recommendation_strength"] = "weak"
    
    return final_recommendation

def _determine_confidence_level(prediction_result: Dict[str, Any], 
                              evaluation_results: Dict[str, Any]) -> str:
    """Determine the overall confidence level."""
    
    prediction_confidence = prediction_result.get("confidence_metrics", {}).get("overall_confidence", 50)
    evaluation_score = evaluation_results.get("overall_score", 50)
    
    # Weighted confidence calculation
    weighted_confidence = (prediction_confidence * 0.7) + (evaluation_score * 0.3)
    
    if weighted_confidence > 80:
        return "very_high"
    elif weighted_confidence > 60:
        return "high"
    elif weighted_confidence > 40:
        return "medium"
    elif weighted_confidence > 20:
        return "low"
    else:
        return "very_low"

def _generate_risk_warnings(prediction_result: Dict[str, Any]) -> list:
    """Generate risk warnings based on prediction results."""
    
    warnings = []
    risk_assessment = prediction_result.get("risk_assessment", {})
    prediction = prediction_result.get("prediction", {})
    
    # Add general risk warning
    warnings.append("This prediction is for informational purposes only and should not be considered as financial advice.")
    
    # Add specific risk warnings
    overall_risk = risk_assessment.get("overall_risk_level", "medium")
    if overall_risk == "high":
        warnings.append("High risk level detected - consider reducing position size or avoiding trade.")
    
    market_risk = risk_assessment.get("market_risk", "medium")
    if market_risk == "high":
        warnings.append("High market risk - consider broader market conditions before trading.")
    
    volatility_risk = risk_assessment.get("volatility_risk", "medium")
    if volatility_risk == "high":
        warnings.append("High volatility expected - consider using stop-loss orders.")
    
    # Add prediction-specific warnings
    confidence = prediction_result.get("confidence_metrics", {}).get("overall_confidence", 50)
    if confidence < 40:
        warnings.append("Low confidence prediction - consider waiting for stronger signals.")
    
    return warnings

def _suggest_next_steps(prediction_result: Dict[str, Any], 
                       evaluation_results: Dict[str, Any]) -> list:
    """Suggest next steps based on prediction and evaluation."""
    
    next_steps = []
    evaluation_score = evaluation_results.get("overall_score", 50)
    recommendation = prediction_result.get("recommendation", {})
    
    # Add immediate action steps
    action = recommendation.get("action", "HOLD")
    if action in ["BUY", "BUY_WEAK"]:
        next_steps.append("Monitor price action for entry confirmation")
        next_steps.append("Set appropriate stop-loss levels")
        next_steps.append("Consider position sizing based on risk tolerance")
    elif action in ["SELL", "SELL_WEAK"]:
        next_steps.append("Monitor for exit confirmation signals")
        next_steps.append("Consider hedging strategies if holding long positions")
    else:
        next_steps.append("Continue monitoring for new signals")
        next_steps.append("Review technical indicators for trend changes")
    
    # Add evaluation-based suggestions
    if evaluation_score < 60:
        next_steps.append("Consider waiting for stronger technical confirmation")
        next_steps.append("Review additional data sources for validation")
    
    # Add general monitoring steps
    next_steps.append("Monitor market news and earnings announcements")
    next_steps.append("Track sector performance for broader context")
    next_steps.append("Review prediction accuracy after market close")
    
    return next_steps

def _calculate_processing_time(state: Dict[str, Any]) -> str:
    """Calculate total processing time for the workflow."""
    
    start_time = state.get("workflow_start_time")
    if not start_time:
        return "unknown"
    
    try:
        from datetime import datetime
        start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        end_dt = datetime.now()
        duration = end_dt - start_dt
        return f"{duration.total_seconds():.2f} seconds"
    except:
        return "unknown"
