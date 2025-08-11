# agents/evaluator_optimizer.py
from typing import Dict, Any
from datetime import datetime
from agents.tools.evaluator_optimizer_tools import evaluate_overall_tool

def evaluate(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluator agent that assesses prediction quality and provides optimization feedback.
    
    Args:
        state: Current state containing prediction results and analysis
        
    Returns:
        Updated state with evaluation results
    """
    try:
        prediction_result = state.get("prediction_result", {})
        technical_analysis = state.get("technical_analysis", {})
        
        if not prediction_result:
            return {
                "status": "error",
                "error": "No prediction result to evaluate",
                "next_agent": "error_handler"
            }
        
        # Extract key metrics for evaluation
        # Support both structures: either prediction_result is the prediction dict,
        # or it's a wrapper with key "prediction"
        prediction = prediction_result if isinstance(prediction_result, dict) and "direction" in prediction_result else prediction_result.get("prediction", {})
        confidence_metrics = prediction_result.get("confidence_metrics", {}) if isinstance(prediction_result, dict) else {}
        recommendation = prediction_result.get("recommendation", {}) if isinstance(prediction_result, dict) else {}
        
        # Prefer deterministic tool-based evaluation for stability
        try:
            tool_res = evaluate_overall_tool.invoke({
                "prediction_result": prediction_result,
                "technical_analysis": technical_analysis,
            })
        except Exception:
            tool_res = None

        if isinstance(tool_res, dict) and tool_res.get("status") == "success":
            evaluation_results = tool_res.get("evaluation_results", {})
        else:
            # Perform evaluation with local helpers (fallback)
            evaluation_results = {
                "evaluation_timestamp": datetime.now().isoformat(),
                "prediction_quality": _assess_prediction_quality(prediction, confidence_metrics),
                "technical_consistency": _assess_technical_consistency(technical_analysis, prediction),
                "risk_assessment": _assess_risk_adequacy(prediction_result),
                "recommendation_strength": _assess_recommendation_strength(recommendation, confidence_metrics),
                "overall_score": 0.0,
                "optimization_suggestions": []
            }
            evaluation_results["overall_score"] = _calculate_evaluation_score(evaluation_results)
            evaluation_results["optimization_suggestions"] = _generate_optimization_suggestions(evaluation_results)
        
        # Add evaluation metadata
        evaluation_results.update({
            "evaluator_version": "1.0",
            "evaluation_criteria": [
                "prediction_quality",
                "technical_consistency", 
                "risk_assessment",
                "recommendation_strength"
            ]
        })
        
        return {
            "status": "success",
            "evaluation_results": evaluation_results,
            "next_agent": "elicitation"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Evaluation failed: {str(e)}",
            "next_agent": "error_handler"
        }

def _assess_prediction_quality(prediction: Dict[str, Any], confidence_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Assess the quality of the prediction."""
    
    direction = prediction.get("direction", "neutral")
    confidence = confidence_metrics.get("overall_confidence", 50)
    reasoning = prediction.get("reasoning", "")
    
    quality_score = 50.0  # Base score
    
    # Assess confidence level
    if confidence > 80:
        quality_score += 20
    elif confidence > 60:
        quality_score += 10
    elif confidence < 30:
        quality_score -= 20
    
    # Assess reasoning quality
    if len(reasoning) > 100:
        quality_score += 10
    elif len(reasoning) < 50:
        quality_score -= 10
    
    # Assess direction clarity
    if direction in ["UP", "DOWN"]:
        quality_score += 10
    elif direction == "neutral":
        quality_score += 5
    
    return {
        "score": max(0, min(100, quality_score)),
        "confidence_adequacy": "high" if confidence > 70 else "medium" if confidence > 50 else "low",
        "reasoning_quality": "good" if len(reasoning) > 100 else "adequate" if len(reasoning) > 50 else "poor",
        "direction_clarity": "clear" if direction in ["UP", "DOWN"] else "neutral"
    }

def _assess_technical_consistency(technical_analysis: Dict[str, Any], prediction: Dict[str, Any]) -> Dict[str, Any]:
    """Assess consistency between technical analysis and prediction."""
    
    technical_score = technical_analysis.get("technical_score", 50)
    prediction_direction = prediction.get("direction", "neutral")
    technical_signals = technical_analysis.get("technical_signals", [])
    
    consistency_score = 50.0
    
    # Check if technical signals align with prediction
    if "BUY" in technical_signals and prediction_direction == "UP":
        consistency_score += 20
    elif "SELL" in technical_signals and prediction_direction == "DOWN":
        consistency_score += 20
    elif "HOLD" in technical_signals and prediction_direction == "neutral":
        consistency_score += 15
    
    # Check technical score alignment
    if technical_score > 70 and prediction_direction == "UP":
        consistency_score += 10
    elif technical_score < 30 and prediction_direction == "DOWN":
        consistency_score += 10
    elif 40 <= technical_score <= 60 and prediction_direction == "neutral":
        consistency_score += 10
    
    return {
        "score": max(0, min(100, consistency_score)),
        "signal_alignment": "strong" if consistency_score > 70 else "moderate" if consistency_score > 50 else "weak",
        "technical_prediction_agreement": "high" if consistency_score > 70 else "medium" if consistency_score > 50 else "low"
    }

def _assess_risk_adequacy(prediction_result: Dict[str, Any]) -> Dict[str, Any]:
    """Assess the adequacy of risk assessment."""
    
    risk_assessment = prediction_result.get("risk_assessment", {})
    risk_factors = prediction_result.get("prediction", {}).get("risk_factors", [])
    
    risk_score = 50.0
    
    # Assess risk factors coverage
    if len(risk_factors) >= 3:
        risk_score += 20
    elif len(risk_factors) >= 2:
        risk_score += 10
    elif len(risk_factors) == 0:
        risk_score -= 20
    
    # Assess risk level appropriateness
    overall_risk = risk_assessment.get("overall_risk_level", "medium")
    if overall_risk in ["low", "medium", "high"]:
        risk_score += 10
    
    return {
        "score": max(0, min(100, risk_score)),
        "risk_coverage": "comprehensive" if len(risk_factors) >= 3 else "adequate" if len(risk_factors) >= 2 else "inadequate",
        "risk_level_appropriateness": "appropriate" if overall_risk in ["low", "medium", "high"] else "inappropriate"
    }

def _assess_recommendation_strength(recommendation: Dict[str, Any], confidence_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Assess the strength of the trading recommendation."""
    
    action = recommendation.get("action", "HOLD")
    confidence = confidence_metrics.get("overall_confidence", 50)
    position_size = recommendation.get("position_size", "normal")
    
    strength_score = 50.0
    
    # Assess action clarity
    if action in ["BUY", "SELL"]:
        strength_score += 15
    elif action in ["BUY_WEAK", "SELL_WEAK"]:
        strength_score += 10
    
    # Assess confidence-action alignment
    if confidence > 70 and action in ["BUY", "SELL"]:
        strength_score += 15
    elif confidence < 40 and action == "HOLD":
        strength_score += 10
    
    # Assess position size appropriateness
    if confidence > 80 and position_size == "large":
        strength_score += 10
    elif confidence < 40 and position_size == "small":
        strength_score += 10
    
    return {
        "score": max(0, min(100, strength_score)),
        "action_clarity": "clear" if action in ["BUY", "SELL"] else "moderate" if action in ["BUY_WEAK", "SELL_WEAK"] else "neutral",
        "confidence_action_alignment": "strong" if strength_score > 70 else "moderate" if strength_score > 50 else "weak"
    }

def _calculate_evaluation_score(evaluation_results: Dict[str, Any]) -> float:
    """Calculate overall evaluation score."""
    
    scores = [
        evaluation_results.get("prediction_quality", {}).get("score", 50),
        evaluation_results.get("technical_consistency", {}).get("score", 50),
        evaluation_results.get("risk_assessment", {}).get("score", 50),
        evaluation_results.get("recommendation_strength", {}).get("score", 50)
    ]
    
    return sum(scores) / len(scores)

def _generate_optimization_suggestions(evaluation_results: Dict[str, Any]) -> list:
    """Generate optimization suggestions based on evaluation."""
    
    suggestions = []
    overall_score = evaluation_results.get("overall_score", 50)
    
    if overall_score < 60:
        suggestions.append("Consider improving data quality and analysis depth")
        suggestions.append("Enhance technical indicator calculations")
        suggestions.append("Strengthen risk assessment methodology")
    
    prediction_quality = evaluation_results.get("prediction_quality", {})
    if prediction_quality.get("confidence_adequacy") == "low":
        suggestions.append("Improve confidence calculation methodology")
    
    technical_consistency = evaluation_results.get("technical_consistency", {})
    if technical_consistency.get("signal_alignment") == "weak":
        suggestions.append("Enhance technical signal alignment with predictions")
    
    risk_assessment = evaluation_results.get("risk_assessment", {})
    if risk_assessment.get("risk_coverage") == "inadequate":
        suggestions.append("Expand risk factor analysis")
    
    return suggestions
