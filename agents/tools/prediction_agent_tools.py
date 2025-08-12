"""
Prediction Agent Tools

Toolized helpers for the prediction stage:
- generate_llm_prediction_tool: delegate to best available LLM and return a prediction
- generate_rule_based_prediction_tool: fallback prediction when LLM unavailable
- generate_ml_prediction_tool: ML-based probability of UP
- ensemble_prediction_tool: combine ML + LLM + rule-based with calibrated weights
- calculate_confidence_metrics_tool: compute confidence metrics from analysis
- generate_recommendation_tool: derive a recommendation from prediction + confidence
"""

from typing import Dict, Any, Optional
from datetime import datetime
from langchain_core.tools import tool
import math

from utils.result_cache import get_cached_result


@tool
def generate_llm_prediction_tool(analysis_summary: str, offline: bool = False) -> Dict[str, Any]:
    """
    Generate a prediction using the best available LLM via delegation.

    Args:
        analysis_summary: Comprehensive analysis summary string

    Returns:
        Dict with keys: prediction_result (dict)
    """
    try:
        if offline:
            return {
                "status": "error",
                "error": "offline mode enabled; LLM prediction disabled",
                "prediction_result": None,
            }
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
def generate_ml_prediction_tool(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a prediction using a lightweight traditional ML model over engineered features.

    Args:
        state: The full pipeline state (to extract features already computed)

    Returns:
        Dict with keys: prediction_result (dict)
    """
    try:
        from ml.features import build_features_from_state
        from ml.loader import load_latest_model
        from ml.model import load_default_model
        import numpy as np
        import pandas as pd

        X, feats = build_features_from_state(state)
        artifact = load_latest_model()
        if artifact and artifact.feature_names:
            # Align to artifact feature order if available
            feature_names = artifact.feature_names
            feats_ordered = []
            for name in feature_names:
                feats_ordered.append(feats.get(name, 0.0))
            Xa = np.array(feats_ordered, dtype=float)
            clf = artifact.model
            try:
                # Use DataFrame with feature names to avoid sklearn warnings and ensure alignment
                X_df = pd.DataFrame([Xa.tolist()], columns=feature_names)
                proba = clf.predict_proba(X_df)[:, 1][0]
                p_up = float(proba)
            except Exception:
                # Fallback to default linear model if shape mismatch
                model = load_default_model(len(X))
                p_up = model.predict_proba_up(X)
        else:
            model = load_default_model(len(X))
            p_up = model.predict_proba_up(X)

        direction = "UP" if p_up >= 0.55 else ("DOWN" if p_up <= 0.45 else "NEUTRAL")
        confidence = float(abs(p_up - 0.5) * 200)  # map 0.5->0, 1.0->100, 0.0->100

        prediction = {
            "direction": direction,
            "confidence": round(confidence, 1),
            "price_target": None,
            "reasoning": "Traditional ML model combining technical and sentiment features.",
            "key_factors": [
                f"RSI={feats.get('rsi'):.2f}",
                f"TrendStrength={feats.get('trend_strength'):.2f}",
                f"IntegratedScore={feats.get('integrated_score'):.2f}",
            ],
            "model": {
                "type": "linear",
                "proba_up": round(float(p_up) * 100.0, 1)
            }
        }

        return {
            "status": "success",
            "prediction_result": prediction,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"ML prediction failed: {str(e)}",
            "prediction_result": None
        }


def _direction_to_proba_up(pred: Dict[str, Any]) -> Optional[float]:
    """Best-effort mapping from a prediction dict to P(UP) in [0,1]."""
    if not isinstance(pred, dict):
        return None
    # Preferred explicit probability
    model = pred.get("model") or {}
    if isinstance(model, dict) and "proba_up" in model:
        try:
            v = float(model["proba_up"]) / (100.0 if model["proba_up"] > 1 else 1.0)
            return max(0.0, min(1.0, v))
        except Exception:
            pass

    direction = str(pred.get("direction", "NEUTRAL")).upper()
    conf = pred.get("confidence")
    try:
        c = float(conf) if conf is not None else 50.0
    except Exception:
        c = 50.0

    # Map direction + confidence -> probability
    # c in [0,100] => delta in [0,0.5]
    delta = max(0.0, min(100.0, c)) / 200.0
    base = 0.5 + delta
    if direction in ("UP", "BUY", "STRONG_BUY"):
        return max(0.0, min(1.0, base))
    if direction in ("DOWN", "SELL", "STRONG_SELL"):
        return 1.0 - max(0.0, min(1.0, base))
    # Neutral
    return 0.5


@tool
def ensemble_prediction_tool(state: Dict[str, Any], analysis_summary: str = "", offline: bool = False) -> Dict[str, Any]:
    """
    Combine ML + LLM + Rule-Based predictions via weighted voting with optional
    Platt-style calibration parameters loaded from cache (fitted offline via backtests).

    Inputs:
        state: Full pipeline state (ticker/timeframe/features)
        analysis_summary: Summary string for LLM/rule-based tools

    Cache keys used if present (set by offline calibration scripts):
        - ensemble_weights::{ticker}::{timeframe} -> {"llm": w1, "ml": w2, "rule": w3}
        - ensemble_platt::{ticker}::{timeframe} -> {"a": a, "b": b}
    """
    try:
        ticker = str(state.get("ticker", "UNKNOWN")).upper()
        timeframe = str(state.get("timeframe", "1d"))

        # Gather component predictions
        try:
            ml_res = generate_ml_prediction_tool.invoke({"state": state})
        except Exception:
            ml_res = {"status": "error", "prediction_result": None}
        if not offline:
            try:
                llm_res = generate_llm_prediction_tool.invoke({"analysis_summary": analysis_summary})
            except Exception:
                llm_res = {"status": "error", "prediction_result": None}
        else:
            llm_res = {"status": "error", "prediction_result": None}
        try:
            rule_res = generate_rule_based_prediction_tool.invoke({"analysis_summary": analysis_summary})
        except Exception:
            rule_res = {"status": "error", "prediction_result": None}

        ml_pred = (ml_res or {}).get("prediction_result")
        llm_pred = (llm_res or {}).get("prediction_result")
        rule_pred = (rule_res or {}).get("prediction_result")

        # Convert to probabilities
        probs = {}
        if ml_pred:
            p = _direction_to_proba_up(ml_pred)
            if p is not None:
                probs["ml"] = float(p)
        if llm_pred:
            p = _direction_to_proba_up(llm_pred)
            if p is not None:
                probs["llm"] = float(p)
        if rule_pred:
            p = _direction_to_proba_up(rule_pred)
            if p is not None:
                probs["rule"] = float(p)

        if not probs:
            return {
                "status": "error",
                "error": "No component predictions available for ensembling",
                "prediction_result": None,
            }

        # Load weights and calibration from cache (if previously calibrated via backtests)
        default_weights = {"llm": 0.5, "ml": 0.4, "rule": 0.1}
        w_cached = get_cached_result(
            f"ensemble_weights::{ticker}::{timeframe}", ttl_seconds=365 * 24 * 3600
        ) or {}
        # Guard: if cached AUC/n_samples are weak, prefer equal weights
        cached_auc = None
        cached_ns = None
        try:
            cached_auc = float(w_cached.get("auc_raw")) if w_cached else None
        except Exception:
            cached_auc = None
        try:
            cached_ns = int(w_cached.get("n_samples")) if w_cached else None
        except Exception:
            cached_ns = None

        weak_cache = (
            cached_auc is None
            or cached_auc < 0.52  # barely above random; treat as unreliable
            or (cached_ns is not None and cached_ns < 30)
        )

        if weak_cache:
            # Equal weights over available components
            eq = 1.0 / float(len(probs))
            weights = {k: (eq if k in probs else 0.0) for k in set(default_weights) | set(probs)}
        else:
            weights = {k: float(w_cached.get(k, default_weights.get(k, 0.0))) for k in set(default_weights) | set(probs)}

        # Normalize to the subset of available models
        total_w = sum(weights[k] for k in probs.keys())
        if total_w <= 0:
            # fallback equal weights on available
            eq = 1.0 / float(len(probs))
            weights = {k: (eq if k in probs else 0.0) for k in weights}
            total_w = 1.0
        else:
            for k in weights:
                if k in probs:
                    weights[k] = weights[k] / total_w
                else:
                    weights[k] = 0.0

        p_ens = sum(weights[k] * probs[k] for k in probs.keys())

        # Platt-style calibration: sigmoid(a*x + b) if available
        platt = get_cached_result(
            f"ensemble_platt::{ticker}::{timeframe}", ttl_seconds=365 * 24 * 3600
        ) or {}
        try:
            a = float(platt.get("a"))
            b = float(platt.get("b"))
            # If cache judged weak, skip calibration and use raw
            if weak_cache:
                p_cal = p_ens
            else:
                p_cal = 1.0 / (1.0 + math.exp(-(a * p_ens + b)))
        except Exception:
            p_cal = p_ens

        # Map to discrete direction and confidence
        if p_cal >= 0.55:
            direction = "UP"
        elif p_cal <= 0.45:
            direction = "DOWN"
        else:
            direction = "NEUTRAL"
        confidence = round(abs(p_cal - 0.5) * 200.0, 1)

        pred = {
            "direction": direction,
            "confidence": confidence,
            "model": {
                "type": "ensemble",
                "proba_up": round(p_cal * 100.0, 1),
                "proba_up_raw": round(p_ens * 100.0, 1),
                "weights": {k: round(float(weights.get(k, 0.0)), 3) for k in sorted(weights.keys())},
            },
            "components": {
                "ml": ml_pred,
                "llm": llm_pred,
                "rule": rule_pred,
                "probs": probs,
            },
            "reasoning": "Ensembled ML, LLM, and rule-based predictions with calibrated weights.",
        }

        return {"status": "success", "prediction_result": pred}
    except Exception as e:
        return {
            "status": "error",
            "error": f"ensemble prediction failed: {e}",
            "prediction_result": None,
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


