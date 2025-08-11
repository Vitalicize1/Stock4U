"""
Risk Assessor Tools

Deterministic tools that compute a structured risk assessment payload from
available analysis artifacts (technical, sentiment, market, price, company).
Designed to be LLM-free and fast.
"""

from typing import Dict, Any, List
from langchain_core.tools import tool


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _derive_market_risk(market_data: Dict[str, Any]) -> str:
    indices = (market_data or {}).get("indices", {})
    vix = _safe_float((indices.get("vix", {}) or {}).get("current", 0.0))
    spx_change_pct = _safe_float((indices.get("sp500", {}) or {}).get("change_pct", 0.0))
    if vix >= 25 or spx_change_pct <= -1.0:
        return "high"
    if vix >= 18 or abs(spx_change_pct) >= 0.7:
        return "medium"
    return "low"


def _derive_volatility_risk(technical_analysis: Dict[str, Any], price_data: Dict[str, Any]) -> str:
    adx = _safe_float((technical_analysis.get("trend_analysis", {}) or {}).get("adx_strength", 0.0))
    vol = price_data.get("volatility")
    vol = _safe_float(vol, None) if vol is not None else None
    if vol is not None:
        if vol >= 0.035 or adx >= 35:
            return "high"
        if vol >= 0.015 or adx >= 20:
            return "medium"
        return "low"
    if adx:
        return "high" if adx >= 35 else ("medium" if adx >= 20 else "low")
    return "unknown"


def _derive_liquidity_risk(price_data: Dict[str, Any]) -> str:
    avg_vol = price_data.get("avg_volume")
    avg_vol = _safe_float(avg_vol, None) if avg_vol is not None else None
    if avg_vol is None:
        return "unknown"
    if avg_vol < 200_000:
        return "high"
    if avg_vol < 1_000_000:
        return "medium"
    return "low"


def _derive_sector_risk(technical_analysis: Dict[str, Any], company_info: Dict[str, Any], fallback_market_risk: str) -> str:
    sector = (company_info or {}).get("sector") or (company_info or {}).get("basic_info", {}).get("sector")
    if not sector:
        return fallback_market_risk
    t_strength = _safe_float((technical_analysis.get("trend_analysis", {}) or {}).get("trend_strength", 0.0))
    if t_strength < 10:
        return "high"
    if t_strength < 25:
        return "medium"
    return "low"


def _derive_sentiment_risk(sentiment_analysis: Dict[str, Any]) -> str:
    s = (sentiment_analysis.get("overall_sentiment", {}) or {}).get("sentiment_score", 0.0)
    try:
        s_abs = abs(float(s))
    except Exception:
        s_abs = 0.0
    if s_abs > 0.5:
        return "high"
    if s_abs > 0.2:
        return "medium"
    return "low"


def _aggregate_overall_risk(parts: List[str]) -> str:
    map_score = {"low": 1, "medium": 2, "high": 3, "unknown": 2}
    scores = [map_score.get(p, 2) for p in parts]
    avg = sum(scores) / max(1, len(scores))
    if avg < 1.5:
        return "low"
    if avg > 2.5:
        return "high"
    return "medium"


def _generate_risk_warnings(risk: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []
    if risk.get("market_risk") == "high":
        warnings.append("Broad market risk elevated (VIX/SPX). Consider smaller positions.")
    if risk.get("volatility_risk") == "high":
        warnings.append("High volatility expected – use wider stops and reduced size.")
    if risk.get("liquidity_risk") in {"medium", "high"}:
        warnings.append("Lower liquidity – beware of slippage and gaps.")
    if risk.get("sentiment_risk") == "high":
        warnings.append("Extreme sentiment – outcomes may be more erratic.")
    return warnings


@tool
def compute_risk_assessment_tool(technical_analysis: Dict[str, Any],
                                 price_data: Dict[str, Any],
                                 market_data: Dict[str, Any],
                                 company_info: Dict[str, Any],
                                 sentiment_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute a structured risk assessment using deterministic heuristics.
    Returns keys: market_risk, volatility_risk, liquidity_risk, sector_risk,
    sentiment_risk, overall_risk_level, risk_warnings.
    """
    try:
        market_risk = _derive_market_risk(market_data or {})
        volatility_risk = _derive_volatility_risk(technical_analysis or {}, price_data or {})
        liquidity_risk = _derive_liquidity_risk(price_data or {})
        sector_risk = _derive_sector_risk(technical_analysis or {}, company_info or {}, market_risk)
        sentiment_risk = _derive_sentiment_risk(sentiment_analysis or {})

        overall = _aggregate_overall_risk([market_risk, volatility_risk, liquidity_risk, sector_risk, sentiment_risk])
        risk = {
            "market_risk": market_risk,
            "volatility_risk": volatility_risk,
            "liquidity_risk": liquidity_risk,
            "sector_risk": sector_risk,
            "sentiment_risk": sentiment_risk,
            "overall_risk_level": overall,
        }
        risk["risk_warnings"] = _generate_risk_warnings(risk)
        return {"status": "success", "risk_assessment": risk}
    except Exception as e:
        return {"status": "error", "error": f"compute_risk_assessment_tool failed: {e}", "risk_assessment": {}}


