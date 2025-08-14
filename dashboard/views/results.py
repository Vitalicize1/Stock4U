from __future__ import annotations

import streamlit as st

from dashboard.components.technical_analysis import display_technical_analysis
from dashboard.components.prediction_details import display_prediction_details
from dashboard.components.risk_assessment import display_risk_assessment
from dashboard.components.market_data import display_market_data
from dashboard.utils import humanize_label


def display_results(ticker: str, result: dict) -> None:
    """Display the prediction results in a comprehensive dashboard."""

    st.header(f"Analysis Results for {ticker}")
    st.markdown("---")

    # Prefer new structure if available, even when a final_summary exists
    if "prediction_result" in result:
        # New structure (enhanced or basic). Support flat or nested prediction objects
        prediction_result = result.get("prediction_result", {})
        technical_source = result.get("enhanced_technical_analysis") or result.get("technical_analysis", {})
        sentiment_integration = result.get("sentiment_integration", {})

        # Extract prediction data (flat or nested under "prediction")
        prediction = prediction_result.get("prediction", prediction_result)

        # Confidence: prefer top-level confidence_metrics; else nested
        cm = result.get("confidence_metrics") or prediction_result.get("confidence_metrics") or {}
        overall_conf = cm.get("overall_confidence")
        # If missing, compute on-the-fly to avoid falling back to raw model confidence
        if overall_conf is None:
            try:
                from agents.tools.prediction_agent_tools import calculate_confidence_metrics_tool
                conf_res = calculate_confidence_metrics_tool.invoke({
                    "technical_analysis": technical_source,
                    "sentiment_integration": sentiment_integration,
                    "prediction_result": prediction,
                    "sentiment_analysis": result.get("sentiment_analysis", {})
                })
                cm = (conf_res or {}).get("confidence_metrics") or {}
                overall_conf = cm.get("overall_confidence")
            except Exception:
                # Fallback to model confidence only if calculation failed
                overall_conf = prediction.get("confidence", 0)

        model_conf = prediction.get("confidence")
        prediction_summary = {
            "direction": humanize_label(prediction.get("direction", "Unknown")),
            "confidence": overall_conf or 0,
            "model_confidence": model_conf if isinstance(model_conf, (int, float)) else None,
            "reasoning": prediction.get("reasoning", "No reasoning provided"),
            "key_factors": prediction.get("key_factors", [])
        }

        # Extract technical data
        technical_summary = {
            "technical_score": technical_source.get("technical_score", 0),
            "indicators": technical_source.get("indicators", {}),
            "trend_analysis": technical_source.get("trend_analysis", {}),
            "trading_signals": technical_source.get("trading_signals", {}),
            "support_resistance": technical_source.get("support_resistance", {}),
            "patterns": technical_source.get("patterns", [])
        }

        # Extract recommendation (prefer top-level, then nested)
        rec = result.get("recommendation") or prediction_result.get("recommendation", {})
        final_recommendation = {
            "action": humanize_label(rec.get("action", "HOLD")),
            "confidence": prediction_summary.get("confidence", 0)
        }

        evaluation_summary = {
            "overall_score": technical_summary.get("technical_score", 0),
            "prediction_quality": {
                "score": prediction_summary.get("confidence", 0),
                "confidence_adequacy": "adequate" if prediction_summary.get("confidence", 0) > 50 else "low",
                "reasoning_quality": "good" if prediction_summary.get("reasoning") else "poor"
            }
        }
        sentiment_block = result.get("sentiment_integration", {})
    elif "final_summary" in result:
        # Old structure
        final_summary = result.get("final_summary", {})
        prediction_summary = final_summary.get("prediction_summary", {})
        technical_summary = final_summary.get("technical_summary", {})
        evaluation_summary = final_summary.get("evaluation_summary", {})
        # Humanize fields for display
        final_recommendation_raw = final_summary.get("final_recommendation", {}) or {}
        final_recommendation = {
            "action": humanize_label(final_recommendation_raw.get("action", "HOLD")),
            "confidence": prediction_summary.get("confidence", 0),
        }
    else:
        st.error("No prediction results available")
        return

    # Show friendly error banner if available
    err = result.get("error_info") or result.get("prediction_result", {}).get("error_info")
    if isinstance(err, dict):
        code = err.get("code", "error")
        msg = err.get("message", "An internal error occurred. A safe fallback was used.")
        st.info(f"Note: {humanize_label(code)} — {msg}")

    # Normalized, UI-stable result for downstream use (e.g., Jira attachment)
    timeframe = result.get("timeframe") or "1d"
    normalized_result = {
        "ticker": ticker,
        "timeframe": timeframe,
        "status": result.get("status", "success"),
        "prediction": prediction_summary,
        "technical_analysis": technical_summary,
        "recommendation": final_recommendation,
        "evaluation": evaluation_summary,
        "sentiment_integration": result.get("sentiment_integration", {}),
        "confidence_metrics": result.get("confidence_metrics") or result.get("prediction_result", {}).get("confidence_metrics", {}),
        "quota_status": result.get("quota_status"),
    }
    # Store normalized result in session for optional integrations (no UI controls)
    st.session_state['normalized_result'] = normalized_result

    # Create three columns for key metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Prediction Direction",
            value=prediction_summary.get("direction", "Unknown"),
            delta=None
        )

    with col2:
        confidence = prediction_summary.get("confidence", 0)
        model_conf = prediction_summary.get("model_confidence")
        subtitle = None if model_conf is None else f"Model: {model_conf:.1f}%"
        st.metric(
            label="Confidence (Blended)",
            value=f"{confidence:.1f}%",
            delta=subtitle
        )

    with col3:
        action = final_recommendation.get("action", "HOLD")
        st.metric(
            label="Recommendation",
            value=action,
            delta=None
        )

    st.markdown("---")

    # Compact sentiment and market context summary cards
    s_col, m_col = st.columns(2)
    with s_col:
        st.subheader("Sentiment Snapshot")
        sa = result.get("sentiment_analysis", {}) or {}
        overall = (sa.get("overall_sentiment", {}) or {})
        label = humanize_label(overall.get("sentiment_label", "neutral"))
        score = overall.get("sentiment_score", 0)
        st.metric("Overall Sentiment", label, f"{score:+.2f}")
        integ = (result.get("sentiment_integration", {}) or {}).get("integrated_analysis", {}) or {}
        if integ:
            st.metric("Integrated Score", f"{float(integ.get('integrated_score', 0)):.1f}/100")

    with m_col:
        st.subheader("Market Snapshot")
        market = ((result.get("data", {}) or {}).get("market_data", {}) or {})
        spx = market.get("sp500_current")
        spx_chg = market.get("sp500_change_pct")
        trend = market.get("market_trend", "neutral")
        if spx is not None and spx_chg is not None:
            st.metric("S&P 500", f"{spx:.2f}", f"{spx_chg:+.2f}%")
        st.metric("Market Trend", trend.title())

    # (Run timings removed from UI)

    # Create tabs for detailed analysis
    tab1, tab2, tab3, tab4 = st.tabs(["Technical Analysis", "Prediction Details", "Risk Assessment", "Market Data"])

    with tab1:
        display_technical_analysis(technical_summary, ticker)

    with tab2:
        display_prediction_details(prediction_summary, evaluation_summary)

    with tab3:
        display_risk_assessment(result)

    with tab4:
        display_market_data(ticker)


