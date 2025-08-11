import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px


def display_risk_assessment(result: dict) -> None:
    """Display risk assessment information with a visual breakdown."""

    # Robust extraction from multiple possible structures
    final_prediction = result.get("final_prediction", {})
    prediction_result = result.get("prediction_result", {})
    risk_assessment = (
        (final_prediction.get("risk_assessment") if isinstance(final_prediction, dict) else None)
        or (prediction_result.get("risk_assessment") if isinstance(prediction_result, dict) else None)
        or result.get("risk_assessment")
        or {}
    )
    sentiment_integration = result.get("sentiment_integration", {})
    # Technical for ADX
    technical_source = result.get("enhanced_technical_analysis") or result.get("technical_analysis", {})

    # Fallback compute if unknown or missing
    def fill_risk_fallbacks(risk: dict) -> dict:
        try:
            ticker = (result.get("ticker") or result.get("prediction_result", {}).get("ticker")) or ""
        except Exception:
            ticker = ""

        # Market risk from VIX/SPX
        if str(risk.get("market_risk", "unknown")).lower() == "unknown":
            try:
                vix = yf.Ticker("^VIX").history(period="5d").iloc[-1]["Close"]
            except Exception:
                vix = 0
            try:
                spx_hist = yf.Ticker("^GSPC").history(period="5d")
                spx_change_pct = 0.0
                if not spx_hist.empty and len(spx_hist) > 1:
                    spx_change_pct = float((spx_hist.iloc[-1]["Close"] - spx_hist.iloc[-2]["Close"]) / spx_hist.iloc[-2]["Close"] * 100)
            except Exception:
                spx_change_pct = 0.0
            if vix >= 25 or spx_change_pct <= -1.0:
                risk["market_risk"] = "high"
            elif vix >= 18 or abs(spx_change_pct) >= 0.7:
                risk["market_risk"] = "medium"
            else:
                risk["market_risk"] = "low"

        # Volatility risk from 20d stdev if missing
        if str(risk.get("volatility_risk", "unknown")).lower() == "unknown":
            try:
                ticker_symbol = result.get("ticker") or ticker
                if ticker_symbol:
                    hist = yf.Ticker(ticker_symbol).history(period="3mo", interval="1d")
                    vol = 0.0
                    if not hist.empty and len(hist) >= 21:
                        ret = hist["Close"].pct_change()
                        vol = float(ret.tail(20).std())
                    if vol >= 0.035:
                        risk["volatility_risk"] = "high"
                    elif vol >= 0.015:
                        risk["volatility_risk"] = "medium"
                    else:
                        risk["volatility_risk"] = "low"
            except Exception:
                risk.setdefault("volatility_risk", "medium")

        # Liquidity risk from avg volume
        if str(risk.get("liquidity_risk", "unknown")).lower() == "unknown":
            try:
                ticker_symbol = result.get("ticker") or ticker
                if ticker_symbol:
                    hist = yf.Ticker(ticker_symbol).history(period="3mo", interval="1d")
                    avg_vol = float(hist["Volume"].mean()) if not hist.empty else 0
                    if avg_vol < 200_000:
                        risk["liquidity_risk"] = "high"
                    elif avg_vol < 1_000_000:
                        risk["liquidity_risk"] = "medium"
                    else:
                        risk["liquidity_risk"] = "low"
            except Exception:
                risk.setdefault("liquidity_risk", "medium")

        # Sector risk from trend strength or mirror market
        if str(risk.get("sector_risk", "unknown")).lower() == "unknown":
            t_strength = (technical_source.get("trend_analysis", {}) or {}).get("trend_strength")
            if isinstance(t_strength, (int, float)):
                if t_strength < 10:
                    risk["sector_risk"] = "high"
                elif t_strength < 25:
                    risk["sector_risk"] = "medium"
                else:
                    risk["sector_risk"] = "low"
            else:
                risk["sector_risk"] = risk.get("market_risk", "medium")

        # Sentiment risk from sentiment score
        if str(risk.get("sentiment_risk", "unknown")).lower() == "unknown":
            s_score = (sentiment_integration.get("sentiment_insights", {}) or {}).get("sentiment_score")
            if s_score is None:
                s_score = (result.get("sentiment_analysis", {}).get("overall_sentiment", {}) or {}).get("sentiment_score")
            try:
                s = abs(float(s_score)) if s_score is not None else 0.0
                if s > 0.5:
                    risk["sentiment_risk"] = "high"
                elif s > 0.2:
                    risk["sentiment_risk"] = "medium"
                else:
                    risk["sentiment_risk"] = "low"
            except Exception:
                risk["sentiment_risk"] = "low"

        # Compute overall if missing
        if not risk.get("overall_risk_level"):
            map_score = {"low": 1, "medium": 2, "high": 3, "unknown": 2}
            parts = [risk.get("market_risk", "unknown"), risk.get("volatility_risk", "unknown"), risk.get("liquidity_risk", "unknown"), risk.get("sector_risk", "unknown"), risk.get("sentiment_risk", "unknown")]
            avg = sum(map_score.get(str(x).lower(), 2) for x in parts) / len(parts)
            risk["overall_risk_level"] = "low" if avg < 1.5 else ("high" if avg > 2.5 else "medium")
        return risk

    risk_assessment = fill_risk_fallbacks(risk_assessment)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("⚠️ Risk Assessment")

        overall_risk = str(risk_assessment.get("overall_risk_level", "Unknown")).upper()
        st.metric("Overall Risk Level", overall_risk)

        # Visual risk breakdown
        level_map = {"low": 1, "medium": 2, "high": 3, "unknown": 2, "Unknown": 2}
        categories = [
            ("Market", risk_assessment.get("market_risk", "unknown")),
            ("Volatility", risk_assessment.get("volatility_risk", "unknown")),
            ("Liquidity", risk_assessment.get("liquidity_risk", "unknown")),
            ("Sector", risk_assessment.get("sector_risk", "unknown")),
            ("Sentiment", risk_assessment.get("sentiment_risk", "unknown")),
        ]
        df = pd.DataFrame({
            "Risk": [c[0] for c in categories],
            "Level": [level_map.get(str(c[1]).lower(), 2) for c in categories],
            "Label": [str(c[1]).title() for c in categories],
        })
        fig = px.bar(df, x="Risk", y="Level", color="Label", range_y=[0, 3], title="Risk Breakdown (0=none, 3=high)",
                     color_discrete_sequence=["#2ca02c", "#ff7f0e", "#d62728", "#1f77b4"])  
        fig.update_layout(showlegend=True, height=320, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

        # Extra metrics if available
        adx_strength = (technical_source.get("trend_analysis", {}) or {}).get("adx_strength")
        if isinstance(adx_strength, (int, float)):
            st.write(f"- ADX Strength: {adx_strength:.1f}")

        # Display sentiment insights if available
        if sentiment_integration:
            sentiment_insights = sentiment_integration.get("sentiment_insights", {})
            if sentiment_insights:
                st.write("**Sentiment Impact:**")
                st.write(f"- Sentiment Score: {sentiment_insights.get('sentiment_score', 0):.3f}")
                st.write(f"- Sentiment Label: {sentiment_insights.get('sentiment_label', 'Unknown')}")
                st.write(f"- Impact Assessment: {sentiment_insights.get('impact_assessment', 'Unknown')}")

    with col2:
        st.subheader("🚨 Risk Warnings")

        # Get risk warnings from various sources
        warnings: list[str] = []

        # From risk assessment
        if risk_assessment.get("risk_warnings"):
            warnings.extend(risk_assessment.get("risk_warnings", []))

        # From sentiment integration
        if sentiment_integration:
            sentiment_insights = sentiment_integration.get("sentiment_insights", {})
            recommendations = sentiment_insights.get("recommendations", [])
            if recommendations:
                warnings.extend(recommendations)

        if warnings:
            for warning in warnings:
                st.warning(warning)
        else:
            st.info("No specific risk warnings")

        st.subheader("📋 Next Steps")

        # Generate next steps based on analysis
        next_steps: list[str] = []

        # Add steps based on prediction direction
        prediction_result = result.get("prediction_result", {})
        prediction = prediction_result.get("prediction", {})
        direction = prediction.get("direction", "Unknown")

        if direction == "UP":
            next_steps.extend([
                "Consider buying with appropriate position sizing",
                "Set stop-loss orders to manage risk",
                "Monitor for confirmation of upward movement"
            ])
        elif direction == "DOWN":
            next_steps.extend([
                "Consider selling or reducing position",
                "Monitor support levels for potential reversal",
                "Wait for clearer signals before new positions"
            ])
        else:
            next_steps.extend([
                "Maintain current positions",
                "Monitor for breakout signals",
                "Wait for clearer directional movement"
            ])

        for step in next_steps:
            st.write(f"• {step}")


