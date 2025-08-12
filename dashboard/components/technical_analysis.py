import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from dashboard.utils import humanize_label


def display_technical_analysis(technical_summary: dict, ticker: str | None = None) -> None:
    """Display technical analysis results, with optional trend chart if ticker provided.

    Ensures trend analysis never shows Unknown by deriving from moving averages as fallback.
    """
    df = None
    df_smas: dict[str, float | None] = {}
    if ticker:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="6mo", interval="1d")
            if not hist.empty:
                df = hist.copy()
                df["SMA10"] = df["Close"].rolling(10).mean()
                df["SMA20"] = df["Close"].rolling(20).mean()
                df["SMA50"] = df["Close"].rolling(50).mean()
                df["SMA200"] = df["Close"].rolling(200).mean()
                df_smas = {
                    "close": float(df["Close"].iloc[-1]),
                    "sma10": float(df["SMA10"].iloc[-1]) if not pd.isna(df["SMA10"].iloc[-1]) else None,
                    "sma20": float(df["SMA20"].iloc[-1]) if not pd.isna(df["SMA20"].iloc[-1]) else None,
                    "sma50": float(df["SMA50"].iloc[-1]) if not pd.isna(df["SMA50"].iloc[-1]) else None,
                    "sma200": float(df["SMA200"].iloc[-1]) if not pd.isna(df["SMA200"].iloc[-1]) else None,
                }
        except Exception:
            df = None

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Technical Indicators")

        technical_score = technical_summary.get("technical_score", 0)
        st.metric("Technical Score", f"{technical_score:.1f}/100")

        # Display key indicators
        indicators = technical_summary.get("indicators", {})
        if indicators:
            st.write("**Key Indicators:**")

            # RSI
            rsi_value = indicators.get('rsi')
            if rsi_value is not None:
                st.write(f"- RSI: {rsi_value:.2f}")
            else:
                st.write("- RSI: N/A")

            # MACD
            macd_value = indicators.get('macd')
            if macd_value is not None:
                st.write(f"- MACD: {macd_value:.4f}")
            else:
                st.write("- MACD: N/A")

            # SMA 20
            sma_20_value = indicators.get('sma_20')
            if sma_20_value is not None:
                st.write(f"- SMA 20: ${sma_20_value:.2f}")
            else:
                st.write("- SMA 20: N/A")

            # SMA 50
            sma_50_value = indicators.get('sma_50')
            if sma_50_value is not None:
                st.write(f"- SMA 50: ${sma_50_value:.2f}")
            else:
                st.write("- SMA 50: N/A")

            # SMA 200
            sma_200_value = indicators.get('sma_200')
            if sma_200_value is not None:
                st.write(f"- SMA 200: ${sma_200_value:.2f}")
            else:
                st.write("- SMA 200: N/A")

        # Display trend analysis
        trend_analysis = technical_summary.get("trend_analysis", {})
        if trend_analysis is not None:
            st.write("**Trend Analysis:**")
            trends = trend_analysis.get("trends", {}) if isinstance(trend_analysis, dict) else {}
            # Pull from enhanced structure or basic structure
            short_term = trends.get("short_term") or trend_analysis.get("short_term_trend")
            medium_term = trends.get("medium_term") or trend_analysis.get("medium_term_trend")
            long_term = trends.get("long_term") or trend_analysis.get("long_term_trend")
            trend_strength_val = trend_analysis.get("trend_strength")

            # Fallback derivation using moving averages when missing
            def derive_short_medium_long_from_ma(values: dict) -> tuple:
                st_trend = short_term
                mt_trend = medium_term
                lt_trend = long_term
                ts_strength = trend_strength_val
                if values and values.get("close"):
                    close = values["close"]
                    sma10 = values.get("sma10")
                    sma20 = values.get("sma20")
                    sma50 = values.get("sma50")
                    sma200 = values.get("sma200")
                    # Short-term: 10-day
                    if not st_trend:
                        if sma10 is not None:
                            st_trend = "bullish" if close > sma10 else "bearish"
                    # Medium-term: 20/50 relationship
                    if not mt_trend:
                        if sma20 is not None and sma50 is not None:
                            if close > sma50 and sma20 > sma50:
                                mt_trend = "bullish"
                            elif close < sma50 and sma20 < sma50:
                                mt_trend = "bearish"
                            else:
                                mt_trend = "sideways"
                    # Long-term: 200-day
                    if not lt_trend and sma200 is not None:
                        lt_trend = "bullish" if close > sma200 else "bearish"
                    # Strength fallback
                    if ts_strength is None and sma50 is not None and sma50 != 0:
                        ts_strength = abs(close - sma50) / abs(sma50) * 100
                return st_trend, mt_trend, lt_trend, ts_strength

            # If any missing or 'Unknown', derive using fetched data
            if (not short_term or short_term == "Unknown") or (not medium_term or medium_term == "Unknown") or (not long_term or long_term == "Unknown") or (trend_strength_val is None):
                short_term, medium_term, long_term, trend_strength_val = derive_short_medium_long_from_ma(df_smas)

            # Defaults if still missing
            short_term = short_term or "sideways"
            medium_term = medium_term or "sideways"
            long_term = long_term or "sideways"
            trend_strength_val = trend_strength_val if isinstance(trend_strength_val, (int, float)) else 0.0

            st.write(f"- Short-term: {short_term}")
            st.write(f"- Medium-term: {medium_term}")
            st.write(f"- Long-term: {long_term}")
            st.write(f"- Trend Strength: {float(trend_strength_val):.1f}%")
            adx_strength = trend_analysis.get("adx_strength")
            if isinstance(adx_strength, (int, float)):
                st.write(f"- ADX Strength: {adx_strength:.1f}")

    with col2:
        st.subheader("Technical Signals")

        # Display trading signals
        trading_signals = technical_summary.get("trading_signals", {})
        if trading_signals:
            overall_recommendation = trading_signals.get("overall_recommendation", "HOLD")
            signal_strength = trading_signals.get("signal_strength", 0)
            total_signals = trading_signals.get("total_signals", 0)

            st.write(f"**Overall Recommendation:** {humanize_label(overall_recommendation)}")
            st.write(f"**Signal Strength:** {signal_strength}")
            st.write(f"**Total Signals:** {total_signals}")

            # Display individual signals
            signals = trading_signals.get("signals", [])
            if signals:
                st.write("**Individual Signals:**")
                for signal in signals[:5]:  # Show first 5 signals
                    if isinstance(signal, dict):
                        signal_type = humanize_label(signal.get("type", "Unknown"))
                        indicator = humanize_label(signal.get("indicator", "Unknown"))
                        strength = humanize_label(signal.get("strength", "Unknown"))
                        reason = signal.get("reason", "")
                        st.write(f"- {signal_type} ({indicator}): {reason}")
                    else:
                        st.write(f"- {signal}")

        # Display support/resistance
        support_resistance = technical_summary.get("support_resistance", {})
        if support_resistance:
            st.write("**Support & Resistance:**")
            current_price = support_resistance.get("current_price", 0)
            nearest_support = support_resistance.get("nearest_support", 0)
            nearest_resistance = support_resistance.get("nearest_resistance", 0)

            if current_price is not None:
                st.write(f"- Current Price: ${current_price:.2f}")
            else:
                st.write("- Current Price: N/A")

            if nearest_support is not None:
                st.write(f"- Nearest Support: ${nearest_support:.2f}")
            else:
                st.write("- Nearest Support: N/A")

            if nearest_resistance is not None:
                st.write(f"- Nearest Resistance: ${nearest_resistance:.2f}")
            else:
                st.write("- Nearest Resistance: N/A")

        # Display patterns
        patterns = technical_summary.get("patterns", [])
        if patterns:
            st.write(f"**Patterns Detected:** {len(patterns)}")
            for pattern in patterns[:3]:  # Show first 3 patterns
                if isinstance(pattern, dict):
                    pattern_name = humanize_label(pattern.get("pattern", "Unknown"))
                    signal = humanize_label(pattern.get("signal", "Unknown"))
                    st.write(f"- {pattern_name}: {signal}")
                else:
                    st.write(f"- {pattern}")

    # Optional trend chart with MAs
    if ticker and df is not None:
        try:
            st.subheader("Trend Chart (6mo)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color="#1f77b4")))
            if "SMA20" in df:
                fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA 20", line=dict(color="#2ca02c", width=1)))
            if "SMA50" in df:
                fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA 50", line=dict(color="#ff7f0e", width=1)))
            if "SMA200" in df:
                fig.add_trace(go.Scatter(x=df.index, y=df["SMA200"], name="SMA 200", line=dict(color="#9467bd", width=1)))
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"Trend chart unavailable: {str(e)}")


