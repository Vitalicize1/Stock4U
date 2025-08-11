import streamlit as st
import pandas as pd
from langgraph_flow import run_prediction, run_chatbot_workflow
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import yfinance as yf
import json

def main():
    st.set_page_config(
        page_title="Agentic Stock Predictor",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("🤖 Agentic Stock Prediction System")
    st.markdown("---")
    
    # Create tabs for different sections (remove workflow tab for end-users)
    tab1, tab3, tab4 = st.tabs(["📊 Predictions", "💬 Chatbot", "📈 Market Data"])
    
    with tab1:
        # Sidebar form so pressing Enter submits and runs prediction
        st.sidebar.header("📊 Prediction Settings")
        with st.sidebar.form("prediction_form", clear_on_submit=False):
            ticker_input = st.text_input(
                "Enter Stock Ticker",
                value="AAPL",
                placeholder="e.g., AAPL, MSFT, GOOGL"
            )
            timeframe = st.selectbox(
                "Prediction Timeframe",
                options=["1d", "1w", "1m"],
                index=0
            )
            low_api_mode = st.toggle("Low API mode (no LLM for prediction)", value=False, help="Use rule-based prediction and skip LLM calls to save quota.")
            fast_ta_mode = st.toggle("Fast TA mode (quicker technical analysis)", value=False, help="Run a minimal technical analysis (skip heavy indicators/patterns) for speed.")
            submitted = st.form_submit_button("🚀 Run Prediction")

        if submitted:
            ticker = (ticker_input or "").upper().strip()
            if ticker:
                with st.spinner(f"Analyzing {ticker}..."):
                    try:
                        result = run_prediction(ticker, timeframe, low_api_mode=low_api_mode, fast_ta_mode=fast_ta_mode)
                        if 'quota_status' in result:
                            st.sidebar.markdown("---")
                            st.sidebar.header("📊 LLM Provider Status")
                            quota_status = result['quota_status']
                            for provider, status in quota_status.items():
                                if status["available"]:
                                    st.sidebar.success(f"✅ {provider.upper()}: {status['reason']}")
                                else:
                                    st.sidebar.error(f"❌ {provider.upper()}: {status['reason']}")
                        display_results(ticker, result)
                    except Exception as e:
                        st.error(f"Error analyzing {ticker}: {str(e)}")
            else:
                st.error("Please enter a valid ticker symbol")
        
        # Quick analysis section
        st.sidebar.markdown("---")
        st.sidebar.header("⚡ Quick Analysis")
        
        popular_tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA"]
        selected_quick = st.sidebar.selectbox("Popular Stocks", popular_tickers)
        
        if st.sidebar.button("Quick Analyze"):
            with st.spinner(f"Quick analysis of {selected_quick}..."):
                try:
                    result = run_prediction(selected_quick, "1d", low_api_mode=low_api_mode, fast_ta_mode=fast_ta_mode)
                    
                    # Display quota status if available
                    if 'quota_status' in result:
                        st.sidebar.markdown("---")
                        st.sidebar.header("📊 LLM Provider Status")
                        
                        quota_status = result['quota_status']
                        for provider, status in quota_status.items():
                            if status["available"]:
                                st.sidebar.success(f"✅ {provider.upper()}: {status['reason']}")
                            else:
                                st.sidebar.error(f"❌ {provider.upper()}: {status['reason']}")
                    
                    display_results(selected_quick, result)
                except Exception as e:
                    st.error(f"Error in quick analysis: {str(e)}")
    
    
    with tab3:
        st.header("💬 AI Chatbot Assistant")
        st.markdown("---")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me about stock predictions, the workflow, or any questions..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate assistant response using LangGraph workflow
            with st.chat_message("assistant"):
                try:
                    with st.spinner("🤖 Processing your request..."):
                        # Use the new chatbot workflow
                        result = run_chatbot_workflow(prompt)
                        
                        # Extract the chatbot response
                        chatbot_response = result.get("chatbot_response", {})
                        response_type = chatbot_response.get("response_type", "greeting")
                        
                        if response_type == "stock_analysis":
                            # If stock analysis was performed, format the results
                            if "final_summary" in result:
                                response = format_stock_analysis_response(result.get("ticker"), result)
                            else:
                                response = chatbot_response.get("message", "Sorry, I couldn't complete the analysis.")
                        else:
                            # For other response types, use the chatbot message
                            response = chatbot_response.get("message", "I'm not sure how to help with that.")
                    
                    st.markdown(response)
                    
                except Exception as e:
                    error_response = f"❌ Sorry, I encountered an error: {str(e)}"
                    st.markdown(error_response)
                    response = error_response
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Sidebar for chatbot features
        st.sidebar.markdown("---")
        st.sidebar.header("🤖 Chatbot Features")
        
        if st.sidebar.button("🔄 Clear Chat"):
            st.session_state.messages = []
            st.rerun()
        
        if st.sidebar.button("📊 Quick Stock Analysis"):
            quick_analysis_prompt = "Can you analyze AAPL stock for me?"
            st.session_state.messages.append({"role": "user", "content": quick_analysis_prompt})
            with st.chat_message("user"):
                st.markdown(quick_analysis_prompt)
            
            with st.chat_message("assistant"):
                try:
                    with st.spinner("🤖 Analyzing AAPL..."):
                        result = run_chatbot_workflow(quick_analysis_prompt)
                        if "final_summary" in result:
                            response = format_stock_analysis_response("AAPL", result)
                        else:
                            response = "Sorry, I couldn't complete the analysis."
                    st.markdown(response)
                except Exception as e:
                    error_response = f"❌ Error: {str(e)}"
                    st.markdown(error_response)
                    response = error_response
            
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    with tab4:
        st.header("📈 Market Data")
        st.markdown("---")
        
        # Market data section
        market_ticker = st.text_input("Enter ticker for market data:", value="AAPL")
        
        if st.button("📊 Get Market Data"):
            try:
                display_market_data(market_ticker)
            except Exception as e:
                st.error(f"Error fetching market data: {str(e)}")

def display_results(ticker: str, result: dict):
    """Display the prediction results in a comprehensive dashboard."""
    
    st.header(f"📈 Analysis Results for {ticker}")
    st.markdown("---")
    
    # Prefer new structure if available, even when a final_summary exists
    if "prediction_result" in result:
        # New structure (enhanced or basic). Support flat or nested prediction objects
        prediction_result = result.get("prediction_result", {})
        technical_source = result.get("enhanced_technical_analysis") or result.get("technical_analysis", {})
        sentiment_integration = result.get("sentiment_integration", {})

        # Extract prediction data (flat or nested under "prediction")
        prediction = prediction_result.get("prediction", prediction_result)

        # Confidence: prefer top-level confidence_metrics; else nested; else fallback to model confidence
        cm = result.get("confidence_metrics") or prediction_result.get("confidence_metrics") or {}
        overall_conf = cm.get("overall_confidence")
        if overall_conf is None:
            overall_conf = prediction.get("confidence", 0)

        model_conf = prediction.get("confidence")
        prediction_summary = {
            "direction": prediction.get("direction", "Unknown"),
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
            "action": rec.get("action", "HOLD"),
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
    elif "final_summary" in result:
        # Old structure
        final_summary = result.get("final_summary", {})
        prediction_summary = final_summary.get("prediction_summary", {})
        technical_summary = final_summary.get("technical_summary", {})
        evaluation_summary = final_summary.get("evaluation_summary", {})
        final_recommendation = final_summary.get("final_recommendation", {})
    else:
        st.error("No prediction results available")
        return
    
    # Create three columns for key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="🎯 Prediction Direction",
            value=prediction_summary.get("direction", "Unknown").upper(),
            delta=None
        )
    
    with col2:
        confidence = prediction_summary.get("confidence", 0)
        model_conf = prediction_summary.get("model_confidence")
        subtitle = None if model_conf is None else f"Model: {model_conf:.1f}%"
        st.metric(
            label="📊 Confidence (Blended)",
            value=f"{confidence:.1f}%",
            delta=subtitle
        )
    
    with col3:
        action = final_recommendation.get("action", "HOLD")
        st.metric(
            label="💡 Recommendation",
            value=action,
            delta=None
        )
    
    st.markdown("---")
    
    # Create tabs for detailed analysis
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Technical Analysis", "🎯 Prediction Details", "⚠️ Risk Assessment", "📈 Market Data"])
    
    with tab1:
        display_technical_analysis(technical_summary, ticker)
    
    with tab2:
        display_prediction_details(prediction_summary, evaluation_summary)
    
    with tab3:
        display_risk_assessment(result)
    
    with tab4:
        display_market_data(ticker)

def display_technical_analysis(technical_summary: dict, ticker: str | None = None):
    """Display technical analysis results, with optional trend chart if ticker provided.

    Ensures trend analysis never shows Unknown by deriving from moving averages as fallback.
    """
    df = None
    df_smas = {}
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
        st.subheader("📈 Technical Indicators")
        
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
        st.subheader("🔍 Technical Signals")
        
        # Display trading signals
        trading_signals = technical_summary.get("trading_signals", {})
        if trading_signals:
            overall_recommendation = trading_signals.get("overall_recommendation", "HOLD")
            signal_strength = trading_signals.get("signal_strength", 0)
            total_signals = trading_signals.get("total_signals", 0)
            
            st.write(f"**Overall Recommendation:** {overall_recommendation}")
            st.write(f"**Signal Strength:** {signal_strength}")
            st.write(f"**Total Signals:** {total_signals}")
            
            # Display individual signals
            signals = trading_signals.get("signals", [])
            if signals:
                st.write("**Individual Signals:**")
                for signal in signals[:5]:  # Show first 5 signals
                    if isinstance(signal, dict):
                        signal_type = signal.get("type", "Unknown")
                        indicator = signal.get("indicator", "Unknown")
                        strength = signal.get("strength", "Unknown")
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
                    pattern_name = pattern.get("pattern", "Unknown")
                    signal = pattern.get("signal", "Unknown")
                    st.write(f"- {pattern_name}: {signal}")
                else:
                    st.write(f"- {pattern}")

    # Optional trend chart with MAs
    if ticker and df is not None:
        try:
            st.subheader("📉 Trend Chart (6mo)")
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

def display_prediction_details(prediction_summary: dict, evaluation_summary: dict):
    """Display detailed prediction information."""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Prediction Analysis")
        
        st.write("**Direction:**", prediction_summary.get("direction", "Unknown"))
        st.write("**Confidence:**", f"{prediction_summary.get('confidence', 0):.1f}%")
        
        price_target = prediction_summary.get("price_target")
        if price_target is not None:
            st.write("**Price Target:**", f"${price_target:.2f}")
        else:
            st.write("**Price Target:**", "N/A")
        
        st.subheader("🧠 Reasoning")
        reasoning = prediction_summary.get("reasoning", "No reasoning provided")
        st.write(reasoning)
        
        st.subheader("🔑 Key Factors")
        key_factors = prediction_summary.get("key_factors", [])
        if key_factors:
            for factor in key_factors:
                st.write(f"• {factor}")
        else:
            st.write("No key factors identified")
    
    with col2:
        st.subheader("📊 Evaluation Results")
        
        overall_score = evaluation_summary.get("overall_score", 0)
        st.metric("Overall Score", f"{overall_score:.1f}/100")
        
        prediction_quality = evaluation_summary.get("prediction_quality", {})
        if prediction_quality:
            st.write("**Prediction Quality:**")
            st.write(f"- Score: {prediction_quality.get('score', 0):.1f}/100")
            st.write(f"- Confidence: {prediction_quality.get('confidence_adequacy', 'Unknown')}")
            st.write(f"- Reasoning: {prediction_quality.get('reasoning_quality', 'Unknown')}")
        
        # Display sentiment integration if available
        if "sentiment_integration" in evaluation_summary:
            sentiment_integration = evaluation_summary.get("sentiment_integration", {})
            st.write("**Sentiment Integration:**")
            integrated_analysis = sentiment_integration.get("integrated_analysis", {})
            if integrated_analysis:
                st.write(f"- Integrated Score: {integrated_analysis.get('integrated_score', 0):.1f}/100")
                st.write(f"- Technical Contribution: {integrated_analysis.get('technical_contribution', 0):.1f}")
                st.write(f"- Sentiment Contribution: {integrated_analysis.get('sentiment_contribution', 0):.1f}")

def display_risk_assessment(result: dict):
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
        warnings = []
        
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
        next_steps = []
        
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

def display_market_data(ticker: str):
    """Display current market data for the ticker."""
    
    try:
        # Fetch current market data
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="5d")
        
        if hist.empty:
            st.error("Unable to fetch market data")
            return
        
        latest = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) > 1 else latest
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Current Price",
                f"${latest['Close']:.2f}",
                f"{latest['Close'] - prev['Close']:.2f}"
            )
        
        with col2:
            change_pct = ((latest['Close'] - prev['Close']) / prev['Close']) * 100
            st.metric(
                "Daily Change",
                f"{change_pct:.2f}%",
                f"{latest['Close'] - prev['Close']:.2f}"
            )
        
        with col3:
            st.metric("Volume", f"{latest['Volume']:,}")
        
        with col4:
            st.metric("Market Cap", f"${info.get('marketCap', 0):,.0f}")
        
        # Create price chart
        st.subheader("📈 Price Chart (Last 5 Days)")
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            name='OHLC'
        ))
        
        fig.update_layout(
            title=f"{ticker} Stock Price",
            yaxis_title="Price ($)",
            xaxis_title="Date",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error fetching market data: {str(e)}")

def generate_chatbot_response(prompt: str) -> str:
    """
    Generate chatbot response based on user input.
    """
    prompt_lower = prompt.lower()
    
    # Stock analysis requests
    if any(word in prompt_lower for word in ["analyze", "stock", "prediction", "price"]):
        ticker_match = None
        for word in prompt.split():
            if len(word) <= 5 and word.isupper():
                ticker_match = word
                break
        
        if ticker_match:
            try:
                result = run_prediction(ticker_match, "1d")
                return format_stock_analysis_response(ticker_match, result)
            except Exception as e:
                return f"❌ Sorry, I couldn't analyze {ticker_match}. Error: {str(e)}"
        else:
            return "📈 I can help you analyze stocks! Please specify a ticker symbol (e.g., 'Analyze AAPL' or 'What's the prediction for MSFT?')"
    
    # Workflow questions
    elif any(word in prompt_lower for word in ["workflow", "agents", "process", "how does it work"]):
        return """🤖 **LangGraph Workflow Overview:**

Our stock prediction system uses 8 specialized AI agents:

1. **🎯 Orchestrator** - Initializes and coordinates the process
2. **📈 Data Collector** - Fetches stock data, company info, market data
3. **🔍 Technical Analyzer** - Performs technical analysis
4. **📰 Sentiment Analyzer** - Analyzes news and social media sentiment
5. **🔗 Sentiment Integrator** - Combines technical and sentiment analysis
6. **🤖 Prediction Agent** - Makes final predictions using LLMs
7. **📊 Evaluator Optimizer** - Evaluates prediction quality
8. **✅ Elicitation** - Final confirmation and summary

**Flow:** ENTRY → Orchestrator → Data Collector → Technical Analyzer → Sentiment Analyzer → Sentiment Integrator → Prediction Agent → Evaluator Optimizer → Elicitation → EXIT

Each agent can exit early if there's an error, ensuring robust error handling."""
    
    # General help
    elif any(word in prompt_lower for word in ["help", "what can you do", "capabilities"]):
        return """🤖 **I can help you with:**

📊 **Stock Analysis:**
- Analyze any stock ticker (e.g., "Analyze AAPL")
- Get predictions and technical analysis
- View sentiment analysis and risk assessment

🤖 **Workflow Information:**
- Explain how the LangGraph workflow works
- Describe each AI agent's role
- Show the complete prediction pipeline

📈 **Market Data:**
- Get real-time stock data
- View technical indicators
- Access company information

💡 **Examples:**
- "Analyze TSLA stock"
- "How does the workflow work?"
- "What's the prediction for GOOGL?"
- "Explain the technical analysis process"

Just ask me anything about stocks, predictions, or the AI workflow!"""
    
    # Default response
    else:
        return """🤖 Hi! I'm your AI assistant for stock predictions and analysis.

I can help you with:
- 📊 Stock analysis and predictions
- 🤖 Workflow explanations
- 📈 Market data and technical analysis
- 💡 General questions about the system

Try asking me to "Analyze AAPL" or "How does the workflow work?" to get started!"""

def format_stock_analysis_response(ticker: str, result: dict) -> str:
    """
    Format stock analysis results for chatbot response.
    """
    try:
        if "final_summary" in result:
            final_summary = result.get("final_summary", {})
            prediction_summary = final_summary.get("prediction_summary", {})
            technical_summary = final_summary.get("technical_summary", {})
            final_recommendation = final_summary.get("final_recommendation", {})
            
            response = f"""📊 **{ticker} Analysis Results:**

🎯 **Prediction:** {prediction_summary.get("direction", "Unknown").upper()}
📊 **Confidence:** {prediction_summary.get("confidence", 0):.1f}%
💡 **Recommendation:** {final_recommendation.get("action", "HOLD")}

📈 **Technical Analysis:**
- Technical Score: {technical_summary.get("technical_score", 0):.1f}/100
- Technical Signals: {', '.join(technical_summary.get("technical_signals", []))}

⚠️ **Risk Assessment:**
- Risk Level: {prediction_summary.get("risk_assessment", {}).get("overall_risk_level", "Unknown")}

The analysis was performed using our 8-agent LangGraph workflow, combining technical analysis, sentiment analysis, and AI predictions."""
            
            return response
        else:
            return f"❌ Sorry, I couldn't get complete analysis results for {ticker}. Please try again."
    
    except Exception as e:
        return f"❌ Error formatting results for {ticker}: {str(e)}"

if __name__ == "__main__":
    main()
