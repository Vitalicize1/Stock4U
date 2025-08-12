from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from langgraph_flow import run_prediction, run_chatbot_workflow

from dashboard.views.results import display_results


def main() -> None:
    st.set_page_config(
        page_title="Stock4U",
        layout="wide"
    )

    # Global typography: Inter for body, Manrope for headings
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@600;700&display=swap');
        :root {
          --app-body-font: 'Inter', ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, Arial, 'Noto Sans', 'Liberation Sans', sans-serif;
          --app-heading-font: 'Manrope', 'Inter', ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, Arial, sans-serif;
        }
        html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
          font-family: var(--app-body-font);
          font-variant-numeric: tabular-nums;
        }
        h1, h2, h3, h4, h5, h6 { 
          font-family: var(--app-heading-font);
          font-weight: 700;
          letter-spacing: 0.1px;
        }
        .stButton button, .stDownloadButton button, .st-emotion-cache button, .stTextInput input, .stTextArea textarea, .stSelectbox div {
          font-family: var(--app-body-font);
        }
        [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {
          font-variant-numeric: tabular-nums;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Stock4U")
    st.markdown("---")

    # Create tabs for different sections (remove workflow tab for end-users)
    tab1, tab3, tab4, tab5 = st.tabs(["Predictions", "Chatbot", "Market Data", "Alerts"])

    with tab1:
        # Sidebar form so pressing Enter submits and runs prediction
        st.sidebar.header("Prediction Settings")
        if "has_prediction_results" not in st.session_state:
            st.session_state["has_prediction_results"] = False
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
            use_ml_model = st.toggle("Use ML model (traditional) for prediction", value=False, help="Run a lightweight ML model using engineered features instead of LLM.")
            submitted = st.form_submit_button("Run Prediction")

        if submitted:
            ticker = (ticker_input or "").upper().strip()
            if ticker:
                with st.spinner(f"Analyzing {ticker}..."):
                    try:
                        result = run_prediction(ticker, timeframe, low_api_mode=low_api_mode, fast_ta_mode=fast_ta_mode, use_ml_model=use_ml_model)
                        if 'quota_status' in result:
                            st.sidebar.markdown("---")
                            st.sidebar.header("LLM Provider Status")
                            quota_status = result['quota_status']
                            for provider, status in quota_status.items():
                                if status["available"]:
                                    st.sidebar.success(f"{provider.upper()}: {status['reason']}")
                                else:
                                    st.sidebar.error(f"{provider.upper()}: {status['reason']}")
                        # Store for rendering after sidebar so we can show a banner above results
                        st.session_state["has_prediction_results"] = True
                        st.session_state["last_result"] = result
                        # Build last run summary
                        pr = result.get("prediction_result", {})
                        prediction = pr.get("prediction", pr)
                        cm = result.get("confidence_metrics") or pr.get("confidence_metrics") or {}
                        overall_conf = cm.get("overall_confidence")
                        if overall_conf is None:
                            overall_conf = prediction.get("confidence")
                        st.session_state["last_run_summary"] = {
                            "ticker": ticker,
                            "timeframe": timeframe,
                            "direction": prediction.get("direction"),
                            "confidence": overall_conf,
                        }
                    except Exception as e:
                        st.error(f"Error analyzing {ticker}: {str(e)}")
            else:
                st.error("Please enter a valid ticker symbol")

        # Quick analysis section
        st.sidebar.markdown("---")
        st.sidebar.header("Quick Analysis")

        popular_tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA"]
        selected_quick = st.sidebar.selectbox("Popular Stocks", popular_tickers)

        if st.sidebar.button("Quick Analyze"):
            with st.spinner(f"Quick analysis of {selected_quick}..."):
                try:
                    result = run_prediction(selected_quick, "1d", low_api_mode=low_api_mode, fast_ta_mode=fast_ta_mode, use_ml_model=use_ml_model)

                    # Display quota status if available
                    if 'quota_status' in result:
                        st.sidebar.markdown("---")
                        st.sidebar.header("LLM Provider Status")

                        quota_status = result['quota_status']
                        for provider, status in quota_status.items():
                            if status["available"]:
                                st.sidebar.success(f"{provider.upper()}: {status['reason']}")
                            else:
                                st.sidebar.error(f"{provider.upper()}: {status['reason']}")

                    st.session_state["has_prediction_results"] = True
                    st.session_state["has_prediction_results"] = True
                    st.session_state["last_result"] = result
                    pr = result.get("prediction_result", {})
                    prediction = pr.get("prediction", pr)
                    cm = result.get("confidence_metrics") or pr.get("confidence_metrics") or {}
                    overall_conf = cm.get("overall_confidence")
                    if overall_conf is None:
                        overall_conf = prediction.get("confidence")
                    st.session_state["last_run_summary"] = {
                        "ticker": selected_quick,
                        "timeframe": "1d",
                        "direction": prediction.get("direction"),
                        "confidence": overall_conf,
                    }
                except Exception as e:
                    st.error(f"Error in quick analysis: {str(e)}")

        # --- Jira Issue Filing ---
        st.sidebar.markdown("---")
        st.sidebar.header("File an Issue")
        # Connection check button
        if st.sidebar.button("Test Jira Connection"):
            try:
                from utils.jira import safe_test_connection
                conn = safe_test_connection()
                if conn.get("status") == "success":
                    st.sidebar.success(f"Connected. Project: {conn.get('project')}")
                else:
                    st.sidebar.error(f"Connection failed: {conn.get('error')}")
            except Exception as e:
                st.sidebar.error(f"Jira test failed: {str(e)}")
        with st.sidebar.form("jira_issue_form", clear_on_submit=True):
            issue_summary = st.text_input("Summary", placeholder="Short title of the issue")
            issue_description = st.text_area("Description", placeholder="Describe the problem or request")
            issue_type = st.selectbox("Issue Type", ["Task", "Bug", "Story"], index=0)
            priority = st.selectbox("Priority", ["Lowest", "Low", "Medium", "High", "Highest"], index=2)
            default_labels = ["stock4u"]
            issue_labels = st.text_input("Labels (comma separated)", value=",".join(default_labels))
            attach_json = st.checkbox("Attach current analysis JSON (if available)", value=True)
            submit_issue = st.form_submit_button("Create Jira Issue")

        if submit_issue:
            try:
                from utils.jira import safe_create_issue, safe_attach_file
                labels = [s.strip() for s in (issue_labels or "").split(",") if s.strip()]
                # Map priority via extra fields (works for default scheme)
                extra_fields = {"priority": {"name": priority}}
                res = safe_create_issue(
                    issue_summary or "Untitled Issue",
                    issue_description or "",
                    issue_type=issue_type,
                    labels=labels,
                    extra_fields=extra_fields,
                )
                if res.get("status") == "success":
                    data = res.get("data", {})
                    key = data.get("key") or data.get("id")
                    issue_url = None
                    import os
                    base = os.getenv("JIRA_BASE_URL", "").rstrip("/")
                    if base and key:
                        issue_url = f"{base}/browse/{key}"
                    # Optional attachment: try to write normalized JSON to a temp file
                    if attach_json and 'normalized_result' in st.session_state:
                        import json as _json, tempfile
                        try:
                            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{key}.json")
                            tmp.write(_json.dumps(st.session_state['normalized_result'], indent=2).encode('utf-8'))
                            tmp.close()
                            safe_attach_file(key, tmp.name)
                        except Exception:
                            pass
                    if issue_url:
                        st.sidebar.markdown(f"Created issue: [{key}]({issue_url})")
                    else:
                        st.sidebar.success(f"Created issue: {key}")
                else:
                    st.sidebar.error(f"Failed to create issue: {res.get('error')}")
            except Exception as e:
                st.sidebar.error(f"Jira integration error: {str(e)}")

        # Render Daily Top Picks
        st.markdown("---")
        try:
            from dashboard.components.daily_picks import display_daily_picks
            # Allow users to paste a large custom universe; keep defaults if empty.
            with st.expander("Daily Picks Universe (optional)"):
                custom_universe_text = st.text_area(
                    "Tickers (comma or newline separated)",
                    value="",
                    height=80,
                    help="Provide a custom list to scan. If empty, a curated default list is used. For very large lists, a rotating subset is scanned daily.",
                )
                max_scan = st.slider("Max tickers to scan today", min_value=50, max_value=1000, value=200, step=50)
            display_daily_picks(custom_tickers_text=custom_universe_text, max_scan=max_scan, top_n=3)
        except Exception as e:
            st.info(f"Daily picks unavailable: {e}")

        # Render last run summary and results (if present), else placeholder
        if st.session_state.get("has_prediction_results", False) and st.session_state.get("last_result") is not None:
            summary = st.session_state.get("last_run_summary", {})
            tkr = summary.get("ticker", "")
            tf = summary.get("timeframe", "")
            direction = summary.get("direction")
            confidence = summary.get("confidence")
            msg_parts = [f"Ticker: {tkr}", f"Timeframe: {tf}"]
            if direction is not None:
                msg_parts.append(f"Direction: {direction}")
            if isinstance(confidence, (int, float)):
                msg_parts.append(f"Confidence: {float(confidence):.1f}%")
            st.info("Last run — " + " | ".join(msg_parts))
            display_results(tkr, st.session_state.get("last_result"))
        else:
            st.subheader("Predictions")
            st.write(
                "Use the form in the sidebar to run a prediction. Choose a ticker and timeframe, then click Run Prediction."
            )

    with tab3:
        st.header("AI Chatbot Assistant")
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
                    with st.spinner("Processing your request..."):
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
                    error_response = f"Sorry, I encountered an error: {str(e)}"
                    st.markdown(error_response)
                    response = error_response

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

        # Sidebar for chatbot features
        st.sidebar.markdown("---")
        st.sidebar.header("Chatbot Features")

        if st.sidebar.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

        if st.sidebar.button("Quick Stock Analysis"):
            quick_analysis_prompt = "Can you analyze AAPL stock for me?"
            st.session_state.messages.append({"role": "user", "content": quick_analysis_prompt})
            with st.chat_message("user"):
                st.markdown(quick_analysis_prompt)

            with st.chat_message("assistant"):
                try:
                    with st.spinner("Analyzing AAPL..."):
                        result = run_chatbot_workflow(quick_analysis_prompt)
                        if "final_summary" in result:
                            response = format_stock_analysis_response("AAPL", result)
                        else:
                            response = "Sorry, I couldn't complete the analysis."
                    st.markdown(response)
                except Exception as e:
                    error_response = f"Error: {str(e)}"
                    st.markdown(error_response)
                    response = error_response

            st.session_state.messages.append({"role": "assistant", "content": response})

    with tab4:
        st.header("Market Data")
        st.markdown("---")

        # Market data section
        market_ticker = st.text_input("Enter ticker for market data:", value="AAPL")

        if st.button("Get Market Data"):
            try:
                from dashboard.components.market_data import display_market_data

                display_market_data(market_ticker)
            except Exception as e:
                st.error(f"Error fetching market data: {str(e)}")

    with tab5:
        st.header("Monitoring & Alerts")
        st.markdown("---")
        from dashboard.components.alerts import display_alerts
        alerts_file = st.text_input("Alerts file path", value="cache/metrics/alerts.log")
        max_rows = st.slider("Max rows", min_value=20, max_value=500, value=200, step=10)
        # Simple alerts view by default for user-friendliness
        display_alerts(alerts_file, max_rows, simple=True)

        st.markdown("---")
        st.subheader("Accuracy Baseline")
        colb1, colb2, colb3 = st.columns([2,1,1])
        with colb1:
            baseline_tickers = st.text_input("Tickers (comma)", value="AAPL,MSFT,GOOGL,NVDA,AMZN")
        with colb2:
            baseline_period = st.selectbox("Period", ["6mo","1y","2y"], index=1)
        with colb3:
            if st.button("Run Baseline"):
                try:
                    from utils.baseline import BaselineConfig, run_baseline
                    cfg = BaselineConfig(
                        tickers=[t.strip().upper() for t in baseline_tickers.split(",") if t.strip()],
                        period=baseline_period,
                        policies=["agent","rule","sma20"],
                        offline=True,
                    )
                    with st.spinner("Running baseline backtests..."):
                        out = run_baseline(cfg)
                    st.success(f"Baseline generated: {out['json']}")
                    try:
                        import json as _json
                        with open(out['json'], 'r', encoding='utf-8') as f:
                            data = _json.load(f)
                        st.json({k: data[k] for k in ["period","tickers","by_policy"]})
                    except Exception:
                        pass
                except Exception as e:
                    st.error(f"Baseline failed: {e}")



    


# Import placed at end to avoid circular at app import time
from dashboard.chatbot.utils import format_stock_analysis_response


