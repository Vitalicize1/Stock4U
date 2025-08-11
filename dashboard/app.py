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
        page_icon="📈",
        layout="wide"
    )

    st.title("📈 Stock4U")
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
            use_ml_model = st.toggle("Use ML model (traditional) for prediction", value=False, help="Run a lightweight ML model using engineered features instead of LLM.")
            submitted = st.form_submit_button("🚀 Run Prediction")

        if submitted:
            ticker = (ticker_input or "").upper().strip()
            if ticker:
                with st.spinner(f"Analyzing {ticker}..."):
                    try:
                        result = run_prediction(ticker, timeframe, low_api_mode=low_api_mode, fast_ta_mode=fast_ta_mode, use_ml_model=use_ml_model)
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
                    result = run_prediction(selected_quick, "1d", low_api_mode=low_api_mode, fast_ta_mode=fast_ta_mode, use_ml_model=use_ml_model)

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

        # --- Jira Issue Filing ---
        st.sidebar.markdown("---")
        st.sidebar.header("📝 File an Issue (Jira)")
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
            issue_labels = st.text_input("Labels (comma separated)", placeholder="bug,ui,high-priority")
            submit_issue = st.form_submit_button("Create Jira Issue")

        if submit_issue:
            try:
                from utils.jira import safe_create_issue
                labels = [s.strip() for s in (issue_labels or "").split(",") if s.strip()]
                res = safe_create_issue(issue_summary or "Untitled Issue", issue_description or "", labels=labels)
                if res.get("status") == "success":
                    data = res.get("data", {})
                    key = data.get("key") or data.get("id")
                    st.sidebar.success(f"Created issue: {key}")
                else:
                    st.sidebar.error(f"Failed to create issue: {res.get('error')}")
            except Exception as e:
                st.sidebar.error(f"Jira integration error: {str(e)}")

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
                from dashboard.components.market_data import display_market_data

                display_market_data(market_ticker)
            except Exception as e:
                st.error(f"Error fetching market data: {str(e)}")


# Import placed at end to avoid circular at app import time
from dashboard.chatbot.utils import format_stock_analysis_response


