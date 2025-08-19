from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from langgraph_flow import run_prediction, run_chatbot_workflow
from utils.validation import InputValidator, ValidationResult

from dashboard.views.results import display_results
from dashboard.auth import show_login_page, show_logout_button, show_user_info, init_auth
from dashboard.components.market_data import display_market_data


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
        /* Make placeholder text smaller in Jira section */
        .stTextInput input::placeholder, .stTextArea textarea::placeholder {
          font-size: 0.75rem !important;
          opacity: 0.6 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Initialize authentication
    auth = init_auth()
    
    # Check if user is authenticated
    if not auth.is_authenticated():
        show_login_page()
        return

    st.title("Stock4U")
    st.markdown("---")

    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Predictions", "Daily Picks & Analysis", "Chatbot", "Market Data"])

    # Show user info and logout button in sidebar
    show_user_info()
    show_logout_button()

    with tab1:
        st.header("Stock Predictions")
        st.markdown("Analyze any stock with our AI-powered prediction system.")
        
        # Initialize session state
        if "has_prediction_results" not in st.session_state:
            st.session_state["has_prediction_results"] = False
        
        # Create two columns for the prediction form
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Main prediction form
            with st.form("prediction_form", clear_on_submit=False):
                st.subheader("Stock Analysis")
                
                # Ticker input with better styling
                ticker_input = st.text_input(
                    "Stock Ticker Symbol",
                    value="AAPL",
                    placeholder="e.g., AAPL, MSFT, GOOGL, TSLA",
                    help="Enter the stock ticker symbol you want to analyze"
                )
                
                # Timeframe selection
                timeframe = st.selectbox(
                    "Prediction Timeframe",
                    options=["1d", "1w", "1m"],
                    index=0,
                    help="Select how far into the future to predict"
                )
                
                # Analysis options in an expander
                with st.expander("Analysis Options", expanded=False):
                    st.markdown("**Choose your analysis mode:**")
                    
                    low_api_mode = st.toggle(
                        "Low API Mode", 
                        value=False, 
                        help="Use rule-based prediction to save API quota (no LLM calls)"
                    )
                    
                    fast_ta_mode = st.toggle(
                        "Fast Technical Analysis", 
                        value=False, 
                        help="Run minimal technical analysis for faster results"
                    )
                    
                    use_ml_model = st.toggle(
                        "Use ML Model", 
                        value=False, 
                        help="Use traditional machine learning instead of LLM"
                    )
                
                # Submit button
                submitted = st.form_submit_button(
                    "Run Prediction",
                    help="Start the AI analysis process"
                )
                
                # Validate inputs before processing
                if submitted:
                    # Validate ticker
                    ticker_validation = InputValidator.validate_ticker_symbol(ticker_input)
                    if not ticker_validation.is_valid:
                        st.error(f"❌ Invalid ticker: {ticker_validation.error_message}")
                        submitted = False
                    
                    # Validate timeframe
                    timeframe_validation = InputValidator.validate_timeframe(timeframe)
                    if not timeframe_validation.is_valid:
                        st.error(f"❌ Invalid timeframe: {timeframe_validation.error_message}")
                        submitted = False
                    
                    # Show warnings if any
                    if ticker_validation.warnings:
                        for warning in ticker_validation.warnings:
                            st.warning(f"⚠️ {warning}")
                    
                    if timeframe_validation.warnings:
                        for warning in timeframe_validation.warnings:
                            st.warning(f"⚠️ {warning}")
        
        with col2:
            # Quick analysis section
            st.subheader("Quick Analysis")
            st.markdown("Popular stocks for instant analysis:")
            
            popular_tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA"]
            
            # Create a grid of quick analyze buttons
            for i in range(0, len(popular_tickers), 2):
                cols = st.columns(2)
                for j in range(2):
                    if i + j < len(popular_tickers):
                        ticker = popular_tickers[i + j]
                        with cols[j]:
                            if st.button(f"{ticker}", key=f"quick_{ticker}"):
                                # Validate ticker before processing
                                ticker_validation = InputValidator.validate_ticker_symbol(ticker)
                                if not ticker_validation.is_valid:
                                    st.error(f"❌ Invalid ticker {ticker}: {ticker_validation.error_message}")
                                else:
                                    with st.spinner(f"Quick analysis of {ticker}..."):
                                        try:
                                            result = run_prediction(ticker, "1d", low_api_mode=False, fast_ta_mode=False, use_ml_model=False)
                                            
                                            # Log the prediction
                                            try:
                                                from utils.prediction_logger import log_prediction
                                                prediction_result = result.get("prediction_result", {})
                                                prediction_data = {
                                                    "direction": prediction_result.get("direction"),
                                                    "confidence": prediction_result.get("confidence"),
                                                    "timeframe": "1d",
                                                    "predicted_price": prediction_result.get("price_target"),
                                                    "current_price": result.get("data", {}).get("market_data", {}).get("current_price")
                                                }
                                                log_prediction(ticker, prediction_data)
                                            except Exception as e:
                                                print(f"Warning: Could not log prediction: {e}")
                                            
                                            # Store results
                                            st.session_state["has_prediction_results"] = True
                                            st.session_state["last_result"] = result
                                            
                                            # Build summary
                                            pr = result.get("prediction_result", {})
                                            prediction = pr.get("prediction", pr)
                                            cm = result.get("confidence_metrics") or pr.get("confidence_metrics") or {}
                                            overall_conf = cm.get("overall_confidence")
                                            if overall_conf is None:
                                                overall_conf = prediction.get("confidence")
                                            
                                            st.session_state["last_run_summary"] = {
                                                "ticker": ticker,
                                                "timeframe": "1d",
                                                "direction": prediction.get("direction"),
                                                "confidence": overall_conf,
                                            }
                                            
                                            st.rerun()
                                            
                                        except Exception as e:
                                            st.error(f"Error analyzing {ticker}: {str(e)}")
        
        # Handle form submission
        if submitted:
            ticker = (ticker_input or "").upper().strip()
            if ticker:
                with st.spinner(f"Analyzing {ticker} with AI..."):
                    try:
                        result = run_prediction(ticker, timeframe, low_api_mode=low_api_mode, fast_ta_mode=fast_ta_mode, use_ml_model=use_ml_model)
                        
                        # Log the prediction for accuracy tracking
                        try:
                            from utils.prediction_logger import log_prediction
                            prediction_result = result.get("prediction_result", {})
                            prediction_data = {
                                "direction": prediction_result.get("direction"),
                                "confidence": prediction_result.get("confidence"),
                                "timeframe": timeframe,
                                "predicted_price": prediction_result.get("price_target"),
                                "current_price": result.get("data", {}).get("market_data", {}).get("current_price")
                            }
                            log_prediction(ticker, prediction_data)
                        except Exception as e:
                            print(f"Warning: Could not log prediction: {e}")
                        
                        # Show quota status in sidebar if available
                        if 'quota_status' in result:
                            st.sidebar.markdown("---")
                            st.sidebar.header("LLM Provider Status")
                            quota_status = result['quota_status']
                            for provider, status in quota_status.items():
                                if status["available"]:
                                    st.sidebar.success(f"{provider.upper()}: {status['reason']}")
                                else:
                                    st.sidebar.error(f"{provider.upper()}: {status['reason']}")
                        
                        # Store results
                        st.session_state["has_prediction_results"] = True
                        st.session_state["last_result"] = result
                        
                        # Build summary
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
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error analyzing {ticker}: {str(e)}")
            else:
                st.error("Please enter a valid ticker symbol")

        # --- Report Issues & Feedback ---
        st.sidebar.markdown("---")
        st.sidebar.header("Report Issues & Feedback")
        
        # Help text
        with st.sidebar.expander("How to report issues", expanded=False):
            st.markdown("""
            **Need help? Here's how to report issues:**
            
            🐛 **Bug Reports:** Describe what went wrong
            💡 **Feature Requests:** Suggest improvements
            ❓ **Questions:** Ask for help or clarification
            
            **Tips for better reports:**
            - Be specific about what you were doing
            - Include error messages if any
            - Mention your browser/device if relevant
            """)
        

        
        # Issue form
        with st.sidebar.form("issue_form", clear_on_submit=True):
            st.markdown("**Create a new issue:**")
            
            # Issue type with better descriptions
            issue_type = st.selectbox(
                "Type of Issue",
                ["Bug", "Feature Request", "Question", "Task"],
                index=0,
                help="Select the most appropriate category for your issue"
            )
            
            # Priority with better labels
            priority_options = {
                "Low": "Low - Minor issue or enhancement",
                "Medium": "Medium - Standard priority",
                "High": "High - Important issue affecting functionality",
                "Critical": "Critical - System breaking issue"
            }
            priority = st.selectbox(
                "Priority",
                list(priority_options.keys()),
                index=1,
                help="How urgent is this issue?"
            )
            
            # Summary with better placeholder
            summary_placeholder = {
                "Bug": "e.g., 'Login button not working'",
                "Feature Request": "e.g., 'Add dark mode theme'",
                "Question": "e.g., 'How to export data?'",
                "Task": "e.g., 'Update documentation'"
            }
            issue_summary = st.text_input(
                "Title",
                placeholder=summary_placeholder.get(issue_type, "Brief description of the issue"),
                help="A clear, concise title for your issue"
            )
            
            # Description with placeholder template
            description_placeholder = {
                "Bug": "What happened? What did you expect? Steps to reproduce: 1. 2. 3. Additional details...",
                "Feature Request": "What feature would you like? Why is this useful? Any examples or mockups?",
                "Question": "What's your question? What have you tried? Additional context...",
                "Task": "What needs to be done? Why is this needed? Any specific requirements?"
            }
            issue_description = st.text_area(
                "Description",
                placeholder=description_placeholder.get(issue_type, "Provide detailed information about your issue"),
                height=150,
                help="Provide detailed information about your issue"
            )
            
            # Labels dropdown with predefined options
            label_options = {
                "Bug": ["Bug", "UI", "Backend", "Data", "Performance", "Security"],
                "Feature Request": ["Enhancement", "UI", "Backend", "Data", "Analytics", "Export"],
                "Question": ["Question", "Help", "Documentation", "Tutorial", "Setup"],
                "Task": ["Task", "Documentation", "Testing", "Refactor", "Maintenance"]
            }
            
            # Get available labels for the selected issue type
            available_labels = label_options.get(issue_type, [])
            
            # Always include stock4u as the base label
            base_label = "stock4u"
            
            # Create multi-select for labels
            selected_labels = st.multiselect(
                "Labels",
                options=available_labels,
                default=[],
                help="Select relevant labels to categorize your issue"
            )
            
            # Combine selected labels with base label
            issue_labels = ",".join([base_label] + selected_labels)
            
            # Attach current analysis
            attach_json = st.checkbox(
                "Include current analysis data",
                value=True,
                help="Attach the current stock analysis data to help with debugging"
            )
            
            # Submit button with better styling
            submit_issue = st.form_submit_button(
                "Submit Issue",
                help="Create the issue in our tracking system"
            )

        # Handle form submission
        if submit_issue:
            if not issue_summary.strip():
                st.sidebar.error("Please provide a title for your issue")
            elif not issue_description.strip():
                st.sidebar.error("Please provide a description for your issue")
            else:
                try:
                    from utils.jira import safe_create_issue, safe_attach_file
                    
                    # Show progress
                    with st.sidebar.spinner("Creating issue..."):
                        # Map priority to Jira format
                        priority_mapping = {
                            "Low": "Low",
                            "Medium": "Medium", 
                            "High": "High",
                            "Critical": "Highest"
                        }
                        
                        # Map issue type to Jira format
                        type_mapping = {
                            "Bug": "Bug",
                            "Feature Request": "Story",
                            "Question": "Task",
                            "Task": "Task"
                        }
                        
                        labels = [s.strip() for s in (issue_labels or "").split(",") if s.strip()]
                        extra_fields = {"priority": {"name": priority_mapping.get(priority, "Medium")}}
                        
                        res = safe_create_issue(
                            issue_summary.strip(),
                            issue_description.strip(),
                            issue_type=type_mapping.get(issue_type, "Task"),
                            labels=labels,
                            extra_fields=extra_fields,
                        )
                        
                        if res.get("status") == "success":
                            data = res.get("data", {})
                            key = data.get("key") or data.get("id")
                            
                            # Create issue URL
                            import os
                            base = os.getenv("JIRA_BASE_URL", "").rstrip("/")
                            issue_url = f"{base}/browse/{key}" if base and key else None
                            
                            # Attach analysis data if requested
                            if attach_json and 'last_result' in st.session_state:
                                import json as _json, tempfile
                                try:
                                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{key}.json")
                                    tmp.write(_json.dumps(st.session_state['last_result'], indent=2).encode('utf-8'))
                                    tmp.close()
                                    safe_attach_file(key, tmp.name)
                                except Exception:
                                    pass
                            
                            # Show success message
                            st.sidebar.success("Issue created successfully!")
                            if issue_url:
                                st.sidebar.markdown(f"**Issue:** [{key}]({issue_url})")
                            else:
                                st.sidebar.markdown(f"**Issue ID:** {key}")
                            
                            # Show next steps
                            st.sidebar.info("📧 You'll receive updates via email when the issue is updated.")
                            
                        else:
                            st.sidebar.error(f"Failed to create issue: {res.get('error')}")
                            
                except Exception as e:
                    st.sidebar.error(f"Error creating issue: {str(e)}")
                    st.sidebar.info("Please try again or contact support directly.")



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
            pass

    with tab2:
        st.header("Daily Picks & Analysis")
        st.markdown("Today's best stock recommendations with detailed analysis.")
        
        # Display prediction accuracy summary
        try:
            from utils.prediction_logger import get_accuracy_summary, get_daily_picks_accuracy
            accuracy_summary = get_accuracy_summary(days=30)
            daily_picks_accuracy = get_daily_picks_accuracy(days=30)
            
            # Create columns for accuracy metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Overall Accuracy", f"{accuracy_summary['accuracy']}%")
                st.caption(f"Last {accuracy_summary['days_analyzed']} days")
            
            with col2:
                st.metric("Total Predictions", accuracy_summary['total_predictions'])
                st.caption(f"{accuracy_summary['predictions_with_results']} with results")
            
            with col3:
                st.metric("Daily Picks", daily_picks_accuracy['total_picks'])
                st.caption(f"{daily_picks_accuracy['total_daily_picks_days']} days tracked")
                
        except Exception as e:
            st.info(f"Accuracy tracking unavailable: {e}")
        
        st.markdown("---")
        
        # Load daily picks data
        try:
            from dashboard.components.daily_picks import load_daily_picks
            data = load_daily_picks("cache/daily_picks.json")
            
            if data and data.get("picks"):
                picks = data["picks"]
                
                # Display picks section
                st.subheader("Today's Top Picks")
                
                # Show generation time
                if "generated_at" in data:
                    st.caption(f"Generated: {data['generated_at']}")
                
                # Display each pick with analysis button
                for i, pick in enumerate(picks[:3]):  # Show top 3 picks
                    ticker = pick.get("ticker", "")
                    direction = pick.get("direction", "Unknown")
                    confidence = pick.get("confidence", 0)
                    
                    # Create columns for layout
                    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
                    
                    with col1:
                        st.metric(f"#{i+1} {ticker}", direction)
                    
                    with col2:
                        # Color code the direction with confidence bar for UP
                        if direction.upper() == "UP":
                            # Create a confidence bar that scales with confidence level
                            confidence_width = min(confidence, 100)  # Cap at 100%
                            green_intensity = int(255 * (confidence / 100))  # Scale green intensity
                            bar_color = f"rgb(0, {green_intensity}, 0)"
                            
                            # Display UP with confidence bar
                            st.markdown(f"""
                            <div style="display: flex; flex-direction: column; align-items: center;">
                                <div style="font-weight: bold; margin-bottom: 5px;">UP</div>
                                <div style="width: 100%; height: 8px; background-color: #f0f0f0; border-radius: 4px; overflow: hidden;">
                                    <div style="width: {confidence_width}%; height: 100%; background-color: {bar_color}; transition: width 0.3s ease;"></div>
                                </div>
                                <div style="font-size: 0.8em; color: #666; margin-top: 2px;">{confidence:.1f}% confidence</div>
                            </div>
                            """, unsafe_allow_html=True)
                        elif direction.upper() == "DOWN":
                            st.markdown("**DOWN**")
                        else:
                            st.markdown("**NEUTRAL**")
                    
                    with col3:
                        st.metric("Confidence", f"{confidence:.1f}%")
                    
                    with col4:
                        # Create a button for detailed analysis
                        if st.button(f"Analyze {ticker}", key=f"analyze_{ticker}"):
                            # Run analysis for this ticker
                            with st.spinner(f"Analyzing {ticker}..."):
                                try:
                                    result = run_prediction(ticker, "1d", low_api_mode=False, fast_ta_mode=False)
                                    
                                    # Log the prediction for accuracy tracking
                                    try:
                                        from utils.prediction_logger import log_prediction
                                        prediction_result = result.get("prediction_result", {})
                                        prediction_data = {
                                            "direction": prediction_result.get("direction"),
                                            "confidence": prediction_result.get("confidence"),
                                            "timeframe": "1d",
                                            "predicted_price": prediction_result.get("price_target"),
                                            "current_price": result.get("data", {}).get("market_data", {}).get("current_price")
                                        }
                                        log_prediction(ticker, prediction_data)
                                    except Exception as e:
                                        st.warning(f"Could not log prediction: {e}")
                                    
                                    st.session_state["has_prediction_results"] = True
                                    st.session_state["last_result"] = result
                                    
                                    # Display the analysis
                                    st.markdown("---")
                                    st.subheader(f"📋 Detailed Analysis for {ticker}")
                                    display_results(ticker, result)
                                    
                                except Exception as e:
                                    st.error(f"Error analyzing {ticker}: {str(e)}")
                    
                    # Add a small gap between picks
                    st.markdown("---")
                
                # Refresh button
                col1, col2 = st.columns([1, 3])
                with col1:
                    if st.button("🔄 Refresh Picks"):
                        st.info("Daily picks are refreshed automatically via scheduled job. Check back in a few minutes!")
                with col2:
                    st.markdown("")
                    
            else:
                st.info("No daily picks available. Click 'Refresh Picks' to generate new recommendations.")
                
        except Exception as e:
            st.error(f"Error loading daily picks: {e}")

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
                display_market_data(market_ticker)
            except Exception as e:
                st.error(f"Error fetching market data: {str(e)}")





    


# Import placed at end to avoid circular at app import time
from dashboard.chatbot.utils import format_stock_analysis_response


