#!/usr/bin/env python3
"""
💬 Standalone Chatbot Interface
Interactive chatbot for the LangGraph stock prediction workflow.
"""

import streamlit as st
from langgraph_flow import run_prediction
import json

def main():
    st.set_page_config(
        page_title="🤖 Stock Prediction Chatbot",
        page_icon="💬",
        layout="wide"
    )
    
    st.title("🤖 AI Stock Prediction Chatbot")
    st.markdown("---")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar for controls
    st.sidebar.header("🤖 Chatbot Controls")
    
    if st.sidebar.button("🔄 Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    if st.sidebar.button("📊 Quick AAPL Analysis"):
        quick_analysis_prompt = "Can you analyze AAPL stock for me?"
        st.session_state.messages.append({"role": "user", "content": quick_analysis_prompt})
        with st.chat_message("user"):
            st.markdown(quick_analysis_prompt)
        
        with st.chat_message("assistant"):
            response = generate_chatbot_response(quick_analysis_prompt)
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    if st.sidebar.button("🤖 Workflow Info"):
        workflow_prompt = "How does the workflow work?"
        st.session_state.messages.append({"role": "user", "content": workflow_prompt})
        with st.chat_message("user"):
            st.markdown(workflow_prompt)
        
        with st.chat_message("assistant"):
            response = generate_chatbot_response(workflow_prompt)
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    if st.sidebar.button("💡 Help"):
        help_prompt = "What can you do?"
        st.session_state.messages.append({"role": "user", "content": help_prompt})
        with st.chat_message("user"):
            st.markdown(help_prompt)
        
        with st.chat_message("assistant"):
            response = generate_chatbot_response(help_prompt)
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
    
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
        
        # Generate assistant response
        with st.chat_message("assistant"):
            response = generate_chatbot_response(prompt)
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Footer with examples
    st.markdown("---")
    st.markdown("**💡 Example questions:**")
    st.markdown("- 'Analyze TSLA stock'")
    st.markdown("- 'How does the workflow work?'")
    st.markdown("- 'What's the prediction for GOOGL?'")
    st.markdown("- 'Explain the technical analysis process'")

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
                with st.spinner(f"🤖 Analyzing {ticker_match} using LangGraph workflow..."):
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