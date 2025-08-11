from __future__ import annotations

from langgraph_flow import run_prediction


def generate_chatbot_response(prompt: str) -> str:
    """Generate chatbot response based on user input."""
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
        return (
            "🤖 **LangGraph Workflow Overview:**\n\n"
            "Our stock prediction system uses 8 specialized AI agents:\n\n"
            "1. **🎯 Orchestrator** - Initializes and coordinates the process\n"
            "2. **📈 Data Collector** - Fetches stock data, company info, market data\n"
            "3. **🔍 Technical Analyzer** - Performs technical analysis\n"
            "4. **📰 Sentiment Analyzer** - Analyzes news and social media sentiment\n"
            "5. **🔗 Sentiment Integrator** - Combines technical and sentiment analysis\n"
            "6. **🤖 Prediction Agent** - Makes final predictions using LLMs\n"
            "7. **📊 Evaluator Optimizer** - Evaluates prediction quality\n"
            "8. **✅ Elicitation** - Final confirmation and summary\n\n"
            "**Flow:** ENTRY → Orchestrator → Data Collector → Technical Analyzer → Sentiment Analyzer → Sentiment Integrator → Prediction Agent → Evaluator Optimizer → Elicitation → EXIT\n\n"
            "Each agent can exit early if there's an error, ensuring robust error handling."
        )

    # General help
    elif any(word in prompt_lower for word in ["help", "what can you do", "capabilities"]):
        return (
            "🤖 **I can help you with:**\n\n"
            "📊 **Stock Analysis:**\n"
            "- Analyze any stock ticker (e.g., \"Analyze AAPL\")\n"
            "- Get predictions and technical analysis\n"
            "- View sentiment analysis and risk assessment\n\n"
            "🤖 **Workflow Information:**\n"
            "- Explain how the LangGraph workflow works\n"
            "- Describe each AI agent's role\n"
            "- Show the complete prediction pipeline\n\n"
            "📈 **Market Data:**\n"
            "- Get real-time stock data\n"
            "- View technical indicators\n"
            "- Access company information\n\n"
            "💡 **Examples:**\n"
            "- \"Analyze TSLA stock\"\n"
            "- \"How does the workflow work?\"\n"
            "- \"What's the prediction for GOOGL?\"\n"
            "- \"Explain the technical analysis process\"\n\n"
            "Just ask me anything about stocks, predictions, or the AI workflow!"
        )

    # Default response
    else:
        return (
            "🤖 Hi! I'm your AI assistant for stock predictions and analysis.\n\n"
            "I can help you with:\n"
            "- 📊 Stock analysis and predictions\n"
            "- 🤖 Workflow explanations\n"
            "- 📈 Market data and technical analysis\n"
            "- 💡 General questions about the system\n\n"
            "Try asking me to \"Analyze AAPL\" or \"How does the workflow work?\" to get started!"
        )


def format_stock_analysis_response(ticker: str, result: dict) -> str:
    """Format stock analysis results for chatbot response."""
    try:
        if "final_summary" in result:
            final_summary = result.get("final_summary", {})
            prediction_summary = final_summary.get("prediction_summary", {})
            technical_summary = final_summary.get("technical_summary", {})
            final_recommendation = final_summary.get("final_recommendation", {})

            response = (
                f"📊 **{ticker} Analysis Results:**\n\n"
                f"🎯 **Prediction:** {prediction_summary.get('direction', 'Unknown').upper()}\n"
                f"📊 **Confidence:** {prediction_summary.get('confidence', 0):.1f}%\n"
                f"💡 **Recommendation:** {final_recommendation.get('action', 'HOLD')}\n\n"
                f"📈 **Technical Analysis:**\n"
                f"- Technical Score: {technical_summary.get('technical_score', 0):.1f}/100\n"
                f"- Technical Signals: {', '.join(technical_summary.get('technical_signals', []))}\n\n"
                f"⚠️ **Risk Assessment:**\n"
                f"- Risk Level: {prediction_summary.get('risk_assessment', {}).get('overall_risk_level', 'Unknown')}\n\n"
                "The analysis was performed using our 8-agent LangGraph workflow, combining technical analysis, sentiment analysis, and AI predictions."
            )

            return response
        else:
            return f"❌ Sorry, I couldn't get complete analysis results for {ticker}. Please try again."

    except Exception as e:
        return f"❌ Error formatting results for {ticker}: {str(e)}"


