#!/usr/bin/env python3
"""
🧪 Chatbot Test Script
Test the chatbot functionality without running the full Streamlit interface.
"""

from langgraph_flow import run_prediction

def test_chatbot_response(prompt: str) -> str:
    """
    Test the chatbot response generation.
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
                print(f"🤖 Analyzing {ticker_match} using LangGraph workflow...")
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

def main():
    """Test the chatbot with various inputs."""
    print("🧪 Testing Chatbot Functionality")
    print("=" * 50)
    
    test_cases = [
        "Analyze AAPL stock",
        "How does the workflow work?",
        "What can you do?",
        "Hello",
        "Analyze TSLA",
        "Explain the process"
    ]
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: '{test_input}'")
        print("-" * 30)
        response = test_chatbot_response(test_input)
        print(f"🤖 Response: {response}")
        print("-" * 30)
    
    print("\n✅ Chatbot testing completed!")

if __name__ == "__main__":
    main() 