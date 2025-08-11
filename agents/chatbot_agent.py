# agents/chatbot_agent.py
from typing import Dict, Any, Optional
import re
from .tavily_search_agent import get_web_search_results

class ChatbotAgent:
    """
    Chatbot agent that can handle user queries and integrate with the LangGraph workflow.
    This agent acts as a natural language interface to the stock prediction system.
    """
    
    def __init__(self):
        self.capabilities = {
            "stock_analysis": "Analyze stock tickers and get predictions",
            "workflow_info": "Explain the LangGraph workflow and agents",
            "general_help": "Provide help and guidance",
            "system_info": "Explain system capabilities and features",
            "web_search": "Search the web for real-time information"
        }
    
    def process_user_query(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process user input and determine the appropriate response.
        
        Args:
            user_input: The user's query
            context: Additional context (e.g., previous stock analysis results)
            
        Returns:
            Dictionary with response type and content
        """
        user_input_lower = user_input.lower()
        
        # Stock analysis requests
        if self._is_stock_analysis_request(user_input_lower):
            ticker = self._extract_ticker(user_input)
            if ticker:
                return {
                    "response_type": "stock_analysis",
                    "ticker": ticker,
                    "user_query": user_input,
                    "requires_workflow": True,
                    "message": f"Analyzing {ticker} stock using the LangGraph workflow..."
                }
            else:
                return {
                    "response_type": "error",
                    "message": "Please specify a stock ticker symbol (e.g., 'Analyze AAPL' or 'What's the prediction for MSFT?')",
                    "suggestions": ["Try: 'Analyze AAPL'", "Try: 'What's the prediction for TSLA?'", "Try: 'Stock analysis for GOOGL'"]
                }
        
        # Web search requests
        elif self._is_web_search_request(user_input_lower):
            return {
                "response_type": "web_search",
                "user_query": user_input,
                "requires_workflow": False,
                "message": "Searching the web for information..."
            }
        
        # Workflow information requests
        elif self._is_workflow_info_request(user_input_lower):
            return {
                "response_type": "workflow_info",
                "message": self._get_workflow_explanation(),
                "requires_workflow": False
            }
        
        # General help requests
        elif self._is_help_request(user_input_lower):
            return {
                "response_type": "help",
                "message": self._get_help_message(),
                "requires_workflow": False
            }
        
        # Default response
        else:
            return {
                "response_type": "greeting",
                "message": self._get_greeting_message(),
                "requires_workflow": False
            }
    
    def _is_stock_analysis_request(self, text: str) -> bool:
        """Check if the user is requesting stock analysis."""
        stock_keywords = ["analyze", "stock", "prediction", "price", "forecast", "analysis"]
        return any(keyword in text for keyword in stock_keywords)
    
    def _is_web_search_request(self, text: str) -> bool:
        """Check if the user is requesting web search."""
        search_keywords = ["search", "find", "look up", "what is", "tell me about", "latest news", "recent"]
        return any(keyword in text for keyword in search_keywords)
    
    def _is_workflow_info_request(self, text: str) -> bool:
        """Check if the user is asking about the workflow."""
        workflow_keywords = ["workflow", "agents", "process", "how does it work", "system", "pipeline"]
        return any(keyword in text for keyword in workflow_keywords)
    
    def _is_help_request(self, text: str) -> bool:
        """Check if the user is asking for help."""
        help_keywords = ["help", "what can you do", "capabilities", "examples", "guide"]
        return any(keyword in text for keyword in help_keywords)
    
    def _extract_ticker(self, text: str) -> Optional[str]:
        """Extract stock ticker from user input."""
        # Look for uppercase words that could be tickers (1-5 characters)
        words = text.split()
        for word in words:
            # Remove common punctuation
            clean_word = re.sub(r'[^\w]', '', word)
            if len(clean_word) <= 5 and clean_word.isupper():
                return clean_word
        return None
    
    def _get_workflow_explanation(self) -> str:
        """Get detailed workflow explanation."""
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
    
    def _get_help_message(self) -> str:
        """Get help message with capabilities."""
        return """🤖 **I can help you with:**

📊 **Stock Analysis:**
- Analyze any stock ticker (e.g., "Analyze AAPL")
- Get predictions and technical analysis
- View sentiment analysis and risk assessment

🔍 **Web Search:**
- Search for real-time information
- Find latest news and trends
- Look up company information

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
- "Search for latest Apple news"
- "How does the workflow work?"
- "What's the prediction for GOOGL?"
- "Tell me about NVIDIA company"

Just ask me anything about stocks, predictions, or the AI workflow!"""
    
    def _get_greeting_message(self) -> str:
        """Get default greeting message."""
        return """🤖 Hi! I'm your AI assistant for stock predictions and analysis.

I can help you with:
- 📊 Stock analysis and predictions
- 🔍 Web search for real-time information
- 🤖 Workflow explanations
- 📈 Market data and technical analysis
- 💡 General questions about the system

Try asking me to "Analyze AAPL", "Search for Tesla news", or "How does the workflow work?" to get started!"""
    
    def format_stock_analysis_response(self, ticker: str, analysis_result: Dict[str, Any]) -> str:
        """
        Format stock analysis results for chatbot response.
        
        Args:
            ticker: Stock ticker symbol
            analysis_result: Results from the LangGraph workflow
            
        Returns:
            Formatted response string
        """
        try:
            if "final_summary" in analysis_result:
                final_summary = analysis_result.get("final_summary", {})
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
    
    def format_web_search_response(self, query: str, search_results: Dict[str, Any]) -> str:
        """
        Format web search results for chatbot response.
        
        Args:
            query: Original search query
            search_results: Results from Tavily search
            
        Returns:
            Formatted response string
        """
        try:
            if search_results.get("status") == "success" and search_results.get("results"):
                results = search_results.get("results", [])
                response_time = search_results.get("response_time", 0)
                
                response = f"""🔍 **Web Search Results for: "{query}"**

📊 **Found {len(results)} results** (Response time: {response_time:.2f}s)

"""
                
                # Add top 3 results
                for i, result in enumerate(results[:3], 1):
                    title = result.get("title", "No title")
                    content = result.get("content", "No content")
                    url = result.get("url", "No URL")
                    
                    # Truncate content for readability
                    content_preview = content[:300] + "..." if len(content) > 300 else content
                    
                    response += f"""**{i}. {title}**
{content_preview}
📎 Source: {url}

"""
                
                response += "💡 *These results are from real-time web search using Tavily Search API.*"
                return response
            else:
                error_msg = search_results.get("error", "Unknown error")
                return f"❌ Sorry, I couldn't find information for '{query}'. Error: {error_msg}"
        
        except Exception as e:
            return f"❌ Error formatting search results: {str(e)}"

# LangGraph node function
def chatbot_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function for chatbot processing.
    
    Args:
        state: Current state containing user input and context
        
    Returns:
        Updated state with chatbot response
    """
    try:
        # Extract user input from state
        user_input = state.get("user_query", "")
        
        # Initialize chatbot agent
        chatbot = ChatbotAgent()
        
        # Process the user query
        response = chatbot.process_user_query(user_input, state)
        
        # Handle web search requests
        if response.get("response_type") == "web_search":
            try:
                # Perform web search
                search_results = get_web_search_results(user_input)
                formatted_response = chatbot.format_web_search_response(user_input, search_results)
                response["message"] = formatted_response
                response["web_search_results"] = search_results
            except Exception as e:
                response["message"] = f"❌ Web search failed: {str(e)}"
        
        # Update state with chatbot response
        state.update({
            "chatbot_response": response,
            "status": "success",
            "next_agent": "workflow_orchestrator" if response.get("requires_workflow") else "end"
        })
        
        # If stock analysis is requested, add ticker to state
        if response.get("response_type") == "stock_analysis":
            state["ticker"] = response.get("ticker")
            state["timeframe"] = "1d"  # Default timeframe
        
        return state
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Chatbot processing failed: {str(e)}",
            "next_agent": "end"
        } 