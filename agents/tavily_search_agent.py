# agents/tavily_search_agent.py
from typing import Dict, Any, List
from langchain_tavily import TavilySearch
import os

class TavilySearchAgent:
    """
    Tavily Search Agent for web search capabilities.
    This agent can search the web for real-time information about stocks, companies, and market data.
    """
    
    def __init__(self, max_results: int = 5):
        """
        Initialize the Tavily search agent.
        
        Args:
            max_results: Maximum number of search results to return
        """
        self.max_results = max_results
        self.search_tool = TavilySearch(max_results=max_results)
        
        # Ensure API key is set
        if not os.getenv("TAVILY_API_KEY"):
            raise ValueError("TAVILY_API_KEY environment variable is required")
    
    def search_stock_info(self, ticker: str) -> Dict[str, Any]:
        """
        Search for stock-specific information.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Search results with stock information
        """
        query = f"{ticker} stock price news analysis financial performance"
        return self._perform_search(query)
    
    def search_company_info(self, company_name: str) -> Dict[str, Any]:
        """
        Search for company information.
        
        Args:
            company_name: Name of the company
            
        Returns:
            Search results with company information
        """
        query = f"{company_name} company financial performance earnings revenue"
        return self._perform_search(query)
    
    def search_market_news(self, topic: str = "stock market") -> Dict[str, Any]:
        """
        Search for market news and trends.
        
        Args:
            topic: Market topic to search for
            
        Returns:
            Search results with market news
        """
        query = f"{topic} latest news trends analysis"
        return self._perform_search(query)
    
    def search_technical_analysis(self, ticker: str) -> Dict[str, Any]:
        """
        Search for technical analysis information.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Search results with technical analysis
        """
        query = f"{ticker} technical analysis chart patterns indicators"
        return self._perform_search(query)
    
    def search_sentiment_analysis(self, ticker: str) -> Dict[str, Any]:
        """
        Search for sentiment analysis and social media sentiment.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Search results with sentiment information
        """
        query = f"{ticker} sentiment analysis social media investor sentiment"
        return self._perform_search(query)
    
    def search_custom_query(self, query: str) -> Dict[str, Any]:
        """
        Perform a custom search query.
        
        Args:
            query: Custom search query
            
        Returns:
            Search results
        """
        return self._perform_search(query)
    
    def _perform_search(self, query: str) -> Dict[str, Any]:
        """
        Perform the actual search using Tavily.
        
        Args:
            query: Search query
            
        Returns:
            Search results
        """
        try:
            results = self.search_tool.invoke(query)
            return {
                "status": "success",
                "query": query,
                "results": results.get("results", []),
                "response_time": results.get("response_time", 0),
                "total_results": len(results.get("results", [])),
                "summary": self._summarize_results(results.get("results", []))
            }
        except Exception as e:
            return {
                "status": "error",
                "query": query,
                "error": str(e),
                "results": [],
                "total_results": 0
            }
    
    def _summarize_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Create a summary of search results.
        
        Args:
            results: List of search results
            
        Returns:
            Summary string
        """
        if not results:
            return "No search results found."
        
        summary_parts = []
        for i, result in enumerate(results[:3], 1):  # Top 3 results
            title = result.get("title", "No title")
            content = result.get("content", "No content")
            url = result.get("url", "No URL")
            
            # Truncate content for summary
            content_preview = content[:200] + "..." if len(content) > 200 else content
            
            summary_parts.append(f"{i}. **{title}**\n   {content_preview}\n   Source: {url}")
        
        return "\n\n".join(summary_parts)

# LangGraph node function
def tavily_search_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function for Tavily search operations.
    
    Args:
        state: Current state containing search parameters
        
    Returns:
        Updated state with search results
    """
    try:
        # Extract search parameters from state
        search_type = state.get("search_type", "stock_info")
        ticker = state.get("ticker", "")
        company_name = state.get("company_name", "")
        custom_query = state.get("custom_query", "")
        
        # Initialize search agent
        search_agent = TavilySearchAgent(max_results=5)
        
        # Perform search based on type
        if search_type == "stock_info" and ticker:
            search_results = search_agent.search_stock_info(ticker)
        elif search_type == "company_info" and company_name:
            search_results = search_agent.search_company_info(company_name)
        elif search_type == "market_news":
            search_results = search_agent.search_market_news()
        elif search_type == "technical_analysis" and ticker:
            search_results = search_agent.search_technical_analysis(ticker)
        elif search_type == "sentiment_analysis" and ticker:
            search_results = search_agent.search_sentiment_analysis(ticker)
        elif search_type == "custom" and custom_query:
            search_results = search_agent.search_custom_query(custom_query)
        else:
            search_results = {
                "status": "error",
                "error": "Invalid search parameters",
                "results": [],
                "total_results": 0
            }
        
        # Update state with search results
        state.update({
            "tavily_search_results": search_results,
            "status": "success",
            "next_agent": "data_integrator" if search_results.get("status") == "success" else "end"
        })
        
        return state
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Tavily search failed: {str(e)}",
            "next_agent": "end"
        }

# Integration function for chatbot
def get_web_search_results(query: str) -> Dict[str, Any]:
    """
    Get web search results for chatbot integration.
    
    Args:
        query: Search query
        
    Returns:
        Search results
    """
    try:
        search_agent = TavilySearchAgent(max_results=3)
        results = search_agent.search_custom_query(query)
        return results
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "results": [],
            "total_results": 0
        } 