# agents/data_collector.py
from typing import Dict, Any, Optional
from datetime import datetime
from agents.tools.data_collector_tools import (
    collect_price_data,
    collect_company_info,
    collect_market_data,
    calculate_technical_indicators,
    validate_data_quality,
    collect_comprehensive_data
)

def collect_data(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhanced data collector that uses comprehensive tools for data collection.
    
    Args:
        state: Current state containing ticker and other data
        
    Returns:
        Updated state with collected data
    """
    try:
        ticker = state.get("ticker")
        timeframe = state.get("timeframe", "1d")
        
        if not ticker:
            return {
                "status": "error",
                "error": "No ticker provided",
                "next_agent": "error_handler"
            }
        
        print(f"📊 Enhanced data collector starting for {ticker} ({timeframe} timeframe)")
        
        # Use the comprehensive data collection tool
        comprehensive_data = collect_comprehensive_data.invoke({"ticker": ticker, "period": "3mo"})
        
        if comprehensive_data.get("status") == "error":
            return {
                "status": "error",
                "error": comprehensive_data.get("error"),
                "next_agent": "error_handler"
            }
        
        # Update state with comprehensive data
        state.update({
            "status": "success",
            "data": comprehensive_data,
            "next_agent": "technical_analyzer",
            "data_collection_timestamp": datetime.now().isoformat(),
            "data_collector_tools_used": [
                "collect_price_data",
                "collect_company_info", 
                "collect_market_data",
                "calculate_technical_indicators",
                "validate_data_quality",
                "collect_comprehensive_data"
            ]
        })
        
        # Add summary information
        summary = comprehensive_data.get("summary", {})
        state.update({
            "current_price": summary.get("current_price", 0),
            "company_name": summary.get("company_name", "Unknown"),
            "sector": summary.get("sector", "Unknown"),
            "market_cap": summary.get("market_cap", 0),
            "data_quality_score": summary.get("quality_score", 0),
            "data_points": summary.get("data_points", 0)
        })
        
        print(f"✅ Data collection completed successfully")
        print(f"   Company: {state.get('company_name')}")
        print(f"   Current Price: ${state.get('current_price', 0):.2f}")
        print(f"   Data Quality Score: {state.get('data_quality_score', 0)}")
        print(f"   Data Points: {state.get('data_points', 0)}")
        
        return state
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Data collection failed: {str(e)}",
            "next_agent": "error_handler"
        }

def collect_data_with_tools(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhanced data collector that uses LangGraph ToolNode pattern for comprehensive data collection.
    
    Args:
        state: Current state containing ticker and analysis parameters
        
    Returns:
        Updated state with comprehensive data collection results
    """
    try:
        ticker = state.get("ticker")
        timeframe = state.get("timeframe", "1d")
        
        if not ticker:
            return {
                "status": "error",
                "error": "No ticker provided",
                "next_agent": "error_handler"
            }
        
        print(f"📊 Enhanced data collector starting for {ticker} ({timeframe} timeframe)")
        
        # Check if we need to execute tools
        needs_tools = state.get("needs_tools", True)
        
        if needs_tools:
            print("🔧 Data collector needs tools - routing to data_collector_tools")
            return {
                "status": "success",
                "needs_tools": True,
                "current_tool_node": "data_collector_tools",
                "next_agent": "data_collector_tools",
                "messages": [
                    {
                        "role": "system",
                        "content": f"Collect comprehensive data for ticker {ticker} with timeframe {timeframe}. Gather price data, company info, market data, and validate data quality."
                    }
                ]
            }
        
        # If tools have been executed, process the results
        tool_results = state.get("tool_results", {})
        
        # Extract results from tool execution
        comprehensive_data = tool_results.get("collect_comprehensive_data", {})
        
        if comprehensive_data.get("status") == "error":
            return {
                "status": "error",
                "error": comprehensive_data.get("error"),
                "next_agent": "error_handler"
            }
        
        # Update state with comprehensive data
        state.update({
            "status": "success",
            "needs_tools": False,
            "data": comprehensive_data,
            "next_agent": "technical_analyzer",
            "data_collection_timestamp": datetime.now().isoformat(),
            "data_collector_tools_used": [
                "collect_price_data",
                "collect_company_info", 
                "collect_market_data",
                "calculate_technical_indicators",
                "validate_data_quality",
                "collect_comprehensive_data"
            ]
        })
        
        # Add summary information
        summary = comprehensive_data.get("summary", {})
        state.update({
            "current_price": summary.get("current_price", 0),
            "company_name": summary.get("company_name", "Unknown"),
            "sector": summary.get("sector", "Unknown"),
            "market_cap": summary.get("market_cap", 0),
            "data_quality_score": summary.get("quality_score", 0),
            "data_points": summary.get("data_points", 0)
        })
        
        print(f"✅ Data collection completed successfully")
        print(f"   Company: {state.get('company_name')}")
        print(f"   Current Price: ${state.get('current_price', 0):.2f}")
        print(f"   Data Quality Score: {state.get('data_quality_score', 0)}")
        print(f"   Data Points: {state.get('data_points', 0)}")
        
        return state
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Data collection failed: {str(e)}",
            "next_agent": "error_handler",
            "needs_tools": False
        }

# Legacy function for backward compatibility
class DataCollectorAgent:
    """
    Legacy data collector agent (kept for backward compatibility).
    Use collect_data_with_tools() for enhanced functionality.
    """
    
    def __init__(self):
        self.session = requests.Session()
    
    def collect_data(self, ticker: str, period: str = "3mo") -> Dict[str, Any]:
        """
        Legacy method to collect data (use collect_data_with_tools for enhanced functionality).
        """
        return collect_data_with_tools({"ticker": ticker, "timeframe": period}) 