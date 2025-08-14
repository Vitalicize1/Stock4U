# agents/orchestrator_tools.py
"""
Orchestrator Tools for Stock Prediction Workflow

This module provides tools for the orchestrator agent to:
- Validate and initialize ticker symbols
- Check market hours and trading status
- Determine analysis parameters
- Coordinate workflow stages
- Handle error conditions
"""

import os
import yfinance as yf
from typing import Dict, Any, List, Optional
from datetime import datetime, time, timedelta
import pytz
from langchain_core.tools import tool
import requests
from dotenv import load_dotenv

# Optional JIRA integration
try:
    from utils.jira import safe_create_issue as _jira_create_issue
except Exception:  # pragma: no cover
    _jira_create_issue = None  # type: ignore

# Load environment variables
load_dotenv()

class OrchestratorTools:
    """
    Tools for the orchestrator agent to manage workflow initialization and coordination.
    """
    
    def __init__(self):
        """Initialize orchestrator tools."""
        self.market_timezone = pytz.timezone('US/Eastern')
        self.nyse_holidays = self._get_nyse_holidays()
    
    def _get_nyse_holidays(self) -> List[str]:
        """Get NYSE holidays for 2024-2025."""
        return [
            "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27",
            "2024-06-19", "2024-07-04", "2024-09-02", "2024-11-28", "2024-12-25",
            "2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26",
            "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25"
        ]

@tool
def validate_ticker_symbol(ticker: str) -> Dict[str, Any]:
    """
    Validate a stock ticker symbol and get basic information.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        
    Returns:
        Dictionary with validation results and ticker info
    """
    try:
        # Use the centralized validation system
        from utils.validation import InputValidator
        
        validation_result = InputValidator.validate_ticker_symbol(ticker)
        
        if not validation_result.is_valid:
            return {
                "valid": False,
                "error": validation_result.error_message,
                "ticker": ticker
            }
        
        # If validation passed, get additional info from yfinance
        ticker_obj = yf.Ticker(validation_result.sanitized_value)
        info = ticker_obj.info
        
        # Check if ticker exists
        if not info or info.get('regularMarketPrice') is None:
            return {
                "valid": False,
                "error": "Ticker not found or invalid",
                "ticker": validation_result.sanitized_value
            }
        
        return {
            "valid": True,
            "ticker": validation_result.sanitized_value,
            "company_name": info.get('longName', 'Unknown'),
            "sector": info.get('sector', 'Unknown'),
            "industry": info.get('industry', 'Unknown'),
            "market_cap": info.get('marketCap'),
            "current_price": info.get('regularMarketPrice'),
            "currency": info.get('currency', 'USD')
        }
        
    except Exception as e:
        return {
            "valid": False,
            "error": f"Validation failed: {str(e)}",
            "ticker": ticker
        }

@tool
def check_market_status() -> Dict[str, Any]:
    """
    Check current market status (open/closed) and trading hours.
    
    Returns:
        Dictionary with market status information
    """
    try:
        now = datetime.now(pytz.timezone('US/Eastern'))
        today = now.strftime('%Y-%m-%d')
        
        # Check if it's a holiday
        is_holiday = today in [
            "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27",
            "2024-06-19", "2024-07-04", "2024-09-02", "2024-11-28", "2024-12-25",
            "2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26",
            "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25"
        ]
        
        # Check if it's weekend
        is_weekend = now.weekday() >= 5
        
        # Define market hours (9:30 AM - 4:00 PM ET)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        # Check if market is currently open
        is_market_open = (
            not is_holiday and 
            not is_weekend and 
            market_open <= now <= market_close
        )
        
        return {
            "market_status": "open" if is_market_open else "closed",
            "current_time": now.strftime('%Y-%m-%d %H:%M:%S ET'),
            "is_holiday": is_holiday,
            "is_weekend": is_weekend,
            "market_hours": {
                "open": "09:30 ET",
                "close": "16:00 ET"
            },
            "next_market_open": _get_next_market_open(now, is_holiday, is_weekend)
        }
        
    except Exception as e:
        return {
            "market_status": "unknown",
            "error": f"Failed to check market status: {str(e)}"
        }

@tool
def determine_analysis_parameters(ticker: str, timeframe: str = "1d") -> Dict[str, Any]:
    """
    Determine optimal analysis parameters based on ticker and timeframe.
    
    Args:
        ticker: Stock ticker symbol
        timeframe: Analysis timeframe (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        
    Returns:
        Dictionary with analysis parameters
    """
    try:
        # Validate ticker first
        ticker_info = validate_ticker_symbol.invoke(ticker)
        if not ticker_info.get("valid"):
            return {
                "error": f"Invalid ticker: {ticker_info.get('error')}",
                "parameters": None
            }
        
        # Define timeframe mappings
        timeframe_mappings = {
            "1d": {"period": "1d", "interval": "1m", "analysis_depth": "intraday"},
            "5d": {"period": "5d", "interval": "5m", "analysis_depth": "short_term"},
            "1mo": {"period": "1mo", "interval": "15m", "analysis_depth": "short_term"},
            "3mo": {"period": "3mo", "interval": "1h", "analysis_depth": "medium_term"},
            "6mo": {"period": "6mo", "interval": "1d", "analysis_depth": "medium_term"},
            "1y": {"period": "1y", "interval": "1d", "analysis_depth": "long_term"},
            "2y": {"period": "2y", "interval": "1d", "analysis_depth": "long_term"},
            "5y": {"period": "5y", "interval": "1d", "analysis_depth": "long_term"},
            "10y": {"period": "10y", "interval": "1d", "analysis_depth": "long_term"},
            "ytd": {"period": "ytd", "interval": "1d", "analysis_depth": "year_to_date"},
            "max": {"period": "max", "interval": "1d", "analysis_depth": "historical"}
        }
        
        # Get parameters for timeframe
        params = timeframe_mappings.get(timeframe, timeframe_mappings["1d"])
        
        # Add ticker-specific parameters
        params.update({
            "ticker": ticker,
            "company_name": ticker_info.get("company_name"),
            "sector": ticker_info.get("sector"),
            "industry": ticker_info.get("industry"),
            "market_cap": ticker_info.get("market_cap"),
            "currency": ticker_info.get("currency", "USD")
        })
        
        return {
            "parameters": params,
            "recommended_indicators": _get_recommended_indicators(params["analysis_depth"]),
            "data_sources": ["yfinance", "technical_indicators", "market_data", "news_sentiment"]
        }
        
    except Exception as e:
        return {
            "error": f"Failed to determine parameters: {str(e)}",
            "parameters": None
        }

@tool
def initialize_workflow_state(ticker: str, timeframe: str = "1d") -> Dict[str, Any]:
    """
    Initialize the complete workflow state for stock prediction.
    
    Args:
        ticker: Stock ticker symbol
        timeframe: Analysis timeframe
        
    Returns:
        Complete workflow state dictionary
    """
    try:
        # Validate ticker
        ticker_info = validate_ticker_symbol.invoke(ticker)
        if not ticker_info.get("valid"):
            return {
                "status": "error",
                "error": f"Invalid ticker: {ticker_info.get('error')}",
                "next_agent": "error_handler"
            }
        
        # Check market status
        market_status = check_market_status.invoke("")
        
        # Get analysis parameters
        analysis_params = determine_analysis_parameters.invoke({"ticker": ticker, "timeframe": timeframe})
        if analysis_params.get("error"):
            return {
                "status": "error",
                "error": analysis_params.get("error"),
                "next_agent": "error_handler"
            }
        
        # Initialize workflow state
        workflow_state = {
            "workflow_start_time": datetime.now().isoformat(),
            "ticker": ticker.upper(),
            "timeframe": timeframe,
            "status": "success",
            "pipeline_stage": "orchestrator",
            "next_agent": "data_collector",
            
            # Workflow metadata
            "workflow_version": "2.0",
            "prediction_type": "stock_movement",
            "analysis_depth": analysis_params["parameters"]["analysis_depth"],
            "data_sources": analysis_params["data_sources"],
            
            # Ticker information
            "company_name": ticker_info.get("company_name"),
            "sector": ticker_info.get("sector"),
            "industry": ticker_info.get("industry"),
            "market_cap": ticker_info.get("market_cap"),
            "currency": ticker_info.get("currency"),
            
            # Market status
            "market_status": market_status.get("market_status"),
            "is_market_open": market_status.get("market_status") == "open",
            
            # Analysis parameters
            "analysis_parameters": analysis_params["parameters"],
            "recommended_indicators": analysis_params["recommended_indicators"],
            
            # Workflow stages
            "workflow_stages": [
                "orchestrator",
                "data_collector", 
                "technical_analyzer",
                "sentiment_analyzer",
                "sentiment_integrator",
                "prediction_agent",
                "evaluator_optimizer",
                "elicitation"
            ],
            "current_stage": 0,
            "total_stages": 8
        }
        
        return workflow_state
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Workflow initialization failed: {str(e)}",
            "next_agent": "error_handler"
        }

@tool
def coordinate_workflow_stage(current_stage: str, stage_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Coordinate the transition between workflow stages.
    
    Args:
        current_stage: Current stage name
        stage_result: Result from current stage
        
    Returns:
        Coordination result with next stage information
    """
    try:
        workflow_stages = [
            "orchestrator",
            "data_collector", 
            "technical_analyzer",
            "sentiment_analyzer",
            "sentiment_integrator",
            "prediction_agent",
            "evaluator_optimizer",
            "elicitation"
        ]
        
        # Find current stage index
        current_index = workflow_stages.index(current_stage) if current_stage in workflow_stages else -1
        
        # Check if stage was successful
        if stage_result.get("status") != "success":
            return {
                "status": "error",
                "error": f"Stage {current_stage} failed: {stage_result.get('error', 'Unknown error')}",
                "next_agent": "error_handler",
                "current_stage": current_stage,
                "stage_index": current_index
            }
        
        # Determine next stage
        if current_index >= len(workflow_stages) - 1:
            return {
                "status": "success",
                "next_agent": "end",
                "workflow_complete": True,
                "current_stage": current_stage,
                "stage_index": current_index
            }
        else:
            next_stage = workflow_stages[current_index + 1]
            return {
                "status": "success",
                "next_agent": next_stage,
                "current_stage": current_stage,
                "next_stage": next_stage,
                "stage_index": current_index,
                "progress": f"{current_index + 1}/{len(workflow_stages)}"
            }
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Workflow coordination failed: {str(e)}",
            "next_agent": "error_handler"
        }

@tool
def handle_workflow_error(error: str, stage: str) -> Dict[str, Any]:
    """
    Handle workflow errors and provide recovery options.
    
    Args:
        error: Error message
        stage: Stage where error occurred
        
    Returns:
        Error handling result with recovery options
    """
    try:
        # Define error recovery strategies
        recovery_strategies = {
            "data_collector": ["retry", "use_cached_data", "skip_stage"],
            "technical_analyzer": ["retry", "use_basic_indicators", "skip_stage"],
            "sentiment_analyzer": ["retry", "use_basic_sentiment", "skip_stage"],
            "prediction_agent": ["retry", "use_technical_only", "skip_stage"],
            "evaluator_optimizer": ["retry", "use_basic_evaluation", "skip_stage"]
        }
        
        # Get recovery options for the stage
        recovery_options = recovery_strategies.get(stage, ["retry", "skip_stage"])
        
        response = {
            "status": "error_handled",
            "error": error,
            "stage": stage,
            "recovery_options": recovery_options,
            "recommended_action": "retry" if "retry" in recovery_options else "skip_stage",
            "error_timestamp": datetime.now().isoformat()
        }
        # Optionally create or deduplicate a Jira issue for the error
        if _jira_create_issue is not None and error:
            import hashlib, json as _json, tempfile
            dedup_key = f"stock4u_error_{hashlib.md5(f'{stage}|{error}'.encode('utf-8')).hexdigest()}"
            summary = f"Stock4U error in {stage}: {str(error)[:80]}"
            desc = (
                "Auto-generated from Stock4U.\n\n" +
                f"Stage: {stage}\n" +
                f"Error: {error}\n" +
                ("\nContext:\n```\n" + _json.dumps(response)[:5000] + "\n```" if response else "")
            )
            try:
                jira_res = _jira_create_issue(summary, desc, issue_type="Bug", labels=["stock4u", f"stage:{stage}"], dedup_key=dedup_key)
                response["jira"] = jira_res
                # If we created a new issue, optionally attach context JSON file
                key = None
                if isinstance(jira_res, dict):
                    if jira_res.get("status") == "success":
                        key = (jira_res.get("data") or {}).get("key")
                    elif jira_res.get("status") == "duplicate":
                        key = (jira_res.get("issue") or {}).get("key")
                if key:
                    try:
                        from utils.jira import safe_attach_file
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{key}_context.json")
                        tmp.write(_json.dumps(response, indent=2).encode("utf-8"))
                        tmp.close()
                        safe_attach_file(key, tmp.name)
                    except Exception:
                        pass
            except Exception:
                pass
        return response
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Error handling failed: {str(e)}",
            "stage": stage
        }

def _get_recommended_indicators(analysis_depth: str) -> List[str]:
    """Get recommended technical indicators based on analysis depth."""
    indicators = {
        "intraday": ["SMA", "EMA", "RSI", "MACD", "Bollinger_Bands", "Volume"],
        "short_term": ["SMA", "EMA", "RSI", "MACD", "Bollinger_Bands", "Stochastic", "Volume"],
        "medium_term": ["SMA", "EMA", "RSI", "MACD", "Bollinger_Bands", "Stochastic", "ATR", "Volume"],
        "long_term": ["SMA", "EMA", "RSI", "MACD", "Bollinger_Bands", "Stochastic", "ATR", "Williams_R", "Volume"],
        "year_to_date": ["SMA", "EMA", "RSI", "MACD", "Bollinger_Bands", "Volume"],
        "historical": ["SMA", "EMA", "RSI", "MACD", "Bollinger_Bands", "ATR", "Volume"]
    }
    return indicators.get(analysis_depth, ["SMA", "EMA", "RSI", "MACD"])

def _get_next_market_open(now: datetime, is_holiday: bool, is_weekend: bool) -> str:
    """Calculate next market open time."""
    if is_holiday or is_weekend:
        # Find next trading day
        next_day = now
        while True:
            next_day = next_day + timedelta(days=1)
            if next_day.weekday() < 5 and next_day.strftime('%Y-%m-%d') not in [
                "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27",
                "2024-06-19", "2024-07-04", "2024-09-02", "2024-11-28", "2024-12-25",
                "2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26",
                "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25"
            ]:
                break
        return next_day.replace(hour=9, minute=30, second=0, microsecond=0).strftime('%Y-%m-%d %H:%M:%S ET')
    else:
        # Market opens tomorrow at 9:30 AM ET
        tomorrow = now + timedelta(days=1)
        return tomorrow.replace(hour=9, minute=30, second=0, microsecond=0).strftime('%Y-%m-%d %H:%M:%S ET')

@tool
def readiness_check_tool(ticker: str, timeframe: str = "1d") -> Dict[str, Any]:
    """
    Quick readiness checklist for orchestrator stage. Validates ticker, checks market
    status, determines analysis parameters, and reports missing API keys.

    Returns a dictionary with readiness status, details, missing_keys, warnings, and next_steps.
    """
    try:
        # 1) Validate ticker
        validation = validate_ticker_symbol.invoke({"ticker": ticker})
        ticker_valid = bool(validation.get("valid"))

        # 2) Market status
        market = check_market_status.invoke({})
        market_open = market.get("market_status") == "open"

        # 3) Determine parameters
        params_res = determine_analysis_parameters.invoke({"ticker": ticker, "timeframe": timeframe})
        params_err = params_res.get("error")
        params = params_res.get("parameters", {}) if not params_err else {}

        # 4) Environment/API keys readiness
        required_env_sets = {
            "llm_any": ["GROQ_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
            "news_optional": ["NEWS_API_KEY"],
            "reddit_optional": ["REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET"],
            "tavily_optional": ["TAVILY_API_KEY"],
        }
        missing_keys: list[str] = []
        # At least one LLM key is preferred (but optional if rule-based)
        if not any(os.getenv(k) for k in required_env_sets["llm_any"]):
            missing_keys.append("(LLM) GROQ_API_KEY/GOOGLE_API_KEY/OPENAI_API_KEY/ANTHROPIC_API_KEY")
        # Optional providers
        for k in required_env_sets["news_optional"] + required_env_sets["reddit_optional"] + required_env_sets["tavily_optional"]:
            if not os.getenv(k):
                missing_keys.append(k)

        # 5) Compose readiness and guidance
        warnings = []
        if not ticker_valid:
            warnings.append("Ticker appears invalid")
        if params_err:
            warnings.append(f"Parameter derivation failed: {params_err}")
        if not market_open:
            warnings.append("Market is currently closed")

        next_steps = []
        if not ticker_valid:
            next_steps.append("Provide a valid ticker (e.g., AAPL, MSFT)")
        if params and not params_err:
            next_steps.append(f"Proceed with analysis_depth='{params.get('analysis_depth')}' and period='{params.get('period')}'")
        if missing_keys:
            next_steps.append("Optionally set missing API keys to enable LLM/search/sentiment features")

        status = "ready" if ticker_valid and not params_err else "attention_required"

        return {
            "status": "success",
            "readiness": {
                "state": status,
                "ticker": ticker.upper().strip(),
                "timeframe": timeframe,
                "ticker_valid": ticker_valid,
                "market_status": market.get("market_status"),
                "is_market_open": market_open,
                "analysis_parameters": params,
                "missing_keys": missing_keys,
                "warnings": warnings,
                "next_steps": next_steps,
            },
        }
    except Exception as e:
        return {"status": "error", "error": f"readiness_check_tool failed: {e}"}

@tool
def delegate_prediction_agent(analysis_summary: str, available_llms: List[str] = None) -> Dict[str, Any]:
    """
    Delegate prediction to the best available LLM agent based on availability and performance.
    
    Args:
        analysis_summary: Comprehensive analysis summary for prediction
        available_llms: List of available LLM providers (e.g., ['groq', 'gemini'])
        
    Returns:
        Dictionary with delegation results and selected agent
    """
    try:
        # Default available LLMs if not provided
        if available_llms is None:
            available_llms = ['groq', 'gemini']
        
        # Check which LLMs are actually available
        working_llms = []
        
        # Test Groq availability
        if 'groq' in available_llms:
            try:
                from llm.groq_client import get_groq_client
                groq_client = get_groq_client()
                working_llms.append('groq')
            except Exception as e:
                print(f"Groq not available: {e}")
        
        # Record Gemini availability without provoking retries
        if 'gemini' in available_llms:
            try:
                from llm.gemini_client import get_gemini_client
                # Create client without test call; if construction fails, it's unavailable
                _ = get_gemini_client()
                working_llms.append('gemini')
            except Exception as e:
                print(f"Gemini not available: {e}")
        
        # Select the best available LLM (prefer Groq for speed)
        if 'groq' in working_llms:
            selected_llm = 'groq'
            reason = "Groq selected for speed and reliability"
        elif 'gemini' in working_llms:
            selected_llm = 'gemini'
            reason = "Gemini selected as fallback"
        else:
            selected_llm = 'rule_based'
            reason = "No LLM available - using rule-based prediction"
        
        # Generate prediction using selected agent
        prediction_result = None
        if selected_llm == 'groq':
            from llm.groq_client import get_groq_prediction
            prediction_result = get_groq_prediction(analysis_summary)
        elif selected_llm == 'gemini':
            from llm.gemini_client import get_gemini_prediction
            prediction_result = get_gemini_prediction(analysis_summary)
        else:
            # Rule-based fallback
            prediction_result = {
                "direction": "NEUTRAL",
                "confidence": 50.0,
                "price_target": None,
                "price_range": {"low": None, "high": None},
                "reasoning": "Rule-based prediction due to LLM unavailability",
                "key_factors": ["Technical analysis indicates mixed signals"],
                "risk_factors": ["Limited analysis due to LLM unavailability"]
            }
        
        return {
            "delegation_successful": True,
            "selected_agent": selected_llm,
            "reason": reason,
            "available_llms": working_llms,
            "prediction_result": prediction_result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "delegation_successful": False,
            "selected_agent": "rule_based",
            "reason": f"Delegation failed: {str(e)}",
            "available_llms": [],
            "prediction_result": {
                "direction": "NEUTRAL",
                "confidence": 50.0,
                "price_target": None,
                "price_range": {"low": None, "high": None},
                "reasoning": f"Prediction failed: {str(e)}",
                "key_factors": ["Analysis failed"],
                "risk_factors": ["Unable to assess risks"]
            },
            "timestamp": datetime.now().isoformat()
        }

# Create orchestrator tools instance
orchestrator_tools = OrchestratorTools() 