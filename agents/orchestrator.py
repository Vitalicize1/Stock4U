# agents/orchestrator.py
from typing import Dict, Any
from datetime import datetime
from agents.tools.orchestrator_tools import (
    validate_ticker_symbol,
    check_market_status,
    determine_analysis_parameters,
    initialize_workflow_state,
    coordinate_workflow_stage,
    handle_workflow_error,
    readiness_check_tool,
)

def orchestrate(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orchestrator agent that initializes and coordinates the stock prediction workflow.
    Now enhanced with comprehensive tools for validation, market status checking, and workflow coordination.
    
    Args:
        state: Initial state containing ticker and timeframe
        
    Returns:
        Updated state with initialization data
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
        
        print(f"🚀 Initializing prediction pipeline for {ticker} ({timeframe} timeframe)")
        
        # Use the tool invocation for comprehensive initialization
        # Note: initialize_workflow_state is a LangChain tool; call via .invoke
        workflow_state = initialize_workflow_state.invoke({
            "ticker": ticker,
            "timeframe": timeframe
        })
        
        if workflow_state.get("status") == "error":
            return workflow_state
        
        # Update the state with comprehensive workflow data
        state.update(workflow_state)
        
        print(f"✅ Workflow initialized successfully")
        print(f"   Company: {state.get('company_name')}")
        print(f"   Sector: {state.get('sector')}")
        print(f"   Market Status: {state.get('market_status')}")
        print(f"   Analysis Depth: {state.get('analysis_depth')}")
        print(f"   Next Agent: {state.get('next_agent')}")
        
        return state
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Orchestrator failed: {str(e)}",
            "next_agent": "error_handler"
        }

def orchestrate_with_tools(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhanced orchestrator that uses direct tool calls for comprehensive workflow management.
    
    Args:
        state: Initial state containing ticker and timeframe
        
    Returns:
        Updated state with comprehensive initialization data
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
        
        print(f"🚀 Enhanced orchestrator initializing for {ticker} ({timeframe} timeframe)")
        
        # Step 1: Validate ticker
        print("🔍 Step 1: Validating ticker...")
        ticker_validation = validate_ticker_symbol.invoke({"ticker": ticker})
        if not ticker_validation.get("valid"):
            return {
                "status": "error",
                "error": f"Invalid ticker: {ticker_validation.get('error')}",
                "next_agent": "error_handler"
            }
        print(f"✅ Ticker validated: {ticker_validation.get('company_name')}")
        
        # Step 2: Check market status
        print("📊 Step 2: Checking market status...")
        market_status = check_market_status.invoke({})
        print(f"✅ Market status: {market_status.get('market_status')}")
        
        # Step 3: Determine analysis parameters
        print("⚙️ Step 3: Determining analysis parameters...")
        analysis_params = determine_analysis_parameters.invoke({"ticker": ticker, "timeframe": timeframe})
        if analysis_params.get("error"):
            return {
                "status": "error",
                "error": analysis_params.get("error"),
                "next_agent": "error_handler"
            }
        print(f"✅ Analysis parameters determined")
        
        # Step 4: Initialize workflow state
        print("🎯 Step 4: Initializing workflow state...")
        workflow_state = initialize_workflow_state.invoke({"ticker": ticker, "timeframe": timeframe})
        
        if workflow_state.get("status") == "error":
            return workflow_state
        
        # Update the state with comprehensive workflow data
        state.update(workflow_state)
        
        # Add tool usage tracking
        state["orchestrator_tools_used"] = [
            "validate_ticker_symbol",
            "check_market_status", 
            "determine_analysis_parameters",
            "initialize_workflow_state"
        ]
        
        print(f"✅ Enhanced orchestrator completed successfully")
        print(f"   Company: {state.get('company_name')}")
        print(f"   Sector: {state.get('sector')}")
        print(f"   Market Status: {state.get('market_status')}")
        print(f"   Analysis Depth: {state.get('analysis_depth')}")
        print(f"   Next Agent: {state.get('next_agent')}")
        print(f"   Tools Used: {len(state.get('orchestrator_tools_used', []))}")
        
        return state
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Enhanced orchestrator failed: {str(e)}",
            "next_agent": "error_handler"
        }

def coordinate_stage_transition(current_stage: str, stage_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Coordinate the transition between workflow stages using the coordination tool.
    
    Args:
        current_stage: Current stage name
        stage_result: Result from current stage
        
    Returns:
        Coordination result with next stage information
    """
    try:
        print(f"🔄 Coordinating transition from {current_stage}...")
        
        coordination_result = coordinate_workflow_stage(current_stage, stage_result)
        
        if coordination_result.get("status") == "success":
            next_agent = coordination_result.get("next_agent")
            progress = coordination_result.get("progress", "unknown")
            print(f"✅ Stage transition successful")
            print(f"   Next Agent: {next_agent}")
            print(f"   Progress: {progress}")
        else:
            print(f"❌ Stage transition failed: {coordination_result.get('error')}")
        
        return coordination_result
        
    except Exception as e:
        error_result = handle_workflow_error(str(e), current_stage)
        return {
            "status": "error",
            "error": f"Stage coordination failed: {str(e)}",
            "next_agent": "error_handler",
            "error_details": error_result
        }

def orchestrator_readiness(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wrapper to invoke readiness_check_tool and merge its result into state.
    """
    try:
        ticker = state.get("ticker", "").upper().strip()
        timeframe = state.get("timeframe", "1d")
        res = readiness_check_tool.invoke({"ticker": ticker, "timeframe": timeframe})
        if isinstance(res, dict) and res.get("status") == "success":
            state.update({"readiness": res.get("readiness", {})})
            state.setdefault("status", "success")
            return state
        return {"status": "error", "error": res.get("error", "Unknown error"), **state}
    except Exception as e:
        return {"status": "error", "error": f"orchestrator_readiness failed: {e}", **state}
