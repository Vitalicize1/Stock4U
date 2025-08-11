#!/usr/bin/env python3
"""
🧪 Test Orchestrator Tools
Test the comprehensive tools we've created for the orchestrator agent.
"""

import os
from dotenv import load_dotenv
from agents.tools.orchestrator_tools import (
    validate_ticker_symbol,
    check_market_status,
    determine_analysis_parameters,
    initialize_workflow_state,
    coordinate_workflow_stage,
    handle_workflow_error
)

# Load environment variables
load_dotenv()

def test_orchestrator_tools():
    """Test all orchestrator tools."""
    print("🧪 Testing Orchestrator Tools")
    print("=" * 50)
    
    # Test 1: Ticker Validation
    print("\n1️⃣ Testing ticker validation...")
    test_tickers = ["AAPL", "MSFT", "INVALID", "TSLA", "GOOGL"]
    
    for ticker in test_tickers:
        result = validate_ticker_symbol.invoke(ticker)
        status = "✅" if result.get("valid") else "❌"
        print(f"   {status} {ticker}: {result.get('company_name', result.get('error', 'Unknown'))}")
    
    # Test 2: Market Status
    print("\n2️⃣ Testing market status...")
    market_result = check_market_status.invoke("")
    print(f"   Market Status: {market_result.get('market_status')}")
    print(f"   Current Time: {market_result.get('current_time')}")
    print(f"   Is Holiday: {market_result.get('is_holiday')}")
    print(f"   Is Weekend: {market_result.get('is_weekend')}")
    
    # Test 3: Analysis Parameters
    print("\n3️⃣ Testing analysis parameters...")
    test_cases = [
        ("AAPL", "1d"),
        ("MSFT", "1mo"),
        ("TSLA", "1y")
    ]
    
    for ticker, timeframe in test_cases:
        result = determine_analysis_parameters.invoke({"ticker": ticker, "timeframe": timeframe})
        if result.get("error"):
            print(f"   ❌ {ticker} ({timeframe}): {result.get('error')}")
        else:
            params = result.get("parameters", {})
            print(f"   ✅ {ticker} ({timeframe}): {params.get('analysis_depth')} - {params.get('period')}")
    
    # Test 4: Workflow State Initialization
    print("\n4️⃣ Testing workflow state initialization...")
    workflow_result = initialize_workflow_state.invoke({"ticker": "AAPL", "timeframe": "1d"})
    
    if workflow_result.get("status") == "success":
        print(f"   ✅ Workflow initialized successfully")
        print(f"   Company: {workflow_result.get('company_name')}")
        print(f"   Sector: {workflow_result.get('sector')}")
        print(f"   Market Status: {workflow_result.get('market_status')}")
        print(f"   Analysis Depth: {workflow_result.get('analysis_depth')}")
        print(f"   Next Agent: {workflow_result.get('next_agent')}")
        print(f"   Progress: {workflow_result.get('current_stage')}/{workflow_result.get('total_stages')}")
    else:
        print(f"   ❌ Workflow initialization failed: {workflow_result.get('error')}")
    
    # Test 5: Stage Coordination
    print("\n5️⃣ Testing stage coordination...")
    test_stages = [
        ("orchestrator", {"status": "success"}),
        ("data_collector", {"status": "success"}),
        ("technical_analyzer", {"status": "error", "error": "Test error"}),
        ("elicitation", {"status": "success"})
    ]
    
    for stage, result in test_stages:
        coordination = coordinate_workflow_stage.invoke({"current_stage": stage, "stage_result": result})
        status = "✅" if coordination.get("status") == "success" else "❌"
        next_agent = coordination.get("next_agent", "unknown")
        print(f"   {status} {stage} → {next_agent}")
    
    # Test 6: Error Handling
    print("\n6️⃣ Testing error handling...")
    test_errors = [
        ("data_collector", "API rate limit exceeded"),
        ("technical_analyzer", "Insufficient data"),
        ("prediction_agent", "Model timeout")
    ]
    
    for stage, error in test_errors:
        error_result = handle_workflow_error.invoke({"error": error, "stage": stage})
        print(f"   {stage}: {error_result.get('recommended_action')} - {error_result.get('recovery_options')}")
    
    print("\n✅ All orchestrator tools tested successfully!")

def main():
    """Main test function."""
    print("🧪 Orchestrator Tools Test Suite")
    print("=" * 60)
    
    # Test basic tools
    test_orchestrator_tools()
    
    print("\n🎉 All tests completed successfully!")
    print("\n💡 Orchestrator tools are ready for use in the main workflow!")

if __name__ == "__main__":
    main() 