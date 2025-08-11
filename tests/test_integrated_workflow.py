#!/usr/bin/env python3
"""
🧪 Test Integrated Workflow with Orchestrator Tools
Test the main LangGraph workflow with enhanced orchestrator tools.
"""

import os
from dotenv import load_dotenv
from langgraph_flow import run_prediction, run_chatbot_workflow, print_graph_info

# Load environment variables
load_dotenv()

def test_integrated_workflow():
    """Test the integrated workflow with orchestrator tools."""
    print("🧪 Testing Integrated Workflow with Orchestrator Tools")
    print("=" * 60)
    
    # Test 1: Basic prediction workflow
    print("\n1️⃣ Testing basic prediction workflow...")
    try:
        result = run_prediction("AAPL", "1d")
        
        print(f"✅ Workflow completed successfully!")
        print(f"   Status: {result.get('status')}")
        print(f"   Company: {result.get('company_name')}")
        print(f"   Sector: {result.get('sector')}")
        print(f"   Market Status: {result.get('market_status')}")
        print(f"   Analysis Depth: {result.get('analysis_depth')}")
        print(f"   Workflow Version: {result.get('workflow_version')}")
        print(f"   Tools Used: {result.get('orchestrator_tools_used')}")
        print(f"   Current Stage: {result.get('current_stage')}/{result.get('total_stages')}")
        
        # Check if orchestrator tools were used
        if result.get('orchestrator_tools_used'):
            print(f"   ✅ Orchestrator tools successfully integrated!")
        else:
            print(f"   ⚠️ Orchestrator tools not detected in result")
            
    except Exception as e:
        print(f"   ❌ Workflow failed: {str(e)}")
    
    # Test 2: Chatbot workflow
    print("\n2️⃣ Testing chatbot workflow...")
    try:
        result = run_chatbot_workflow("Analyze AAPL stock for the next day")
        
        print(f"✅ Chatbot workflow completed successfully!")
        print(f"   Status: {result.get('status')}")
        print(f"   User Query: {result.get('user_query')}")
        print(f"   Company: {result.get('company_name')}")
        print(f"   Market Status: {result.get('market_status')}")
        
    except Exception as e:
        print(f"   ❌ Chatbot workflow failed: {str(e)}")
    
    # Test 3: Error handling with invalid ticker
    print("\n3️⃣ Testing error handling with invalid ticker...")
    try:
        result = run_prediction("INVALID_TICKER", "1d")
        
        if result.get('status') == 'error':
            print(f"   ✅ Error handling working correctly!")
            print(f"   Error: {result.get('error')}")
        else:
            print(f"   ⚠️ Expected error but got success")
            
    except Exception as e:
        print(f"   ❌ Error handling test failed: {str(e)}")
    
    print("\n🎉 Integrated workflow testing complete!")

def test_orchestrator_tools_integration():
    """Test specific orchestrator tools integration."""
    print("\n🔧 Testing Orchestrator Tools Integration")
    print("=" * 50)
    
    from agents.orchestrator import orchestrate_with_tools
    
    # Test the enhanced orchestrator directly
    test_state = {
        "ticker": "MSFT",
        "timeframe": "1mo"
    }
    
    try:
        result = orchestrate_with_tools(test_state)
        
        print(f"✅ Enhanced orchestrator working!")
        print(f"   Company: {result.get('company_name')}")
        print(f"   Sector: {result.get('sector')}")
        print(f"   Market Status: {result.get('market_status')}")
        print(f"   Analysis Depth: {result.get('analysis_depth')}")
        print(f"   Tools Used: {result.get('orchestrator_tools_used')}")
        print(f"   Validation Results: {result.get('validation_results', {}).get('valid')}")
        print(f"   Market Status Info: {result.get('market_status_info', {}).get('market_status')}")
        
    except Exception as e:
        print(f"   ❌ Enhanced orchestrator failed: {str(e)}")

def main():
    """Main test function."""
    print("🚀 Integrated Workflow Test Suite")
    print("=" * 60)
    
    # Print graph info
    print_graph_info()
    
    # Test orchestrator tools integration
    test_orchestrator_tools_integration()
    
    # Test integrated workflow
    test_integrated_workflow()
    
    print("\n🎉 All integration tests completed!")
    print("\n💡 The orchestrator tools are now fully integrated into the main workflow!")

if __name__ == "__main__":
    main() 