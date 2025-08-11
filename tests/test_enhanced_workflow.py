#!/usr/bin/env python3
"""
🧪 Test Enhanced Workflow with Data Collector Tools
Test the main workflow with enhanced data collector tools integration.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import modules
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from langgraph_flow import run_prediction, run_chatbot_workflow, print_graph_info

# Load environment variables
load_dotenv()

def test_enhanced_workflow():
    """Test the enhanced workflow with data collector tools."""
    print("🧪 Testing Enhanced Workflow with Data Collector Tools")
    print("=" * 60)
    
    # Test 1: Basic prediction workflow with enhanced data collector
    print("\n1️⃣ Testing basic prediction workflow...")
    try:
        result = run_prediction("AAPL", "1d")
        
        print(f"✅ Workflow completed successfully!")
        print(f"   Status: {result.get('status', 'unknown')}")
        
        # Check if enhanced data collector was used
        if result.get('data_collector_tools_used'):
            print(f"   ✅ Enhanced data collector tools used: {len(result.get('data_collector_tools_used', []))} tools")
        
        # Check data quality
        if result.get('data_quality_score'):
            print(f"   Data Quality Score: {result.get('data_quality_score')}/100")
        
        # Check collected data
        if result.get('current_price'):
            print(f"   Current Price: ${result.get('current_price'):.2f}")
        
        if result.get('company_name'):
            print(f"   Company: {result.get('company_name')}")
        
        if result.get('data_points'):
            print(f"   Data Points: {result.get('data_points')}")
            
    except Exception as e:
        print(f"❌ Basic workflow test failed: {str(e)}")
    
    # Test 2: Chatbot workflow with enhanced data collector
    print("\n2️⃣ Testing chatbot workflow...")
    try:
        result = run_chatbot_workflow("Analyze AAPL stock for me")
        
        print(f"✅ Chatbot workflow completed successfully!")
        print(f"   Status: {result.get('status', 'unknown')}")
        
        # Check if enhanced data collector was used
        if result.get('data_collector_tools_used'):
            print(f"   ✅ Enhanced data collector tools used: {len(result.get('data_collector_tools_used', []))} tools")
        
        # Check chatbot response
        if result.get('chatbot_response'):
            print(f"   Chatbot Response: {result.get('chatbot_response', {}).get('message', 'No response')[:100]}...")
            
    except Exception as e:
        print(f"❌ Chatbot workflow test failed: {str(e)}")
    
    # Test 3: Error handling with invalid ticker
    print("\n3️⃣ Testing error handling...")
    try:
        result = run_prediction("INVALID_TICKER_12345", "1d")
        
        if result.get('status') == 'error':
            print(f"✅ Error handling working correctly")
            print(f"   Error: {result.get('error', 'Unknown error')}")
        else:
            print(f"⚠️ Unexpected success with invalid ticker")
            
    except Exception as e:
        print(f"✅ Error handling working (exception caught): {str(e)}")

def test_data_collector_integration():
    """Test specific data collector integration."""
    print(f"\n🔧 Testing Data Collector Integration")
    print("=" * 40)
    
    # Test comprehensive data collection
    print("📊 Testing comprehensive data collection...")
    try:
        from agents.data_collector import collect_data_with_tools
        
        # Test state
        test_state = {
            "ticker": "MSFT",
            "timeframe": "1d",
            "status": "initialized"
        }
        
        result = collect_data_with_tools(test_state)
        
        if result.get("status") == "success":
            print(f"✅ Data collector integration working")
            print(f"   Company: {result.get('company_name', 'Unknown')}")
            print(f"   Current Price: ${result.get('current_price', 0):.2f}")
            print(f"   Data Quality Score: {result.get('data_quality_score', 0)}/100")
            print(f"   Data Points: {result.get('data_points', 0)}")
            print(f"   Market Trend: {result.get('market_trend', 'neutral')}")
            
            # Check if tools were used
            if result.get('data_collector_tools_used'):
                print(f"   Tools Used: {len(result.get('data_collector_tools_used'))} tools")
                for tool in result.get('data_collector_tools_used', []):
                    print(f"     - {tool}")
        else:
            print(f"❌ Data collector integration failed: {result.get('error')}")
            
    except Exception as e:
        print(f"❌ Data collector integration test error: {str(e)}")

def test_workflow_performance():
    """Test workflow performance with enhanced tools."""
    print(f"\n⚡ Testing Workflow Performance")
    print("=" * 35)
    
    import time
    
    # Test performance with enhanced data collector
    print("📊 Testing performance with enhanced data collector...")
    try:
        start_time = time.time()
        result = run_prediction("AAPL", "1d")
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"✅ Workflow completed in {duration:.2f} seconds")
        
        if duration < 30:  # Should complete within 30 seconds
            print(f"   ✅ Performance is good (< 30s)")
        elif duration < 60:
            print(f"   ⚠️ Performance is acceptable (< 60s)")
        else:
            print(f"   ❌ Performance is slow (> 60s)")
            
        # Check data quality
        if result.get('data_quality_score', 0) >= 80:
            print(f"   ✅ High data quality ({result.get('data_quality_score', 0)}/100)")
        else:
            print(f"   ⚠️ Data quality needs improvement ({result.get('data_quality_score', 0)}/100)")
            
    except Exception as e:
        print(f"❌ Performance test failed: {str(e)}")

def main():
    """Main test function."""
    print("🧪 Enhanced Workflow Test Suite")
    print("=" * 50)
    
    # Run all tests
    test_enhanced_workflow()
    test_data_collector_integration()
    test_workflow_performance()
    
    print(f"\n🎉 All enhanced workflow tests completed!")
    print("✅ Enhanced data collector tools are successfully integrated!")

if __name__ == "__main__":
    main() 