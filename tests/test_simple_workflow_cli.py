#!/usr/bin/env python3
"""
Simple test to check workflow with enhanced technical analysis
"""

import sys
import os
# Ensure project root on sys.path when running from tests/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_simple_workflow():
    """Test a simple workflow run."""
    
    print("🔍 Testing Simple Workflow")
    print("=" * 40)
    
    try:
        # Import the workflow function
        from langgraph_flow import run_prediction
        
        print("\n1. Running prediction for AAPL...")
        result = run_prediction("AAPL", "1d")
        
        print(f"\n2. Result Status: {result.get('status', 'N/A')}")
        print(f"3. Result Keys: {list(result.keys())}")
        
        if result.get('status') == 'success':
            print("✅ Workflow completed successfully!")
            
            # Check what data is available
            print(f"\n4. Available Data:")
            print(f"   - Technical Analysis: {'Present' if 'technical_analysis' in result else 'Missing'}")
            print(f"   - Enhanced Technical Analysis: {'Present' if 'enhanced_technical_analysis' in result else 'Missing'}")
            print(f"   - Sentiment Analysis: {'Present' if 'sentiment_analysis' in result else 'Missing'}")
            print(f"   - Prediction Result: {'Present' if 'prediction_result' in result else 'Missing'}")
            
            # Show prediction if available
            if 'prediction_result' in result:
                prediction = result['prediction_result']
                print(f"\n5. Prediction Details:")
                print(f"   - Direction: {prediction.get('direction', 'N/A')}")
                print(f"   - Confidence: {prediction.get('confidence', 'N/A')}")
                # Keep the output concise for CLI runs
            
        else:
            print(f"❌ Workflow failed: {result.get('error', 'Unknown error')}")
            
            # Show messages if available
            if 'messages' in result:
                print(f"\n6. Messages from LLM agents:")
                for i, msg in enumerate(result['messages']):
                    print(f"   Message {i}: {type(msg).__name__}")
                    if hasattr(msg, 'content'):
                        print(f"   Content: {msg.content[:100]}...")
                    if hasattr(msg, 'tool_calls'):
                        print(f"   Tool calls: {len(msg.tool_calls) if msg.tool_calls else 0}")
        
        return result
        
    except Exception as e:
        print(f"❌ Exception occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_simple_workflow() 