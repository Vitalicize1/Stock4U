#!/usr/bin/env python3
"""
Debug script to test the workflow step by step
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langgraph_flow import run_prediction

def test_workflow():
    """Test the workflow step by step."""
    
    print("🔍 Testing Workflow Step by Step")
    print("=" * 60)
    
    try:
        print("\n1. Testing workflow with AAPL...")
        result = run_prediction("AAPL", "1d")
        
        print(f"\n2. Workflow Result:")
        print(f"   Status: {result.get('status', 'N/A')}")
        print(f"   Error: {result.get('error', 'None')}")
        
        if result.get('status') == 'success':
            print(f"   Technical Analysis: {'Present' if 'technical_analysis' in result else 'Missing'}")
            print(f"   Sentiment Analysis: {'Present' if 'sentiment_analysis' in result else 'Missing'}")
            print(f"   Prediction: {'Present' if 'prediction_result' in result else 'Missing'}")
            
            # Show technical analysis details
            if 'technical_analysis' in result:
                tech_analysis = result['technical_analysis']
                print(f"\n3. Technical Analysis Details:")
                print(f"   Type: {type(tech_analysis)}")
                if isinstance(tech_analysis, dict):
                    print(f"   Keys: {list(tech_analysis.keys())}")
                    if 'enhanced_technical_analysis' in tech_analysis:
                        enhanced = tech_analysis['enhanced_technical_analysis']
                        print(f"   Enhanced Analysis Present: {'Yes' if enhanced else 'No'}")
                        if enhanced:
                            print(f"   Technical Score: {enhanced.get('technical_score', 'N/A')}")
                            print(f"   Patterns: {len(enhanced.get('patterns', []))}")
                    elif 'technical_score' in tech_analysis:
                        print(f"   Basic Technical Score: {tech_analysis.get('technical_score', 'N/A')}")
            
            # Show prediction details
            if 'prediction_result' in result:
                prediction = result['prediction_result']
                print(f"\n4. Prediction Details:")
                print(f"   Type: {type(prediction)}")
                if isinstance(prediction, dict):
                    print(f"   Keys: {list(prediction.keys())}")
                    print(f"   Prediction: {prediction.get('prediction', 'N/A')}")
                    print(f"   Confidence: {prediction.get('confidence', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Workflow test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_workflow() 