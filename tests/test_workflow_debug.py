#!/usr/bin/env python3
"""
🔍 Debug Workflow Test
Trace exactly where the workflow is failing and identify the issue.
"""

import os
import sys
from pathlib import Path
import time

# Add the parent directory to the path so we can import modules
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from langgraph_flow import run_prediction, build_graph
from agents.orchestrator import orchestrate_with_tools
from agents.data_collector import collect_data_with_tools
from agents.technical_analyzer import analyze_technical_with_tools
from agents.sentiment_analyzer import analyze_sentiment
from agents.sentiment_integrator import integrate_sentiment
from agents.prediction_agent import make_prediction

# Load environment variables
load_dotenv()

def test_workflow_step_by_step():
    """Test each step of the workflow to identify where it's failing."""
    print("🔍 Debug Workflow Test")
    print("=" * 50)
    
    # Test state
    test_state = {
        "ticker": "AAPL",
        "timeframe": "1d",
        "status": "initialized"
    }
    
    print(f"🚀 Starting with state: {test_state}")
    
    # Step 1: Orchestrator
    print(f"\n1️⃣ Testing Orchestrator...")
    try:
        orchestrator_result = orchestrate_with_tools(test_state)
        print(f"✅ Orchestrator result: {orchestrator_result.get('status')}")
        if orchestrator_result.get('status') == 'success':
            print(f"   Company: {orchestrator_result.get('company_name')}")
            print(f"   Next Agent: {orchestrator_result.get('next_agent')}")
        else:
            print(f"❌ Orchestrator failed: {orchestrator_result.get('error')}")
            return False
    except Exception as e:
        print(f"❌ Orchestrator exception: {str(e)}")
        return False
    
    # Step 2: Data Collector
    print(f"\n2️⃣ Testing Data Collector...")
    try:
        data_result = collect_data_with_tools(orchestrator_result)
        print(f"✅ Data Collector result: {data_result.get('status')}")
        if data_result.get('status') == 'success':
            print(f"   Company: {data_result.get('company_name')}")
            print(f"   Current Price: ${data_result.get('current_price', 0):.2f}")
            print(f"   Next Agent: {data_result.get('next_agent')}")
        else:
            print(f"❌ Data Collector failed: {data_result.get('error')}")
            return False
    except Exception as e:
        print(f"❌ Data Collector exception: {str(e)}")
        return False
    
    # Step 3: Technical Analyzer
    print(f"\n3️⃣ Testing Technical Analyzer...")
    try:
        technical_result = analyze_technical_with_tools(data_result)
        print(f"✅ Technical Analyzer result: {technical_result.get('status')}")
        if technical_result.get('status') == 'success':
            technical_analysis = technical_result.get('technical_analysis', {})
            print(f"   Technical Score: {technical_analysis.get('technical_score', 0):.1f}/100")
            print(f"   Signals: {', '.join(technical_analysis.get('technical_signals', []))}")
            print(f"   Next Agent: {technical_result.get('next_agent')}")
        else:
            print(f"❌ Technical Analyzer failed: {technical_result.get('error')}")
            return False
    except Exception as e:
        print(f"❌ Technical Analyzer exception: {str(e)}")
        return False
    
    # Step 4: Sentiment Analyzer
    print(f"\n4️⃣ Testing Sentiment Analyzer...")
    try:
        sentiment_result = analyze_sentiment(technical_result)
        print(f"✅ Sentiment Analyzer result: {sentiment_result.get('status')}")
        if sentiment_result.get('status') == 'success':
            sentiment_analysis = sentiment_result.get('sentiment_analysis', {})
            overall_sentiment = sentiment_analysis.get('overall_sentiment', {})
            print(f"   Overall Sentiment: {overall_sentiment.get('sentiment_label', 'neutral')}")
            print(f"   Sentiment Score: {overall_sentiment.get('sentiment_score', 0):.2f}")
            print(f"   Next Agent: {sentiment_result.get('next_agent')}")
        else:
            print(f"❌ Sentiment Analyzer failed: {sentiment_result.get('error')}")
            return False
    except Exception as e:
        print(f"❌ Sentiment Analyzer exception: {str(e)}")
        return False
    
    # Step 5: Sentiment Integrator
    print(f"\n5️⃣ Testing Sentiment Integrator...")
    try:
        integration_result = integrate_sentiment(sentiment_result)
        print(f"✅ Sentiment Integrator result: {integration_result.get('status')}")
        if integration_result.get('status') == 'success':
            sentiment_integration = integration_result.get('sentiment_integration', {})
            print(f"   Combined Score: {sentiment_integration.get('combined_score', 0):.1f}")
            print(f"   Final Sentiment: {sentiment_integration.get('final_sentiment', 'neutral')}")
            print(f"   Next Agent: {integration_result.get('next_agent')}")
        else:
            print(f"❌ Sentiment Integrator failed: {integration_result.get('error')}")
            return False
    except Exception as e:
        print(f"❌ Sentiment Integrator exception: {str(e)}")
        return False
    
    # Step 6: Prediction Agent
    print(f"\n6️⃣ Testing Prediction Agent...")
    try:
        prediction_result = make_prediction(integration_result)
        print(f"✅ Prediction Agent result: {prediction_result.get('status')}")
        if prediction_result.get('status') == 'success':
            prediction = prediction_result.get('prediction', {})
            recommendation = prediction_result.get('recommendation', {})
            print(f"   Direction: {prediction.get('direction', 'neutral')}")
            print(f"   Recommendation: {recommendation.get('action', 'HOLD')}")
            print(f"   Confidence: {prediction_result.get('confidence_metrics', {}).get('overall_confidence', 0):.1f}%")
        else:
            print(f"❌ Prediction Agent failed: {prediction_result.get('error')}")
            return False
    except Exception as e:
        print(f"❌ Prediction Agent exception: {str(e)}")
        return False
    
    print(f"\n🎉 All workflow steps completed successfully!")
    return True

def test_complete_workflow():
    """Test the complete workflow using LangGraph."""
    print(f"\n🔄 Testing Complete LangGraph Workflow")
    print("=" * 50)
    
    try:
        start_time = time.time()
        result = run_prediction("AAPL", "1d")
        end_time = time.time()
        
        print(f"✅ Complete workflow completed in {end_time - start_time:.2f} seconds")
        print(f"   Final Status: {result.get('status')}")
        
        # Check if we have prediction results
        if result.get('prediction_result'):
            prediction = result.get('prediction_result', {})
            recommendation = prediction.get('recommendation', {})
            print(f"   Prediction Available: ✅")
            print(f"   Action: {recommendation.get('action', 'HOLD')}")
            print(f"   Direction: {prediction.get('direction', 'neutral')}")
        else:
            print(f"   Prediction Available: ❌")
            print(f"   Error: No prediction_result in final state")
        
        # Check if we have technical analysis
        if result.get('technical_analysis'):
            technical = result.get('technical_analysis', {})
            print(f"   Technical Analysis Available: ✅")
            print(f"   Technical Score: {technical.get('technical_score', 0):.1f}/100")
        else:
            print(f"   Technical Analysis Available: ❌")
        
        # Check if we have sentiment analysis
        if result.get('sentiment_analysis'):
            sentiment = result.get('sentiment_analysis', {})
            print(f"   Sentiment Analysis Available: ✅")
            print(f"   Sentiment: {sentiment.get('overall_sentiment', {}).get('sentiment_label', 'neutral')}")
        else:
            print(f"   Sentiment Analysis Available: ❌")
        
        return result
        
    except Exception as e:
        print(f"❌ Complete workflow failed: {str(e)}")
        return None

def test_frontend_data_structure():
    """Test what data structure the frontend would receive."""
    print(f"\n🖥️ Testing Frontend Data Structure")
    print("=" * 50)
    
    try:
        result = run_prediction("AAPL", "1d")
        
        # Simulate what the frontend would receive
        frontend_data = {
            "status": result.get("status"),
            "ticker": result.get("ticker"),
            "company_name": result.get("company_name"),
            "current_price": result.get("current_price"),
            "technical_analysis": result.get("technical_analysis"),
            "sentiment_analysis": result.get("sentiment_analysis"),
            "prediction_result": result.get("prediction_result"),
            "final_summary": result.get("final_summary")
        }
        
        print(f"📊 Frontend Data Structure:")
        for key, value in frontend_data.items():
            if value is not None:
                print(f"   ✅ {key}: Available")
                if key == "technical_analysis" and isinstance(value, dict):
                    print(f"      Technical Score: {value.get('technical_score', 0):.1f}/100")
                elif key == "prediction_result" and isinstance(value, dict):
                    recommendation = value.get('recommendation', {})
                    print(f"      Recommendation: {recommendation.get('action', 'HOLD')}")
            else:
                print(f"   ❌ {key}: Missing")
        
        return frontend_data
        
    except Exception as e:
        print(f"❌ Frontend data structure test failed: {str(e)}")
        return None

def main():
    """Main debug function."""
    print("🔍 Workflow Debug Test Suite")
    print("=" * 60)
    
    # Test 1: Step by step workflow
    print("🧪 Test 1: Step by Step Workflow")
    step_success = test_workflow_step_by_step()
    
    # Test 2: Complete workflow
    print("\n🧪 Test 2: Complete Workflow")
    complete_result = test_complete_workflow()
    
    # Test 3: Frontend data structure
    print("\n🧪 Test 3: Frontend Data Structure")
    frontend_data = test_frontend_data_structure()
    
    # Summary
    print(f"\n📊 Debug Summary:")
    print(f"   Step-by-step test: {'✅ Passed' if step_success else '❌ Failed'}")
    print(f"   Complete workflow: {'✅ Passed' if complete_result else '❌ Failed'}")
    print(f"   Frontend data: {'✅ Available' if frontend_data else '❌ Missing'}")
    
    if step_success and complete_result and frontend_data:
        print(f"\n🎉 All tests passed! Workflow should be working correctly.")
    else:
        print(f"\n⚠️ Some tests failed. Check the output above for issues.")

if __name__ == "__main__":
    main() 