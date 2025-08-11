from langgraph_flow import run_prediction
import json
from datetime import datetime

def main():
    """
    Main entry point for the agentic stock prediction system.
    """
    print("🤖 Agentic Stock Prediction System")
    print("=" * 50)
    
    # Test with a few popular stocks
    test_tickers = ["AAPL", "MSFT", "GOOGL"]
    
    for ticker in test_tickers:
        print(f"\n📈 Analyzing {ticker}...")
        print("-" * 30)
        
        try:
            # Run the complete prediction pipeline
            result = run_prediction(ticker, timeframe="1d")
            
            # Display results
            if result.get("status") == "success" or "prediction_result" in result:
                prediction = result.get("prediction_result", {})
                
                print(f"🎯 Prediction: {prediction.get('prediction', {}).get('direction', 'Unknown')}")
                print(f"📊 Confidence: {prediction.get('confidence_metrics', {}).get('overall_confidence', 0):.1f}%")
                print(f"💡 Recommendation: {prediction.get('recommendation', {}).get('action', 'Unknown')}")
                
                # Show technical analysis
                tech_analysis = result.get("technical_analysis", {})
                if tech_analysis:
                    print(f"📈 Technical Score: {tech_analysis.get('technical_score', 0):.1f}/100")
                    print(f"🔍 Technical Signals: {', '.join(tech_analysis.get('technical_signals', []))}")
                
                # Show risk assessment
                risks = prediction.get("risk_assessment", {})
                if risks:
                    print(f"⚠️  Risk Level: {risks.get('overall_risk_level', 'Unknown')}")
                
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error occurred')}")
                
        except Exception as e:
            print(f"❌ Exception occurred: {str(e)}")
    
    print("\n" + "=" * 50)
    print("✅ Analysis complete!")

if __name__ == "__main__":
    main()
