#!/usr/bin/env python3
"""
Test script to check enhanced technical analysis integration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.technical_analyzer import TechnicalAnalyzerAgent

def test_enhanced_integration():
    """Test the enhanced technical analysis integration."""
    
    print("🔍 Testing Enhanced Technical Analysis Integration")
    print("=" * 60)
    
    # Test data
    test_data = {
        "ticker": "AAPL",
        "period": "6mo"
    }
    
    try:
        # Create agent
        agent = TechnicalAnalyzerAgent()
        
        print("\n1. Testing Enhanced Analysis Method...")
        enhanced_result = agent.analyze_technical_data_with_tools(test_data)
        
        if enhanced_result.get("status") == "success":
            print("✅ Enhanced analysis completed successfully!")
            
            enhanced_analysis = enhanced_result.get("enhanced_technical_analysis", {})
            
            # Check if enhanced analysis has the expected structure
            print(f"\n2. Enhanced Analysis Structure:")
            print(f"   - Has indicators: {'indicators' in enhanced_analysis}")
            print(f"   - Has patterns: {'patterns' in enhanced_analysis}")
            print(f"   - Has support_resistance: {'support_resistance' in enhanced_analysis}")
            print(f"   - Has trend_analysis: {'trend_analysis' in enhanced_analysis}")
            print(f"   - Has trading_signals: {'trading_signals' in enhanced_analysis}")
            print(f"   - Has validation: {'validation' in enhanced_analysis}")
            
            # Show some key metrics
            if 'indicators' in enhanced_analysis:
                indicators = enhanced_analysis['indicators']
                print(f"\n3. Key Indicators:")
                print(f"   - RSI: {indicators.get('rsi', 'N/A')}")
                print(f"   - MACD: {indicators.get('macd', 'N/A')}")
                print(f"   - SMA 20: {indicators.get('sma_20', 'N/A')}")
            
            if 'trading_signals' in enhanced_analysis:
                signals = enhanced_analysis['trading_signals']
                print(f"\n4. Trading Signals:")
                print(f"   - Overall Recommendation: {signals.get('overall_recommendation', 'N/A')}")
                print(f"   - Signal Strength: {signals.get('signal_strength', 'N/A')}")
                print(f"   - Total Signals: {signals.get('total_signals', 'N/A')}")
            
            if 'trend_analysis' in enhanced_analysis:
                trend = enhanced_analysis['trend_analysis']
                print(f"\n5. Trend Analysis:")
                print(f"   - Current Price: {trend.get('current_price', 'N/A')}")
                print(f"   - Short-term Trend: {trend.get('trends', {}).get('short_term', 'N/A')}")
                print(f"   - Medium-term Trend: {trend.get('trends', {}).get('medium_term', 'N/A')}")
            
            print(f"\n6. Technical Score: {enhanced_analysis.get('technical_score', 'N/A')}")
            
        else:
            print(f"❌ Enhanced analysis failed: {enhanced_result.get('error', 'Unknown error')}")
            
            # Try basic analysis as fallback
            print("\n7. Testing Basic Analysis Fallback...")
            basic_result = agent.analyze_technical_data(test_data)
            
            if basic_result.get("status") == "success":
                print("✅ Basic analysis completed successfully!")
                basic_analysis = basic_result.get("technical_analysis", {})
                print(f"   - Technical Score: {basic_analysis.get('technical_score', 'N/A')}")
                print(f"   - Signals: {basic_analysis.get('technical_signals', [])}")
            else:
                print(f"❌ Basic analysis also failed: {basic_result.get('error', 'Unknown error')}")
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_integration() 