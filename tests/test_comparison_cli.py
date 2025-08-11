#!/usr/bin/env python3
"""
Test script to compare basic vs enhanced technical analysis
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.technical_analyzer import TechnicalAnalyzerAgent

def test_comparison():
    """Compare basic vs enhanced technical analysis."""
    
    print("🔍 Comparing Basic vs Enhanced Technical Analysis")
    print("=" * 60)
    
    # Test data
    test_data = {
        "ticker": "AAPL",
        "period": "6mo"
    }
    
    try:
        # Create agent
        agent = TechnicalAnalyzerAgent()
        
        print("\n📊 BASIC ANALYSIS RESULTS:")
        print("-" * 30)
        
        # Test basic analysis (this will fail due to data structure, but shows the difference)
        basic_result = agent.analyze_technical_data(test_data)
        
        if basic_result.get("status") == "success":
            basic_analysis = basic_result.get("technical_analysis", {})
            print(f"   Technical Score: {basic_analysis.get('technical_score', 'N/A')}")
            print(f"   Signals: {basic_analysis.get('technical_signals', [])}")
        else:
            print(f"   ❌ Basic analysis failed: {basic_result.get('error', 'Unknown error')}")
            print("   (This is expected - basic analysis needs different data structure)")
        
        print("\n🚀 ENHANCED ANALYSIS RESULTS:")
        print("-" * 30)
        
        # Test enhanced analysis
        enhanced_result = agent.analyze_technical_data_with_tools(test_data)
        
        if enhanced_result.get("status") == "success":
            enhanced_analysis = enhanced_result.get("enhanced_technical_analysis", {})
            
            print(f"   Technical Score: {enhanced_analysis.get('technical_score', 'N/A')}")
            
            # Show indicators
            indicators = enhanced_analysis.get("indicators", {})
            print(f"   RSI: {indicators.get('rsi', 'N/A')}")
            print(f"   MACD: {indicators.get('macd', 'N/A')}")
            print(f"   SMA 20: {indicators.get('sma_20', 'N/A')}")
            print(f"   SMA 50: {indicators.get('sma_50', 'N/A')}")
            
            # Show patterns
            patterns = enhanced_analysis.get("patterns", [])
            print(f"   Patterns Detected: {len(patterns)}")
            for pattern in patterns[:3]:  # Show first 3
                print(f"     - {pattern.get('pattern', 'Unknown')}: {pattern.get('signal', 'Unknown')}")
            
            # Show trading signals
            trading_signals = enhanced_analysis.get("trading_signals", {})
            print(f"   Overall Recommendation: {trading_signals.get('overall_recommendation', 'N/A')}")
            print(f"   Signal Strength: {trading_signals.get('signal_strength', 'N/A')}")
            print(f"   Total Signals: {trading_signals.get('total_signals', 'N/A')}")
            
            # Show trend analysis
            trend_analysis = enhanced_analysis.get("trend_analysis", {})
            trends = trend_analysis.get("trends", {})
            print(f"   Short-term Trend: {trends.get('short_term', 'N/A')}")
            print(f"   Medium-term Trend: {trends.get('medium_term', 'N/A')}")
            print(f"   Long-term Trend: {trends.get('long_term', 'N/A')}")
            print(f"   Trend Strength: {trend_analysis.get('trend_strength', 'N/A')}%")
            
            # Show support/resistance
            support_resistance = enhanced_analysis.get("support_resistance", {})
            current_price = support_resistance.get("current_price", "N/A")
            nearest_support = support_resistance.get("nearest_support", "N/A")
            nearest_resistance = support_resistance.get("nearest_resistance", "N/A")
            print(f"   Current Price: ${current_price}")
            print(f"   Nearest Support: ${nearest_support}")
            print(f"   Nearest Resistance: ${nearest_resistance}")
            
            # Show validation
            validation = enhanced_analysis.get("validation", {})
            print(f"   Validation Score: {validation.get('validation_score', 'N/A')}/100")
            print(f"   Data Quality: {validation.get('data_quality', 'N/A')}")
            
        else:
            print(f"   ❌ Enhanced analysis failed: {enhanced_result.get('error', 'Unknown error')}")
        
        print("\n" + "=" * 60)
        print("✅ COMPARISON COMPLETE!")
        print("\n🎯 KEY DIFFERENCES:")
        print("   📈 Enhanced analysis provides:")
        print("      - More detailed indicators (Williams %R, CCI, ATR)")
        print("      - Pattern recognition (Doji, Hammer, Golden Cross)")
        print("      - Support/resistance analysis (Pivot points, Fibonacci)")
        print("      - Multi-timeframe trend analysis")
        print("      - Comprehensive trading signals")
        print("      - Analysis validation")
        print("      - Enhanced technical scoring algorithm")
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_comparison() 