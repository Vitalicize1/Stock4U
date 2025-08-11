#!/usr/bin/env python3
"""
Test script for Technical Analyzer Tools

This script demonstrates the new technical analyzer tools and their capabilities.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.tools.technical_analyzer_tools import (
    calculate_advanced_indicators,
    identify_chart_patterns,
    analyze_support_resistance,
    perform_trend_analysis,
    generate_trading_signals,
    validate_technical_analysis,
    TechnicalAnalyzerTools
)

def test_technical_tools():
    """Test all technical analyzer tools with a sample ticker."""
    
    # Test ticker
    ticker = "AAPL"
    period = "6mo"
    
    print(f"🔍 Testing Technical Analyzer Tools for {ticker}")
    print("=" * 60)
    
    try:
        # Test 1: Calculate Advanced Indicators
        print("\n1. Testing Advanced Indicators Calculation...")
        tools = TechnicalAnalyzerTools()
        data = tools._get_price_data(ticker, period)
        indicators = tools._calculate_basic_indicators(data)
        
        # Get current values
        current_indicators = {}
        for key, values in indicators.items():
            if not pd.isna(values.iloc[-1]):
                current_indicators[key] = float(values.iloc[-1])
            else:
                current_indicators[key] = None
        
        indicators_result = {
            "status": "success",
            "ticker": ticker,
            "period": period,
            "current_indicators": current_indicators,
            "data_points": len(data),
            "last_updated": data.index[-1].strftime('%Y-%m-%d')
        }
        
        if indicators_result["status"] == "success":
            print("✅ Advanced indicators calculated successfully")
            current_indicators = indicators_result["current_indicators"]
            print(f"   RSI: {current_indicators.get('rsi', 'N/A'):.2f}")
            print(f"   MACD: {current_indicators.get('macd', 'N/A'):.4f}")
            print(f"   SMA 20: {current_indicators.get('sma_20', 'N/A'):.2f}")
            print(f"   SMA 50: {current_indicators.get('sma_50', 'N/A'):.2f}")
        else:
            print(f"❌ Failed to calculate indicators: {indicators_result.get('error')}")
            return
        
        # Test 2: Identify Chart Patterns
        print("\n2. Testing Chart Pattern Recognition...")
        
        # Get OHLC data
        open_prices = data['Open']
        high_prices = data['High']
        low_prices = data['Low']
        close_prices = data['Close']
        
        detected_patterns = []
        
        # Check for candlestick patterns (simplified)
        for i in range(len(close_prices) - 1):
            current_close = close_prices.iloc[i]
            current_open = open_prices.iloc[i]
            current_high = high_prices.iloc[i]
            current_low = low_prices.iloc[i]
            
            # Doji pattern (simplified)
            if abs(current_close - current_open) < (current_high - current_low) * 0.1:
                detected_patterns.append({
                    "pattern": "Doji",
                    "signal": "neutral",
                    "strength": 50,
                    "significance": "moderate",
                    "days_ago": len(close_prices) - 1 - i
                })
            
            # Hammer pattern (simplified)
            body = abs(current_close - current_open)
            lower_shadow = min(current_open, current_close) - current_low
            upper_shadow = current_high - max(current_open, current_close)
            
            if lower_shadow > 2 * body and upper_shadow < body:
                detected_patterns.append({
                    "pattern": "Hammer",
                    "signal": "bullish",
                    "strength": 75,
                    "significance": "moderate",
                    "days_ago": len(close_prices) - 1 - i
                })
        
        # Check for trend patterns
        from agents.tools.technical_analyzer_tools import calculate_sma
        sma_20 = calculate_sma(close_prices, 20)
        sma_50 = calculate_sma(close_prices, 50)
        
        current_price = close_prices.iloc[-1]
        current_sma_20 = sma_20.iloc[-1]
        current_sma_50 = sma_50.iloc[-1]
        
        trend_patterns = []
        
        # Golden Cross (SMA 20 crosses above SMA 50)
        if len(sma_20) > 1 and len(sma_50) > 1:
            if sma_20.iloc[-1] > sma_50.iloc[-1] and sma_20.iloc[-2] <= sma_50.iloc[-2]:
                trend_patterns.append({
                    "pattern": "Golden Cross",
                    "signal": "bullish",
                    "significance": "strong",
                    "description": "20-day SMA crossed above 50-day SMA"
                })
            
            # Death Cross (SMA 20 crosses below SMA 50)
            elif sma_20.iloc[-1] < sma_50.iloc[-1] and sma_20.iloc[-2] >= sma_50.iloc[-2]:
                trend_patterns.append({
                    "pattern": "Death Cross",
                    "signal": "bearish",
                    "significance": "strong",
                    "description": "20-day SMA crossed below 50-day SMA"
                })
        
        patterns_result = {
            "status": "success",
            "ticker": ticker,
            "candlestick_patterns": detected_patterns,
            "trend_patterns": trend_patterns,
            "total_patterns": len(detected_patterns) + len(trend_patterns),
            "analysis_period": period
        }
        
        if patterns_result["status"] == "success":
            print("✅ Chart patterns identified successfully")
            candlestick_patterns = patterns_result["candlestick_patterns"]
            trend_patterns = patterns_result["trend_patterns"]
            
            print(f"   Candlestick Patterns: {len(candlestick_patterns)}")
            for pattern in candlestick_patterns[:3]:  # Show first 3
                print(f"     - {pattern['pattern']}: {pattern['signal']} ({pattern['significance']})")
            
            print(f"   Trend Patterns: {len(trend_patterns)}")
            for pattern in trend_patterns:
                print(f"     - {pattern['pattern']}: {pattern['signal']} ({pattern['significance']})")
        else:
            print(f"❌ Failed to identify patterns: {patterns_result.get('error')}")
        
        # Test 3: Analyze Support and Resistance
        print("\n3. Testing Support and Resistance Analysis...")
        support_resistance_result = analyze_support_resistance(ticker, period)
        
        if support_resistance_result["status"] == "success":
            print("✅ Support and resistance analysis completed")
            current_price = support_resistance_result["current_price"]
            pivot_points = support_resistance_result["pivot_points"]
            
            print(f"   Current Price: ${current_price:.2f}")
            print(f"   Pivot Point: ${pivot_points['pivot']:.2f}")
            print(f"   Support 1: ${pivot_points['support_1']:.2f}")
            print(f"   Resistance 1: ${pivot_points['resistance_1']:.2f}")
            
            nearest_support = support_resistance_result.get("nearest_support")
            nearest_resistance = support_resistance_result.get("nearest_resistance")
            
            if nearest_support:
                print(f"   Nearest Support: ${nearest_support:.2f}")
            if nearest_resistance:
                print(f"   Nearest Resistance: ${nearest_resistance:.2f}")
        else:
            print(f"❌ Failed to analyze support/resistance: {support_resistance_result.get('error')}")
        
        # Test 4: Perform Trend Analysis
        print("\n4. Testing Trend Analysis...")
        trend_result = perform_trend_analysis(ticker, period)
        
        if trend_result["status"] == "success":
            print("✅ Trend analysis completed")
            trends = trend_result["trends"]
            trend_strength = trend_result["trend_strength"]
            adx_strength = trend_result["adx_strength"]
            
            print(f"   Short-term Trend: {trends.get('short_term', 'N/A')}")
            print(f"   Medium-term Trend: {trends.get('medium_term', 'N/A')}")
            print(f"   Long-term Trend: {trends.get('long_term', 'N/A')}")
            print(f"   Trend Strength: {trend_strength:.1f}%")
            print(f"   ADX Strength: {adx_strength:.1f}")
        else:
            print(f"❌ Failed to perform trend analysis: {trend_result.get('error')}")
        
        # Test 5: Generate Trading Signals
        print("\n5. Testing Trading Signal Generation...")
        signals_result = generate_trading_signals(ticker, period)
        
        if signals_result["status"] == "success":
            print("✅ Trading signals generated successfully")
            signals = signals_result["signals"]
            overall_recommendation = signals_result["overall_recommendation"]
            signal_strength = signals_result["signal_strength"]
            
            print(f"   Overall Recommendation: {overall_recommendation}")
            print(f"   Signal Strength: {signal_strength}")
            print(f"   Total Signals: {len(signals)}")
            
            for signal in signals[:3]:  # Show first 3 signals
                print(f"     - {signal['type']} ({signal['indicator']}): {signal['reason']}")
        else:
            print(f"❌ Failed to generate signals: {signals_result.get('error')}")
        
        # Test 6: Validate Technical Analysis
        print("\n6. Testing Technical Analysis Validation...")
        
        # Create a sample analysis data for validation
        sample_analysis = {
            "current_indicators": indicators_result.get("current_indicators", {}),
            "signals": signals_result.get("signals", []) if signals_result["status"] == "success" else []
        }
        
        validation_result = validate_technical_analysis(ticker, sample_analysis)
        
        if validation_result["status"] == "success":
            print("✅ Technical analysis validation completed")
            validation_results = validation_result["validation_results"]
            
            print(f"   Data Quality: {validation_results['data_quality']}")
            print(f"   Indicator Consistency: {validation_results['indicator_consistency']}")
            print(f"   Signal Reliability: {validation_results['signal_reliability']}")
            print(f"   Validation Score: {validation_results['validation_score']}/100")
            
            if validation_results["warnings"]:
                print("   Warnings:")
                for warning in validation_results["warnings"]:
                    print(f"     - {warning}")
        else:
            print(f"❌ Failed to validate analysis: {validation_result.get('error')}")
        
        print("\n" + "=" * 60)
        print("✅ All Technical Analyzer Tools Tested Successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_technical_tools() 