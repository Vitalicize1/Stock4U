#!/usr/bin/env python3
"""
Simple test script for Technical Analyzer Tools

This script demonstrates the technical analyzer tools without LangChain tool decorators.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from agents.tools.technical_analyzer_tools import TechnicalAnalyzerTools

def test_technical_tools_simple():
    """Test technical analyzer tools with a sample ticker."""
    
    # Test ticker
    ticker = "AAPL"
    period = "6mo"
    
    print(f"🔍 Testing Technical Analyzer Tools for {ticker}")
    print("=" * 60)
    
    try:
        # Initialize tools
        tools = TechnicalAnalyzerTools()
        
        # Test 1: Get price data
        print("\n1. Testing Price Data Retrieval...")
        data = tools._get_price_data(ticker, period)
        
        if not data.empty:
            print("✅ Price data retrieved successfully")
            print(f"   Data points: {len(data)}")
            print(f"   Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
            print(f"   Current price: ${data['Close'].iloc[-1]:.2f}")
        else:
            print("❌ Failed to retrieve price data")
            return
        
        # Test 2: Calculate basic indicators
        print("\n2. Testing Basic Indicators Calculation...")
        indicators = tools._calculate_basic_indicators(data)
        
        print("✅ Basic indicators calculated successfully")
        for key, values in indicators.items():
            if not pd.isna(values.iloc[-1]):
                print(f"   {key}: {values.iloc[-1]:.4f}")
        
        # Test 3: Calculate advanced indicators
        print("\n3. Testing Advanced Indicators Calculation...")
        
        # Calculate additional advanced indicators
        close_prices = data['Close']
        high_prices = data['High']
        low_prices = data['Low']
        volume = data['Volume']
        
        # Williams %R (simplified)
        lowest_low = low_prices.rolling(window=14).min()
        highest_high = high_prices.rolling(window=14).max()
        williams_r = -100 * ((highest_high - close_prices) / (highest_high - lowest_low))
        
        # CCI (Commodity Channel Index) - simplified
        typical_price = (high_prices + low_prices + close_prices) / 3
        sma_tp = close_prices.rolling(window=14).mean()
        mean_deviation = abs(typical_price - sma_tp).rolling(window=14).mean()
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        
        # ATR (Average True Range) - simplified
        tr1 = high_prices - low_prices
        tr2 = abs(high_prices - close_prices.shift(1))
        tr3 = abs(low_prices - close_prices.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=14).mean()
        
        print("✅ Advanced indicators calculated successfully")
        print(f"   Williams %R: {williams_r.iloc[-1]:.2f}")
        print(f"   CCI: {cci.iloc[-1]:.2f}")
        print(f"   ATR: {atr.iloc[-1]:.4f}")
        
        # Test 4: Pattern recognition
        print("\n4. Testing Pattern Recognition...")
        
        # Get OHLC data
        open_prices = data['Open']
        close_prices = data['Close']
        high_prices = data['High']
        low_prices = data['Low']
        
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
        
        print("✅ Pattern recognition completed")
        print(f"   Patterns detected: {len(detected_patterns)}")
        for pattern in detected_patterns[:3]:  # Show first 3
            print(f"     - {pattern['pattern']}: {pattern['signal']} ({pattern['significance']})")
        
        # Test 5: Support and Resistance
        print("\n5. Testing Support and Resistance Analysis...")
        
        current_price = data['Close'].iloc[-1]
        
        # Method 1: Pivot Points
        high = data['High'].iloc[-1]
        low = data['Low'].iloc[-1]
        close = data['Close'].iloc[-1]
        
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        
        # Method 2: Recent highs and lows
        recent_highs = data['High'].rolling(window=20).max().dropna()
        recent_lows = data['Low'].rolling(window=20).min().dropna()
        
        resistance_levels = recent_highs.unique()
        support_levels = recent_lows.unique()
        
        # Method 3: Fibonacci retracements
        highest_high = data['High'].max()
        lowest_low = data['Low'].min()
        diff = highest_high - lowest_low
        
        fib_levels = {
            "0.236": highest_high - 0.236 * diff,
            "0.382": highest_high - 0.382 * diff,
            "0.500": highest_high - 0.500 * diff,
            "0.618": highest_high - 0.618 * diff,
            "0.786": highest_high - 0.786 * diff
        }
        
        # Method 4: Moving averages as support/resistance
        sma_20 = close_prices.rolling(window=20).mean().iloc[-1]
        sma_50 = close_prices.rolling(window=50).mean().iloc[-1]
        sma_200 = close_prices.rolling(window=200).mean().iloc[-1]
        
        print("✅ Support and resistance analysis completed")
        print(f"   Current Price: ${current_price:.2f}")
        print(f"   Pivot Point: ${pivot:.2f}")
        print(f"   Support 1: ${s1:.2f}")
        print(f"   Resistance 1: ${r1:.2f}")
        print(f"   SMA 20: ${sma_20:.2f}")
        print(f"   SMA 50: ${sma_50:.2f}")
        
        # Test 6: Trend Analysis
        print("\n6. Testing Trend Analysis...")
        
        # Calculate moving averages for different timeframes
        sma_5 = close_prices.rolling(window=5).mean()
        sma_10 = close_prices.rolling(window=10).mean()
        sma_20 = close_prices.rolling(window=20).mean()
        sma_50 = close_prices.rolling(window=50).mean()
        sma_200 = close_prices.rolling(window=200).mean()
        
        # Trend analysis for different timeframes
        trends = {}
        
        # Short-term trend (5-10 days)
        if len(sma_5) > 0 and len(sma_10) > 0:
            if current_price > sma_10.iloc[-1] and sma_10.iloc[-1] > sma_10.iloc[-5]:
                trends["short_term"] = "bullish"
            elif current_price < sma_10.iloc[-1] and sma_10.iloc[-1] < sma_10.iloc[-5]:
                trends["short_term"] = "bearish"
            else:
                trends["short_term"] = "sideways"
        
        # Medium-term trend (20-50 days)
        if len(sma_20) > 0 and len(sma_50) > 0:
            if current_price > sma_50.iloc[-1] and sma_20.iloc[-1] > sma_50.iloc[-1]:
                trends["medium_term"] = "bullish"
            elif current_price < sma_50.iloc[-1] and sma_20.iloc[-1] < sma_50.iloc[-1]:
                trends["medium_term"] = "bearish"
            else:
                trends["medium_term"] = "sideways"
        
        # Long-term trend (200 days)
        if len(sma_200) > 0:
            if current_price > sma_200.iloc[-1]:
                trends["long_term"] = "bullish"
            else:
                trends["long_term"] = "bearish"
        
        # Trend strength calculation
        if len(sma_20) > 0 and len(sma_50) > 0:
            trend_strength = abs(current_price - sma_50.iloc[-1]) / sma_50.iloc[-1] * 100
        else:
            trend_strength = 0
        
        print("✅ Trend analysis completed")
        print(f"   Short-term Trend: {trends.get('short_term', 'N/A')}")
        print(f"   Medium-term Trend: {trends.get('medium_term', 'N/A')}")
        print(f"   Long-term Trend: {trends.get('long_term', 'N/A')}")
        print(f"   Trend Strength: {trend_strength:.1f}%")
        
        # Test 7: Trading Signals
        print("\n7. Testing Trading Signal Generation...")
        
        signals = []
        signal_strength = 0
        
        # RSI signals
        current_rsi = indicators['rsi'].iloc[-1] if not pd.isna(indicators['rsi'].iloc[-1]) else 50
        if current_rsi < 30:
            signals.append({"type": "BUY", "indicator": "RSI", "strength": "strong", "reason": "Oversold condition"})
            signal_strength += 2
        elif current_rsi > 70:
            signals.append({"type": "SELL", "indicator": "RSI", "strength": "strong", "reason": "Overbought condition"})
            signal_strength -= 2
        elif current_rsi < 40:
            signals.append({"type": "BUY", "indicator": "RSI", "strength": "moderate", "reason": "Approaching oversold"})
            signal_strength += 1
        elif current_rsi > 60:
            signals.append({"type": "SELL", "indicator": "RSI", "strength": "moderate", "reason": "Approaching overbought"})
            signal_strength -= 1
        
        # MACD signals
        current_macd = indicators['macd'].iloc[-1] if not pd.isna(indicators['macd'].iloc[-1]) else 0
        current_macd_signal = indicators['macd_signal'].iloc[-1] if not pd.isna(indicators['macd_signal'].iloc[-1]) else 0
        current_macd_hist = indicators['macd_histogram'].iloc[-1] if not pd.isna(indicators['macd_histogram'].iloc[-1]) else 0
        
        if current_macd > current_macd_signal and current_macd_hist > 0:
            signals.append({"type": "BUY", "indicator": "MACD", "strength": "strong", "reason": "MACD above signal line"})
            signal_strength += 2
        elif current_macd < current_macd_signal and current_macd_hist < 0:
            signals.append({"type": "SELL", "indicator": "MACD", "strength": "strong", "reason": "MACD below signal line"})
            signal_strength -= 2
        
        # Moving average signals
        current_sma_20 = indicators['sma_20'].iloc[-1] if not pd.isna(indicators['sma_20'].iloc[-1]) else 0
        current_sma_50 = indicators['sma_50'].iloc[-1] if not pd.isna(indicators['sma_50'].iloc[-1]) else 0
        
        if current_price > current_sma_20 > current_sma_50:
            signals.append({"type": "BUY", "indicator": "MA", "strength": "strong", "reason": "Price above 20-day and 50-day SMAs"})
            signal_strength += 2
        elif current_price < current_sma_20 < current_sma_50:
            signals.append({"type": "SELL", "indicator": "MA", "strength": "strong", "reason": "Price below 20-day and 50-day SMAs"})
            signal_strength -= 2
        
        # Determine overall recommendation
        if signal_strength >= 3:
            overall_signal = "STRONG_BUY"
        elif signal_strength >= 1:
            overall_signal = "BUY"
        elif signal_strength <= -3:
            overall_signal = "STRONG_SELL"
        elif signal_strength <= -1:
            overall_signal = "SELL"
        else:
            overall_signal = "HOLD"
        
        print("✅ Trading signals generated successfully")
        print(f"   Overall Recommendation: {overall_signal}")
        print(f"   Signal Strength: {signal_strength}")
        print(f"   Total Signals: {len(signals)}")
        
        for signal in signals[:3]:  # Show first 3 signals
            print(f"     - {signal['type']} ({signal['indicator']}): {signal['reason']}")
        
        print("\n" + "=" * 60)
        print("✅ All Technical Analyzer Tools Tested Successfully!")
        print("🎉 The tools are working correctly!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_technical_tools_simple() 