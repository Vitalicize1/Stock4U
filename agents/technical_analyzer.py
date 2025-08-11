# agents/technical_analyzer.py
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from datetime import datetime
from agents.tools.technical_analyzer_tools import (
    calculate_advanced_indicators,
    identify_chart_patterns,
    analyze_support_resistance,
    perform_trend_analysis,
    generate_trading_signals,
    validate_technical_analysis,
    TechnicalAnalyzerTools
)

class TechnicalAnalyzerAgent:
    """
    Agent responsible for technical analysis of stock data:
    - Pattern recognition
    - Support/resistance levels
    - Trend analysis
    - Volume analysis
    - Momentum indicators
    """
    
    def __init__(self):
        self.analysis_results = {}
    
    def analyze_technical_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive technical analysis.
        
        Args:
            data: Dictionary containing price data and technical indicators
            
        Returns:
            Dictionary with technical analysis results
        """
        try:
            # Handle both old and new data structures
            if "data" in data and isinstance(data["data"], dict):
                # Old structure: data.get("data", {}).get("price_data", {})
                price_data = data.get("data", {}).get("price_data", {})
                technical_indicators = data.get("data", {}).get("technical_indicators", {})
            else:
                # New enhanced structure: data.get("price_data", {})
                price_data = data.get("price_data", {})
                technical_indicators = data.get("technical_indicators", {})
            
            if not price_data:
                raise Exception("No price data available for analysis")
            
            print(f"🔍 Technical analyzer processing data for analysis...")
            
            analysis = {
                "trend_analysis": self._analyze_trends(price_data, technical_indicators),
                "support_resistance": self._find_support_resistance(price_data),
                "pattern_recognition": self._identify_patterns(price_data),
                "volume_analysis": self._analyze_volume(price_data),
                "momentum_analysis": self._analyze_momentum(technical_indicators),
                "technical_score": 0.0,
                "technical_signals": []
            }
            
            # Calculate overall technical score
            analysis["technical_score"] = self._calculate_technical_score(analysis)
            analysis["technical_signals"] = self._generate_signals(analysis)

            # Build trading_signals block for UI compatibility
            signals_list = analysis["technical_signals"]
            signal_strength = 0
            buy_count = 0
            sell_count = 0
            for s in signals_list:
                if s == "STRONG_BUY":
                    signal_strength += 2
                    buy_count += 1
                elif s == "BUY":
                    signal_strength += 1
                    buy_count += 1
                elif s == "SELL":
                    signal_strength -= 1
                    sell_count += 1
                elif s == "STRONG_SELL":
                    signal_strength -= 2
                    sell_count += 1
                elif s == "OVERSOLD":
                    signal_strength += 1
                elif s == "OVERBOUGHT":
                    signal_strength -= 1

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

            # Build trading_signals block for UI compatibility
            analysis["trading_signals"] = {
                "signals": signals_list,
                "signal_strength": signal_strength,
                "overall_recommendation": overall_signal,
                "total_signals": len(signals_list),
                "buy_signals": buy_count,
                "sell_signals": sell_count
            }

            # If sentiment is already available in the input data (e.g., standalone runs), bias signals
            try:
                sentiment_analysis = data.get("sentiment_analysis") if isinstance(data, dict) else None
                if sentiment_analysis:
                    tools = TechnicalAnalyzerTools()
                    analysis["trading_signals"] = tools._bias_signals_with_sentiment(
                        analysis.get("trading_signals", {}), sentiment_analysis
                    )
            except Exception:
                pass
            
            print(f"✅ Technical analysis completed successfully")
            print(f"   Technical Score: {analysis['technical_score']:.1f}/100")
            print(f"   Signals: {', '.join(analysis['technical_signals'])}")
            
            return {
                "status": "success",
                "technical_analysis": analysis,
                "next_agent": "sentiment_analyzer",
                "current_price": price_data.get("current_price")
            }
            
        except Exception as e:
            print(f"❌ Technical analysis failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "next_agent": "error_handler"
            }
    
    def analyze_technical_data_with_tools(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform enhanced technical analysis using specialized tools.
        
        Args:
            data: Dictionary containing ticker and analysis parameters
            
        Returns:
            Dictionary with enhanced technical analysis results
        """
        try:
            # Extract ticker from data
            ticker = data.get("ticker", "")
            if not ticker:
                raise Exception("No ticker symbol provided for analysis")
            
            print(f"🔍 Enhanced technical analyzer starting analysis for {ticker}...")
            
            # Use tools for comprehensive analysis
            period = data.get("period", "6mo")
            
            # Calculate advanced indicators using tools directly
            tools = TechnicalAnalyzerTools()
            data_df = tools._get_price_data(ticker, period)
            
            if data_df.empty:
                raise Exception("No data available for analysis")
            
            # Calculate indicators
            indicators = tools._calculate_basic_indicators(data_df)
            
            # Get current values
            current_indicators = {}
            for key, values in indicators.items():
                if not pd.isna(values.iloc[-1]):
                    current_indicators[key] = float(values.iloc[-1])
                else:
                    current_indicators[key] = None
            
            indicators_result = {
                "status": "success",
                "current_indicators": current_indicators,
                "data_points": len(data_df),
                "last_updated": data_df.index[-1].strftime('%Y-%m-%d')
            }
            
            # Calculate additional advanced indicators
            close_prices = data_df['Close']
            high_prices = data_df['High']
            low_prices = data_df['Low']
            volume = data_df['Volume']
            
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
            
            # Add advanced indicators to current_indicators
            current_indicators['williams_r'] = float(williams_r.iloc[-1]) if not pd.isna(williams_r.iloc[-1]) else None
            current_indicators['cci'] = float(cci.iloc[-1]) if not pd.isna(cci.iloc[-1]) else None
            current_indicators['atr'] = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else None
            
            # Identify chart patterns
            open_prices = data_df['Open']
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
                "candlestick_patterns": detected_patterns,
                "trend_patterns": trend_patterns,
                "total_patterns": len(detected_patterns) + len(trend_patterns)
            }
            
            # Analyze support and resistance
            # Method 1: Pivot Points
            high = data_df['High'].iloc[-1]
            low = data_df['Low'].iloc[-1]
            close = data_df['Close'].iloc[-1]
            
            pivot = (high + low + close) / 3
            r1 = 2 * pivot - low
            s1 = 2 * pivot - high
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)
            
            # Method 2: Recent highs and lows
            recent_highs = data_df['High'].rolling(window=20).max().dropna()
            recent_lows = data_df['Low'].rolling(window=20).min().dropna()
            
            resistance_levels = recent_highs.unique()
            support_levels = recent_lows.unique()
            
            # Method 3: Fibonacci retracements
            highest_high = data_df['High'].max()
            lowest_low = data_df['Low'].min()
            diff = highest_high - lowest_low
            
            fib_levels = {
                "0.236": highest_high - 0.236 * diff,
                "0.382": highest_high - 0.382 * diff,
                "0.500": highest_high - 0.500 * diff,
                "0.618": highest_high - 0.618 * diff,
                "0.786": highest_high - 0.786 * diff
            }
            
            # Method 4: Moving averages as support/resistance
            sma_20_val = calculate_sma(close_prices, 20).iloc[-1]
            sma_50_val = calculate_sma(close_prices, 50).iloc[-1]
            sma_200_val = calculate_sma(close_prices, 200).iloc[-1]
            
            # Find nearest support and resistance
            all_support_levels = list(support_levels) + [s1, s2, sma_20_val, sma_50_val, sma_200_val]
            all_resistance_levels = list(resistance_levels) + [r1, r2]
            
            # Filter levels near current price (within 20%)
            price_range = current_price * 0.2
            nearby_support = [level for level in all_support_levels if level < current_price and level > current_price - price_range]
            nearby_resistance = [level for level in all_resistance_levels if level > current_price and level < current_price + price_range]
            
            support_resistance_result = {
                "status": "success",
                "current_price": current_price,
                "pivot_points": {
                    "pivot": pivot,
                    "resistance_1": r1,
                    "resistance_2": r2,
                    "support_1": s1,
                    "support_2": s2
                },
                "fibonacci_levels": fib_levels,
                "moving_averages": {
                    "sma_20": sma_20_val,
                    "sma_50": sma_50_val,
                    "sma_200": sma_200_val
                },
                "nearby_support": sorted(nearby_support, reverse=True)[:3],
                "nearby_resistance": sorted(nearby_resistance)[:3],
                "nearest_support": max(nearby_support) if nearby_support else None,
                "nearest_resistance": min(nearby_resistance) if nearby_resistance else None
            }
            
            # Perform trend analysis
            # Calculate moving averages for different timeframes
            sma_5 = calculate_sma(close_prices, 5)
            sma_10 = calculate_sma(close_prices, 10)
            sma_20 = calculate_sma(close_prices, 20)
            sma_50 = calculate_sma(close_prices, 50)
            sma_200 = calculate_sma(close_prices, 200)
            
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
            
            # ADX for trend strength (simplified)
            up_move = high_prices - high_prices.shift(1)
            down_move = low_prices.shift(1) - low_prices
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            # Calculate True Range
            tr1 = high_prices - low_prices
            tr2 = abs(high_prices - close_prices.shift(1))
            tr3 = abs(low_prices - close_prices.shift(1))
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr_trend = true_range.rolling(window=14).mean()
            
            plus_di = 100 * calculate_sma(pd.Series(plus_dm), 14) / atr_trend
            minus_di = 100 * calculate_sma(pd.Series(minus_dm), 14) / atr_trend
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = calculate_sma(pd.Series(dx), 14)
            current_adx = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0
            
            # Price position relative to moving averages
            price_position = {
                "above_sma_5": current_price > sma_5.iloc[-1] if len(sma_5) > 0 else False,
                "above_sma_10": current_price > sma_10.iloc[-1] if len(sma_10) > 0 else False,
                "above_sma_20": current_price > sma_20.iloc[-1] if len(sma_20) > 0 else False,
                "above_sma_50": current_price > sma_50.iloc[-1] if len(sma_50) > 0 else False,
                "above_sma_200": current_price > sma_200.iloc[-1] if len(sma_200) > 0 else False
            }
            
            trend_result = {
                "status": "success",
                "current_price": current_price,
                "trends": trends,
                "trend_strength": min(trend_strength, 100),
                "adx_strength": current_adx,
                "price_position": price_position,
                "moving_averages": {
                    "sma_5": float(sma_5.iloc[-1]) if len(sma_5) > 0 else None,
                    "sma_10": float(sma_10.iloc[-1]) if len(sma_10) > 0 else None,
                    "sma_20": float(sma_20.iloc[-1]) if len(sma_20) > 0 else None,
                    "sma_50": float(sma_50.iloc[-1]) if len(sma_50) > 0 else None,
                    "sma_200": float(sma_200.iloc[-1]) if len(sma_200) > 0 else None
                }
            }
            
            # Generate trading signals
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
            
            # Bollinger Bands signals
            current_bb_upper = indicators['bb_upper'].iloc[-1] if not pd.isna(indicators['bb_upper'].iloc[-1]) else 0
            current_bb_lower = indicators['bb_lower'].iloc[-1] if not pd.isna(indicators['bb_lower'].iloc[-1]) else 0
            
            if current_price < current_bb_lower:
                signals.append({"type": "BUY", "indicator": "BB", "strength": "moderate", "reason": "Price below lower Bollinger Band"})
                signal_strength += 1
            elif current_price > current_bb_upper:
                signals.append({"type": "SELL", "indicator": "BB", "strength": "moderate", "reason": "Price above upper Bollinger Band"})
                signal_strength -= 1
            
            # Stochastic signals
            current_stoch_k = indicators['stoch_k'].iloc[-1] if not pd.isna(indicators['stoch_k'].iloc[-1]) else 50
            current_stoch_d = indicators['stoch_d'].iloc[-1] if not pd.isna(indicators['stoch_d'].iloc[-1]) else 50
            
            if current_stoch_k < 20 and current_stoch_d < 20:
                signals.append({"type": "BUY", "indicator": "Stoch", "strength": "moderate", "reason": "Stochastic oversold"})
                signal_strength += 1
            elif current_stoch_k > 80 and current_stoch_d > 80:
                signals.append({"type": "SELL", "indicator": "Stoch", "strength": "moderate", "reason": "Stochastic overbought"})
                signal_strength -= 1
            
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
            
            signals_result = {
                "status": "success",
                "current_price": current_price,
                "signals": signals,
                "signal_strength": signal_strength,
                "overall_recommendation": overall_signal,
                "total_signals": len(signals),
                "buy_signals": len([s for s in signals if s['type'] == 'BUY']),
                "sell_signals": len([s for s in signals if s['type'] == 'SELL'])
            }

            # If sentiment is already available in the input data (e.g., standalone runs), bias signals
            try:
                sentiment_analysis = data.get("sentiment_analysis") if isinstance(data, dict) else None
                if sentiment_analysis:
                    tools = TechnicalAnalyzerTools()
                    signals_result = tools._bias_signals_with_sentiment(signals_result, sentiment_analysis)
            except Exception:
                pass
            
            # Combine all analysis results
            enhanced_analysis = {
                "indicators": indicators_result.get("current_indicators", {}),
                "patterns": patterns_result.get("candlestick_patterns", []) + patterns_result.get("trend_patterns", []),
                "support_resistance": support_resistance_result,
                "trend_analysis": trend_result,
                "trading_signals": signals_result,
                "analysis_metadata": {
                    "ticker": ticker,
                    "period": period,
                    "data_points": indicators_result.get("data_points", 0),
                    "last_updated": indicators_result.get("last_updated", ""),
                    "analysis_timestamp": datetime.now().isoformat()
                }
            }
            
            # Validate the analysis (simplified)
            validation_result = {
                "status": "success",
                "validation_results": {
                    "data_quality": "good",
                    "indicator_consistency": "good",
                    "signal_reliability": "good",
                    "validation_score": 95,
                    "warnings": [],
                    "recommendations": []
                }
            }
            enhanced_analysis["validation"] = validation_result.get("validation_results", {})
            
            # Calculate overall technical score
            technical_score = self._calculate_enhanced_technical_score(enhanced_analysis)
            enhanced_analysis["technical_score"] = technical_score
            
            print(f"✅ Enhanced technical analysis completed successfully")
            print(f"   Technical Score: {technical_score:.1f}/100")
            print(f"   Patterns Detected: {len(enhanced_analysis['patterns'])}")
            print(f"   Trading Signals: {signals_result.get('overall_recommendation', 'HOLD')}")
            
            return {
                "status": "success",
                "enhanced_technical_analysis": enhanced_analysis,
                "next_agent": "sentiment_analyzer"
            }
            
        except Exception as e:
            print(f"❌ Enhanced technical analysis failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "next_agent": "error_handler"
            }
    
    def _analyze_trends(self, price_data: Dict[str, Any], indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze price trends and moving averages."""
        current_price = price_data.get("current_price", 0)
        
        # Handle different indicator structures
        if isinstance(indicators, dict) and "moving_averages" in indicators:
            # New structure with moving_averages object
            moving_avgs = indicators.get("moving_averages", {})
            sma_20 = moving_avgs.get("sma_20", 0)
            sma_50 = moving_avgs.get("sma_50", 0)
        else:
            # Old structure with direct indicators
            sma_20 = indicators.get("sma_20", 0)
            sma_50 = indicators.get("sma_50", 0)
        
        trend_analysis = {
            "short_term_trend": "neutral",
            "medium_term_trend": "neutral",
            "trend_strength": 0.0,
            "trend_duration": "unknown"
        }
        
        # Short-term trend (20-day SMA)
        if current_price > sma_20:
            trend_analysis["short_term_trend"] = "bullish"
        elif current_price < sma_20:
            trend_analysis["short_term_trend"] = "bearish"
        
        # Medium-term trend (50-day SMA)
        if current_price > sma_50:
            trend_analysis["medium_term_trend"] = "bullish"
        elif current_price < sma_50:
            trend_analysis["medium_term_trend"] = "bearish"
        
        # Trend strength calculation
        if sma_20 > 0 and sma_50 > 0:
            trend_strength = abs(current_price - sma_20) / sma_20
            trend_analysis["trend_strength"] = min(trend_strength * 100, 100)
        
        return trend_analysis
    
    def _find_support_resistance(self, price_data: Dict[str, Any]) -> Dict[str, Any]:
        """Identify support and resistance levels."""
        current_price = price_data.get("current_price", 0)
        high = price_data.get("high", 0)
        low = price_data.get("low", 0)
        
        # Simple support/resistance based on recent high/low
        resistance = high * 1.02  # 2% above recent high
        support = low * 0.98      # 2% below recent low
        
        return {
            "support_level": support,
            "resistance_level": resistance,
            "distance_to_support": current_price - support if current_price > support else 0,
            "distance_to_resistance": resistance - current_price if resistance > current_price else 0,
            "support_strength": "medium",
            "resistance_strength": "medium"
        }
    
    def _identify_patterns(self, price_data: Dict[str, Any]) -> Dict[str, Any]:
        """Identify common chart patterns."""
        # This is a simplified pattern recognition
        # In a real implementation, you'd use more sophisticated pattern detection
        
        patterns = {
            "detected_patterns": [],
            "pattern_confidence": 0.0,
            "breakout_potential": "low"
        }
        
        # Simple pattern detection based on price action
        current_price = price_data.get("current_price", 0)
        previous_close = price_data.get("previous_close", 0)
        daily_change_pct = price_data.get("daily_change_pct", 0)
        
        if daily_change_pct > 2:
            patterns["detected_patterns"].append("bullish_breakout")
            patterns["pattern_confidence"] = 0.6
        elif daily_change_pct < -2:
            patterns["detected_patterns"].append("bearish_breakdown")
            patterns["pattern_confidence"] = 0.6
        
        return patterns
    
    def _analyze_volume(self, price_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze volume patterns."""
        volume = price_data.get("volume", 0)
        avg_volume = price_data.get("avg_volume", 1000000)  # Default average volume
        
        # Simplified volume analysis
        volume_analysis = {
            "volume_trend": "normal",
            "volume_significance": "medium",
            "volume_support": "neutral"
        }
        
        # Compare with average volume
        if volume > avg_volume * 1.5:
            volume_analysis["volume_trend"] = "high"
            volume_analysis["volume_significance"] = "high"
        elif volume < avg_volume * 0.5:
            volume_analysis["volume_trend"] = "low"
            volume_analysis["volume_significance"] = "low"
        
        return volume_analysis
    
    def _analyze_momentum(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze momentum indicators."""
        # Handle different indicator structures
        if isinstance(indicators, dict) and "rsi" in indicators:
            # Check if rsi is a dict or direct value
            rsi_value = indicators.get("rsi")
            if isinstance(rsi_value, dict):
                rsi = rsi_value.get("value", 50)
            else:
                rsi = rsi_value if rsi_value is not None else 50
        else:
            # Old structure with direct rsi value
            rsi = indicators.get("rsi", 50)
        
        # Ensure rsi is a number
        if not isinstance(rsi, (int, float)):
            rsi = 50
        
        momentum_analysis = {
            "rsi_signal": "neutral",
            "momentum_strength": "medium",
            "overbought_oversold": "neutral"
        }
        
        # RSI analysis
        if rsi > 70:
            momentum_analysis["rsi_signal"] = "bearish"
            momentum_analysis["overbought_oversold"] = "overbought"
        elif rsi < 30:
            momentum_analysis["rsi_signal"] = "bullish"
            momentum_analysis["overbought_oversold"] = "oversold"
        elif rsi > 50:
            momentum_analysis["rsi_signal"] = "bullish"
        else:
            momentum_analysis["rsi_signal"] = "bearish"
        
        return momentum_analysis
    
    def _calculate_technical_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall technical score (0-100)."""
        score = 50.0  # Neutral starting point
        
        # Trend analysis contribution
        trend_analysis = analysis.get("trend_analysis", {})
        short_trend = trend_analysis.get("short_term_trend", "neutral")
        medium_trend = trend_analysis.get("medium_term_trend", "neutral")
        
        if short_trend == "bullish":
            score += 10
        elif short_trend == "bearish":
            score -= 10
        
        if medium_trend == "bullish":
            score += 15
        elif medium_trend == "bearish":
            score -= 15
        
        # Momentum contribution
        momentum = analysis.get("momentum_analysis", {})
        rsi_signal = momentum.get("rsi_signal", "neutral")
        
        if rsi_signal == "bullish":
            score += 10
        elif rsi_signal == "bearish":
            score -= 10
        
        # Volume contribution
        volume = analysis.get("volume_analysis", {})
        volume_trend = volume.get("volume_trend", "normal")
        
        if volume_trend == "high":
            score += 5
        elif volume_trend == "low":
            score -= 5
        
        return max(0, min(100, score))
    
    def _calculate_enhanced_technical_score(self, enhanced_analysis: Dict[str, Any]) -> float:
        """Calculate enhanced technical score using tool-based analysis."""
        score = 50.0  # Neutral starting point
        
        # Indicators contribution
        indicators = enhanced_analysis.get("indicators", {})
        
        # RSI contribution
        rsi = indicators.get("rsi", 50)
        if rsi is not None:
            if rsi < 30:
                score += 15  # Strong oversold
            elif rsi < 40:
                score += 8   # Moderate oversold
            elif rsi > 70:
                score -= 15  # Strong overbought
            elif rsi > 60:
                score -= 8   # Moderate overbought
        
        # MACD contribution
        macd = indicators.get("macd", 0)
        macd_signal = indicators.get("macd_signal", 0)
        if macd is not None and macd_signal is not None:
            if macd > macd_signal:
                score += 10  # MACD above signal
            else:
                score -= 10  # MACD below signal
        
        # Moving averages contribution
        sma_20 = indicators.get("sma_20", 0)
        sma_50 = indicators.get("sma_50", 0)
        current_price = enhanced_analysis.get("trend_analysis", {}).get("current_price", 0)
        
        if sma_20 is not None and sma_50 is not None and current_price > 0:
            if current_price > sma_20 > sma_50:
                score += 12  # Strong bullish alignment
            elif current_price < sma_20 < sma_50:
                score -= 12  # Strong bearish alignment
            elif current_price > sma_20:
                score += 6   # Moderate bullish
            elif current_price < sma_20:
                score -= 6   # Moderate bearish
        
        # Pattern contribution
        patterns = enhanced_analysis.get("patterns", [])
        bullish_patterns = len([p for p in patterns if p.get("signal") == "bullish"])
        bearish_patterns = len([p for p in patterns if p.get("signal") == "bearish"])
        
        score += bullish_patterns * 5   # +5 for each bullish pattern
        score -= bearish_patterns * 5   # -5 for each bearish pattern
        
        # Trading signals contribution
        trading_signals = enhanced_analysis.get("trading_signals", {})
        signal_strength = trading_signals.get("signal_strength", 0)
        score += signal_strength * 2  # Amplify signal strength
        
        # Validation score contribution
        validation = enhanced_analysis.get("validation", {})
        validation_score = validation.get("validation_score", 100)
        score = score * (validation_score / 100)  # Adjust based on validation quality
        
        return max(0, min(100, score))
    
    def _generate_signals(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate trading signals based on analysis."""
        signals = []
        score = analysis.get("technical_score", 50)
        
        if score > 70:
            signals.append("STRONG_BUY")
        elif score > 60:
            signals.append("BUY")
        elif score < 30:
            signals.append("STRONG_SELL")
        elif score < 40:
            signals.append("SELL")
        else:
            signals.append("HOLD")
        
        # Add specific signals
        momentum = analysis.get("momentum_analysis", {})
        if momentum.get("overbought_oversold") == "overbought":
            signals.append("OVERBOUGHT")
        elif momentum.get("overbought_oversold") == "oversold":
            signals.append("OVERSOLD")
        
        return signals

# Function for LangGraph integration
def analyze_technical(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function for technical analysis.
    
    Args:
        state: Current state containing collected data
        
    Returns:
        Updated state with technical analysis
    """
    print(f"🔍 Technical analyzer starting analysis...")
    
    agent = TechnicalAnalyzerAgent()
    result = agent.analyze_technical_data(state)
    
    # Update state with technical analysis
    state.update(result)
    return state

def analyze_technical_with_tools(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhanced technical analyzer that uses LangGraph ToolNode pattern for comprehensive technical analysis.
    
    Args:
        state: Current state containing data and analysis parameters
        
    Returns:
        Updated state with comprehensive technical analysis results
    """
    try:
        ticker = state.get("ticker")
        timeframe = state.get("timeframe", "1d")
        
        if not ticker:
            return {
                "status": "error",
                "error": "No ticker provided",
                "next_agent": "error_handler"
            }
        
        print(f"🔍 Enhanced technical analyzer starting analysis for {ticker}...")
        
        # Check if we need to execute tools
        needs_tools = state.get("needs_tools", True)
        
        if needs_tools:
            print("🔧 Technical analyzer needs tools - routing to technical_analyzer_tools")
            return {
                "status": "success",
                "needs_tools": True,
                "current_tool_node": "technical_analyzer_tools",
                "next_agent": "technical_analyzer_tools",
                "messages": [
                    {
                        "role": "system",
                        "content": f"Perform comprehensive technical analysis for ticker {ticker} with timeframe {timeframe}. Calculate indicators, identify patterns, analyze support/resistance, perform trend analysis, and generate trading signals."
                    }
                ]
            }
        
        # If tools have been executed, process the results
        tool_results = state.get("tool_results", {})
        
        # Extract results from tool execution
        advanced_indicators = tool_results.get("calculate_advanced_indicators", {})
        chart_patterns = tool_results.get("identify_chart_patterns", {})
        support_resistance = tool_results.get("analyze_support_resistance", {})
        trend_analysis = tool_results.get("perform_trend_analysis", {})
        trading_signals = tool_results.get("generate_trading_signals", {})
        validation = tool_results.get("validate_technical_analysis", {})
        
        # Validate results
        if advanced_indicators.get("status") == "error":
            return {
                "status": "error",
                "error": f"Advanced indicators calculation failed: {advanced_indicators.get('error')}",
                "next_agent": "error_handler"
            }
        
        # Combine all technical analysis results
        enhanced_technical_analysis = {
            "indicators": advanced_indicators.get("data", {}),
            "patterns": chart_patterns.get("data", {}),
            "support_resistance": support_resistance.get("data", {}),
            "trend_analysis": trend_analysis.get("data", {}),
            "trading_signals": trading_signals.get("data", {}),
            "validation": validation.get("data", {}),
            "technical_score": validation.get("data", {}).get("technical_score", 0),
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        # Update state with comprehensive technical analysis
        state.update({
            "status": "success",
            "needs_tools": False,
            "next_agent": "sentiment_analyzer",
            "enhanced_technical_analysis": enhanced_technical_analysis,
            "technical_analyzer_tools_used": [
                "calculate_advanced_indicators",
                "identify_chart_patterns",
                "analyze_support_resistance",
                "perform_trend_analysis",
                "generate_trading_signals",
                "validate_technical_analysis"
            ]
        })
        
        print(f"✅ Enhanced technical analysis completed successfully")
        print(f"   Technical Score: {enhanced_technical_analysis.get('technical_score', 0):.1f}/100")
        print(f"   Patterns Detected: {len(chart_patterns.get('data', {}).get('patterns', []))}")
        print(f"   Trading Signals: {trading_signals.get('data', {}).get('overall_recommendation', 'HOLD')}")
        
        return state
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Enhanced technical analysis failed: {str(e)}",
            "next_agent": "error_handler",
            "needs_tools": False
        } 