# agents/technical_analyzer_tools.py
"""
Technical Analyzer Tools for Stock Prediction Workflow

This module provides tools for the technical analyzer agent to:
- Calculate advanced technical indicators
- Identify chart patterns and formations
- Analyze support and resistance levels
- Perform trend analysis and forecasting
- Generate trading signals
- Validate technical analysis results
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import yfinance as yf
from langchain_core.tools import tool
import warnings
warnings.filterwarnings('ignore')
import os
import pickle

# Simple in-memory cache for price data within a single run
_INMEM_PRICE_CACHE: Dict[Tuple[str, str], Tuple[float, pd.DataFrame]] = {}
_PRICE_TTL_SECONDS = 10 * 60  # 10 minutes
_PRICE_CACHE_DIR = os.path.join("cache", "technical_analyzer")
os.makedirs(_PRICE_CACHE_DIR, exist_ok=True)

# Simplified technical analysis functions without TA-Lib
def calculate_sma(data: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return data.rolling(window=period).mean()

def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return data.ewm(span=period).mean()

def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD."""
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands."""
    sma = calculate_sma(data, period)
    std = data.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series]:
    """Calculate Stochastic Oscillator."""
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = calculate_sma(k_percent, 3)
    return k_percent, d_percent

class TechnicalAnalyzerTools:
    """
    Tools for the technical analyzer agent to perform comprehensive technical analysis.
    """
    
    def __init__(self):
        """Initialize technical analyzer tools."""
        self.pattern_functions = {}  # Simplified without TA-Lib candlestick patterns
    
    def _get_price_data(self, ticker: str, period: str = "6mo") -> pd.DataFrame:
        """Get price data with caching and a strict fetch timeout.

        - Checks in-memory and disk cache first
        - Falls back to expired disk cache if live fetch times out
        - Timeout configurable via STOCK4U_TA_TIMEOUT_FETCH (seconds, default 8)
        """
        try:
            now = datetime.now().timestamp()
            key = (ticker, period)

            # In-memory cache
            if key in _INMEM_PRICE_CACHE:
                ts, df = _INMEM_PRICE_CACHE[key]
                if now - ts <= _PRICE_TTL_SECONDS and isinstance(df, pd.DataFrame) and not df.empty:
                    return df

            # Disk cache
            disk_key = f"price_{ticker}_{period}.pkl"
            disk_path = os.path.join(_PRICE_CACHE_DIR, disk_key)
            expired_df: Optional[pd.DataFrame] = None
            if os.path.exists(disk_path):
                try:
                    with open(disk_path, "rb") as f:
                        payload = pickle.load(f)
                    if now - payload.get("_ts", 0) <= _PRICE_TTL_SECONDS:
                        df = payload.get("data")
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            _INMEM_PRICE_CACHE[key] = (now, df)
                            return df
                    else:
                        # keep expired cache as fallback
                        expired_df = payload.get("data") if isinstance(payload.get("data"), pd.DataFrame) else None
                except Exception:
                    pass

            # Fetch fresh
            import concurrent.futures as _f
            # Increase timeout for cloud environments
            default_timeout = "15" if os.getenv("STOCK4U_CLOUD") == "1" else "8"
            timeout_s = int(os.getenv("STOCK4U_TA_TIMEOUT_FETCH", default_timeout))
            def _fetch():
                return yf.Ticker(ticker).history(period=period)
            with _f.ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(_fetch)
                try:
                    data = fut.result(timeout=timeout_s)
                except Exception as e:
                    print(f"Warning: Data fetch timeout for {ticker} ({period}): {e}")
                    data = expired_df if isinstance(expired_df, pd.DataFrame) else pd.DataFrame()

            # Write caches (best-effort)
            if isinstance(data, pd.DataFrame) and not data.empty:
                _INMEM_PRICE_CACHE[key] = (now, data)
                try:
                    with open(disk_path, "wb") as f:
                        pickle.dump({"_ts": now, "data": data}, f)
                except Exception:
                    pass

            return data
        except Exception as e:
            raise Exception(f"Failed to get price data: {str(e)}")
    
    def _calculate_basic_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic technical indicators."""
        indicators = {}
        
        # Moving averages
        indicators['sma_20'] = calculate_sma(data['Close'], 20)
        indicators['sma_50'] = calculate_sma(data['Close'], 50)
        indicators['sma_200'] = calculate_sma(data['Close'], 200)
        indicators['ema_12'] = calculate_ema(data['Close'], 12)
        indicators['ema_26'] = calculate_ema(data['Close'], 26)
        
        # MACD
        macd, macd_signal, macd_hist = calculate_macd(data['Close'])
        indicators['macd'] = macd
        indicators['macd_signal'] = macd_signal
        indicators['macd_histogram'] = macd_hist
        
        # RSI
        indicators['rsi'] = calculate_rsi(data['Close'], 14)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(data['Close'])
        indicators['bb_upper'] = bb_upper
        indicators['bb_middle'] = bb_middle
        indicators['bb_lower'] = bb_lower
        
        # Stochastic
        slowk, slowd = calculate_stochastic(data['High'], data['Low'], data['Close'])
        indicators['stoch_k'] = slowk
        indicators['stoch_d'] = slowd
        
        # Volume indicators (simplified)
        indicators['obv'] = data['Volume'].cumsum()  # Simplified OBV
        indicators['ad'] = data['Volume']  # Simplified AD
        
        return indicators

    def _bias_signals_with_sentiment(self, signals_payload: Dict[str, Any], sentiment_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Lightweight re-weighting of trading signals by sentiment strength and confidence."""
        try:
            if not isinstance(signals_payload, dict):
                return signals_payload
            overall = signals_payload.get("overall_recommendation", "HOLD")
            s = float((sentiment_analysis.get("overall_sentiment", {}) or {}).get("sentiment_score", 0.0) or 0.0)
            conf = float((sentiment_analysis.get("overall_sentiment", {}) or {}).get("confidence", 0.0) or 0.0)
            strength = float(signals_payload.get("signal_strength", 0) or 0)
            # Apply a modest bias up to +/-1 based on abs(sentiment)*confidence
            bias = min(1.0, max(-1.0, s * (0.5 + 0.5 * min(1.0, conf))))
            adjusted_strength = strength + bias
            signals_payload["signal_strength"] = adjusted_strength
            # If confidence and sentiment are strong, upgrade/downgrade overall recommendation by one
            if abs(s) > 0.4 and conf > 0.6:
                if s > 0 and overall in ("BUY", "HOLD"):
                    signals_payload["overall_recommendation"] = "STRONG_BUY" if overall == "BUY" else "BUY"
                if s < 0 and overall in ("SELL", "HOLD"):
                    signals_payload["overall_recommendation"] = "STRONG_SELL" if overall == "SELL" else "SELL"
            return signals_payload
        except Exception:
            return signals_payload

@tool
def calculate_advanced_indicators(ticker: str, period: str = "6mo") -> Dict[str, Any]:
    """
    Calculate comprehensive technical indicators for a stock.
    
    Args:
        ticker: Stock ticker symbol
        period: Data period for analysis
        
    Returns:
        Dictionary with all calculated technical indicators
    """
    try:
        tools = TechnicalAnalyzerTools()
        data = tools._get_price_data(ticker, period)
        
        if data.empty:
            return {
                "status": "error",
                "error": "No data available for analysis"
            }
        
        # Calculate basic indicators
        indicators = tools._calculate_basic_indicators(data)
        
        # Calculate additional advanced indicators (simplified)
        close_prices = data['Close']
        high_prices = data['High']
        low_prices = data['Low']
        volume = data['Volume']
        
        # Williams %R (simplified)
        lowest_low = low_prices.rolling(window=14).min()
        highest_high = high_prices.rolling(window=14).max()
        williams_r = -100 * ((highest_high - close_prices) / (highest_high - lowest_low))
        indicators['williams_r'] = williams_r
        
        # CCI (Commodity Channel Index) - simplified
        typical_price = (high_prices + low_prices + close_prices) / 3
        sma_tp = calculate_sma(typical_price, 14)
        mean_deviation = abs(typical_price - sma_tp).rolling(window=14).mean()
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        indicators['cci'] = cci
        
        # ATR (Average True Range) - simplified
        tr1 = high_prices - low_prices
        tr2 = abs(high_prices - close_prices.shift(1))
        tr3 = abs(low_prices - close_prices.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = calculate_sma(true_range, 14)
        indicators['atr'] = atr
        
        # Simplified ADX (using trend strength)
        up_move = high_prices - high_prices.shift(1)
        down_move = low_prices.shift(1) - low_prices
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        plus_di = 100 * calculate_sma(pd.Series(plus_dm), 14) / atr
        minus_di = 100 * calculate_sma(pd.Series(minus_dm), 14) / atr
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = calculate_sma(pd.Series(dx), 14)
        indicators['adx'] = adx
        
        # Simplified Parabolic SAR
        sar = close_prices.copy()
        sar.iloc[0] = low_prices.iloc[0]
        for i in range(1, len(sar)):
            if close_prices.iloc[i] > sar.iloc[i-1]:
                sar.iloc[i] = min(sar.iloc[i-1], low_prices.iloc[i-1])
            else:
                sar.iloc[i] = max(sar.iloc[i-1], high_prices.iloc[i-1])
        indicators['sar'] = sar
        
        # Get current values (use .iloc for Series-safe indexing)
        current_indicators = {}
        for key, series in indicators.items():
            try:
                last_val = series.iloc[-1]
                current_indicators[key] = float(last_val) if not pd.isna(last_val) else None
            except Exception:
                current_indicators[key] = None
        
        return {
            "status": "success",
            "ticker": ticker,
            "period": period,
            "current_indicators": current_indicators,
            "data_points": len(data),
            "last_updated": data.index[-1].strftime('%Y-%m-%d')
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Failed to calculate indicators: {str(e)}"
        }

@tool
def calculate_short_term_indicators(ticker: str, period: str = "3mo") -> Dict[str, Any]:
    """
    Calculate short-term indicators tuned for 1-day horizon.

    Returns a compact set of fast-reacting indicators useful for daily decisions.
    """
    try:
        tools = TechnicalAnalyzerTools()
        data = tools._get_price_data(ticker, period)

        if data.empty:
            return {"status": "error", "error": "No data available"}

        close = data["Close"]
        high = data["High"]
        low = data["Low"]
        volume = data["Volume"]

        # RSI variants
        rsi_2 = calculate_rsi(close, 2)
        rsi_7 = calculate_rsi(close, 7)

        # EMAs
        ema_9 = calculate_ema(close, 9)
        ema_21 = calculate_ema(close, 21)

        # Rate of Change (ROC)
        roc_1 = close.pct_change(1) * 100
        roc_5 = close.pct_change(5) * 100

        # Money Flow Index (MFI)
        typical = (high + low + close) / 3
        raw_money = typical * volume
        positive_flow = raw_money.where(typical > typical.shift(1), 0.0)
        negative_flow = raw_money.where(typical < typical.shift(1), 0.0)
        pos_sum = positive_flow.rolling(14).sum()
        neg_sum = negative_flow.rolling(14).sum().replace(0, 1e-9)
        mfr = pos_sum / neg_sum
        mfi_14 = 100 - (100 / (1 + mfr))

        # Rolling VWAP (20 periods)
        vwap_num = (typical * volume).rolling(20).sum()
        vwap_den = volume.rolling(20).sum().replace(0, 1e-9)
        vwap_20 = vwap_num / vwap_den

        current = {
            "rsi_2": float(rsi_2.iloc[-1]) if not pd.isna(rsi_2.iloc[-1]) else None,
            "rsi_7": float(rsi_7.iloc[-1]) if not pd.isna(rsi_7.iloc[-1]) else None,
            "ema_9": float(ema_9.iloc[-1]) if not pd.isna(ema_9.iloc[-1]) else None,
            "ema_21": float(ema_21.iloc[-1]) if not pd.isna(ema_21.iloc[-1]) else None,
            "roc_1d": float(roc_1.iloc[-1]) if not pd.isna(roc_1.iloc[-1]) else None,
            "roc_5d": float(roc_5.iloc[-1]) if not pd.isna(roc_5.iloc[-1]) else None,
            "mfi_14": float(mfi_14.iloc[-1]) if not pd.isna(mfi_14.iloc[-1]) else None,
            "vwap_20": float(vwap_20.iloc[-1]) if not pd.isna(vwap_20.iloc[-1]) else None,
        }

        return {
            "status": "success",
            "ticker": ticker,
            "period": period,
            "current_indicators": current,
        }
    except Exception as e:
        return {"status": "error", "error": f"Failed to calculate short-term indicators: {str(e)}"}

@tool
def compute_supertrend(ticker: str, period: str = "6mo", atr_period: int = 10, multiplier: float = 3.0) -> Dict[str, Any]:
    """
    Compute a simplified Supertrend indicator and current trend.
    """
    try:
        tools = TechnicalAnalyzerTools()
        data = tools._get_price_data(ticker, period)
        if data.empty:
            return {"status": "error", "error": "No data available"}

        high = data["High"]
        low = data["Low"]
        close = data["Close"]

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(atr_period).mean()

        hl2 = (high + low) / 2
        upper_band = hl2 + multiplier * atr
        lower_band = hl2 - multiplier * atr

        # Simplified trend decision using latest bands
        last_close = close.iloc[-1]
        last_upper = upper_band.iloc[-1]
        last_lower = lower_band.iloc[-1]
        trend = "bullish" if last_close > last_upper else ("bearish" if last_close < last_lower else "neutral")

        return {
            "status": "success",
            "ticker": ticker,
            "current_indicators": {
                "supertrend_upper": float(last_upper) if not pd.isna(last_upper) else None,
                "supertrend_lower": float(last_lower) if not pd.isna(last_lower) else None,
                "supertrend_trend": trend,
            },
        }
    except Exception as e:
        return {"status": "error", "error": f"Failed to compute Supertrend: {str(e)}"}

@tool
def compute_ichimoku_cloud(ticker: str, period: str = "6mo") -> Dict[str, Any]:
    """
    Compute Ichimoku Cloud components and a basic signal.
    """
    try:
        tools = TechnicalAnalyzerTools()
        data = tools._get_price_data(ticker, period)
        if data.empty:
            return {"status": "error", "error": "No data available"}

        high = data["High"]
        low = data["Low"]
        close = data["Close"]

        conv = (high.rolling(9).max() + low.rolling(9).min()) / 2
        base = (high.rolling(26).max() + low.rolling(26).min()) / 2
        span_a = ((conv + base) / 2)
        span_b = (high.rolling(52).max() + low.rolling(52).min()) / 2

        last_close = close.iloc[-1]
        last_span_a = span_a.iloc[-1]
        last_span_b = span_b.iloc[-1]
        cloud_top = max(last_span_a, last_span_b)
        cloud_bottom = min(last_span_a, last_span_b)
        signal = "bullish" if last_close > cloud_top else ("bearish" if last_close < cloud_bottom else "neutral")

        return {
            "status": "success",
            "ticker": ticker,
            "current_indicators": {
                "ichimoku_conversion": float(conv.iloc[-1]) if not pd.isna(conv.iloc[-1]) else None,
                "ichimoku_base": float(base.iloc[-1]) if not pd.isna(base.iloc[-1]) else None,
                "ichimoku_span_a": float(last_span_a) if not pd.isna(last_span_a) else None,
                "ichimoku_span_b": float(last_span_b) if not pd.isna(last_span_b) else None,
                "ichimoku_signal": signal,
            },
        }
    except Exception as e:
        return {"status": "error", "error": f"Failed to compute Ichimoku: {str(e)}"}

@tool
def compute_keltner_channels(ticker: str, period: str = "6mo", ema_period: int = 20, atr_period: int = 10, multiplier: float = 2.0) -> Dict[str, Any]:
    """
    Compute Keltner Channels for volatility-based short-term analysis.
    """
    try:
        tools = TechnicalAnalyzerTools()
        data = tools._get_price_data(ticker, period)
        if data.empty:
            return {"status": "error", "error": "No data available"}

        close = data["Close"]
        high = data["High"]
        low = data["Low"]
        ema = calculate_ema(close, ema_period)

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(atr_period).mean()

        upper = ema + multiplier * atr
        lower = ema - multiplier * atr

        return {
            "status": "success",
            "ticker": ticker,
            "current_indicators": {
                "keltner_upper": float(upper.iloc[-1]) if not pd.isna(upper.iloc[-1]) else None,
                "keltner_middle": float(ema.iloc[-1]) if not pd.isna(ema.iloc[-1]) else None,
                "keltner_lower": float(lower.iloc[-1]) if not pd.isna(lower.iloc[-1]) else None,
            },
        }
    except Exception as e:
        return {"status": "error", "error": f"Failed to compute Keltner Channels: {str(e)}"}

@tool
def compute_donchian_channels(ticker: str, period: str = "6mo", window: int = 20) -> Dict[str, Any]:
    """
    Compute Donchian Channels and provide current breakout context.
    """
    try:
        tools = TechnicalAnalyzerTools()
        data = tools._get_price_data(ticker, period)
        if data.empty:
            return {"status": "error", "error": "No data available"}

        high = data["High"].rolling(window).max()
        low = data["Low"].rolling(window).min()
        upper = high
        lower = low
        middle = (upper + lower) / 2
        close = data["Close"]

        return {
            "status": "success",
            "ticker": ticker,
            "current_indicators": {
                "donchian_upper": float(upper.iloc[-1]) if not pd.isna(upper.iloc[-1]) else None,
                "donchian_middle": float(middle.iloc[-1]) if not pd.isna(middle.iloc[-1]) else None,
                "donchian_lower": float(lower.iloc[-1]) if not pd.isna(lower.iloc[-1]) else None,
            },
        }
    except Exception as e:
        return {"status": "error", "error": f"Failed to compute Donchian Channels: {str(e)}"}

@tool
def identify_chart_patterns(ticker: str, period: str = "6mo") -> Dict[str, Any]:
    """
    Identify candlestick patterns and chart formations.
    
    Args:
        ticker: Stock ticker symbol
        period: Data period for analysis
        
    Returns:
        Dictionary with detected patterns and their significance
    """
    try:
        tools = TechnicalAnalyzerTools()
        data = tools._get_price_data(ticker, period)
        
        if data.empty:
            return {
                "status": "error",
                "error": "No data available for pattern analysis"
            }
        
        # Get OHLC data as pandas Series (not numpy). Many downstream
        # helpers use Series methods like .rolling/.shift and .iloc.
        open_prices = data['Open']
        high_prices = data['High']
        low_prices = data['Low']
        close_prices = data['Close']
        
        detected_patterns = []
        
        # Check for candlestick patterns (simplified without TA-Lib)
        # Simple pattern detection based on price action
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
        sma_20 = calculate_sma(close_prices, 20)
        sma_50 = calculate_sma(close_prices, 50)
        
        current_price = float(close_prices.iloc[-1])
        current_sma_20 = float(sma_20.iloc[-1]) if len(sma_20) else None
        current_sma_50 = float(sma_50.iloc[-1]) if len(sma_50) else None
        
        trend_patterns = []
        
        # Golden Cross (SMA 20 crosses above SMA 50)
        if len(sma_20) > 1 and len(sma_50) > 1:
            if sma_20[-1] > sma_50[-1] and sma_20[-2] <= sma_50[-2]:
                trend_patterns.append({
                    "pattern": "Golden Cross",
                    "signal": "bullish",
                    "significance": "strong",
                    "description": "20-day SMA crossed above 50-day SMA"
                })
            
            # Death Cross (SMA 20 crosses below SMA 50)
            elif sma_20[-1] < sma_50[-1] and sma_20[-2] >= sma_50[-2]:
                trend_patterns.append({
                    "pattern": "Death Cross",
                    "signal": "bearish",
                    "significance": "strong",
                    "description": "20-day SMA crossed below 50-day SMA"
                })
        
        return {
            "status": "success",
            "ticker": ticker,
            "candlestick_patterns": detected_patterns,
            "trend_patterns": trend_patterns,
            "total_patterns": len(detected_patterns) + len(trend_patterns),
            "analysis_period": period
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Failed to identify patterns: {str(e)}"
        }

@tool
def analyze_support_resistance(ticker: str, period: str = "6mo") -> Dict[str, Any]:
    """
    Analyze support and resistance levels using multiple methods.
    
    Args:
        ticker: Stock ticker symbol
        period: Data period for analysis
        
    Returns:
        Dictionary with support and resistance levels
    """
    try:
        tools = TechnicalAnalyzerTools()
        data = tools._get_price_data(ticker, period)
        
        if data.empty:
            # Try a more aggressive fallback approach for cloud environments
            if os.getenv("STOCK4U_CLOUD") == "1":
                try:
                    # Try with a shorter period and different approach
                    fallback_data = yf.Ticker(ticker).history(period="1mo")
                    if not fallback_data.empty:
                        data = fallback_data
                    else:
                        # Last resort: try to get current price only
                        stock_info = yf.Ticker(ticker).info
                        current_price = stock_info.get('currentPrice') or stock_info.get('regularMarketPrice')
                        if current_price:
                            return {
                                "status": "partial_success",
                                "ticker": ticker,
                                "current_price": current_price,
                                "nearest_support": current_price * 0.95,  # 5% below current
                                "nearest_resistance": current_price * 1.05,  # 5% above current
                                "error": "Limited data - using estimated support/resistance levels"
                            }
                except Exception:
                    pass
            
            if data.empty:
                return {
                    "status": "error",
                    "error": "No data available for support/resistance analysis"
                }
        
        current_price = data['Close'].iloc[-1]
        high_prices = data['High'].values
        low_prices = data['Low'].values
        close_prices = data['Close'].values
        
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
        sma_20 = calculate_sma(close_prices, 20).iloc[-1]
        sma_50 = calculate_sma(close_prices, 50).iloc[-1]
        sma_200 = calculate_sma(close_prices, 200).iloc[-1]
        
        # Find nearest support and resistance
        all_support_levels = list(support_levels) + [s1, s2, sma_20, sma_50, sma_200]
        all_resistance_levels = list(resistance_levels) + [r1, r2]
        
        # Filter levels near current price (within 20%)
        price_range = current_price * 0.2
        nearby_support = [level for level in all_support_levels if level < current_price and level > current_price - price_range]
        nearby_resistance = [level for level in all_resistance_levels if level > current_price and level < current_price + price_range]
        
        return {
            "status": "success",
            "ticker": ticker,
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
                "sma_20": sma_20,
                "sma_50": sma_50,
                "sma_200": sma_200
            },
            "nearby_support": sorted(nearby_support, reverse=True)[:3],
            "nearby_resistance": sorted(nearby_resistance)[:3],
            "nearest_support": max(nearby_support) if nearby_support else None,
            "nearest_resistance": min(nearby_resistance) if nearby_resistance else None
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Failed to analyze support/resistance: {str(e)}"
        }

@tool
def perform_trend_analysis(ticker: str, period: str = "6mo") -> Dict[str, Any]:
    """
    Perform comprehensive trend analysis using multiple timeframes.
    
    Args:
        ticker: Stock ticker symbol
        period: Data period for analysis
        
    Returns:
        Dictionary with trend analysis results
    """
    try:
        tools = TechnicalAnalyzerTools()
        data = tools._get_price_data(ticker, period)
        
        if data.empty:
            return {
                "status": "error",
                "error": "No data available for trend analysis"
            }
        
        close_prices = data['Close']
        current_price = float(close_prices.iloc[-1])
        
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
            if current_price > sma_10[-1] and sma_10[-1] > sma_10[-5]:
                trends["short_term"] = "bullish"
            elif current_price < sma_10[-1] and sma_10[-1] < sma_10[-5]:
                trends["short_term"] = "bearish"
            else:
                trends["short_term"] = "sideways"
        
        # Medium-term trend (20-50 days)
        if len(sma_20) > 0 and len(sma_50) > 0:
            if current_price > sma_50[-1] and sma_20[-1] > sma_50[-1]:
                trends["medium_term"] = "bullish"
            elif current_price < sma_50[-1] and sma_20[-1] < sma_50[-1]:
                trends["medium_term"] = "bearish"
            else:
                trends["medium_term"] = "sideways"
        
        # Long-term trend (200 days)
        if len(sma_200) > 0:
            if current_price > sma_200[-1]:
                trends["long_term"] = "bullish"
            else:
                trends["long_term"] = "bearish"
        
        # Trend strength calculation
        if len(sma_20) > 0 and len(sma_50) > 0:
            trend_strength = abs(current_price - sma_50[-1]) / sma_50[-1] * 100
        else:
            trend_strength = 0
        
        # ADX for trend strength (simplified)
        high_prices = data['High']
        low_prices = data['Low']
        
        # Simplified ADX calculation
        up_move = high_prices - high_prices.shift(1)
        down_move = low_prices.shift(1) - low_prices
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Calculate True Range
        tr1 = high_prices - low_prices
        tr2 = (high_prices - close_prices.shift(1)).abs()
        tr3 = (low_prices - close_prices.shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = calculate_sma(true_range, 14)
        
        plus_di = 100 * calculate_sma(pd.Series(plus_dm), 14) / atr
        minus_di = 100 * calculate_sma(pd.Series(minus_dm), 14) / atr
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = calculate_sma(pd.Series(dx), 14)
        current_adx = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0
        
        # Price position relative to moving averages
        price_position = {
            "above_sma_5": current_price > float(sma_5.iloc[-1]) if len(sma_5) > 0 else False,
            "above_sma_10": current_price > float(sma_10.iloc[-1]) if len(sma_10) > 0 else False,
            "above_sma_20": current_price > float(sma_20.iloc[-1]) if len(sma_20) > 0 else False,
            "above_sma_50": current_price > float(sma_50.iloc[-1]) if len(sma_50) > 0 else False,
            "above_sma_200": current_price > float(sma_200.iloc[-1]) if len(sma_200) > 0 else False
        }
        
        return {
            "status": "success",
            "ticker": ticker,
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
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Failed to perform trend analysis: {str(e)}"
        }

@tool
def generate_trading_signals(ticker: str, period: str = "6mo") -> Dict[str, Any]:
    """
    Generate comprehensive trading signals based on technical analysis.
    
    Args:
        ticker: Stock ticker symbol
        period: Data period for analysis
        
    Returns:
        Dictionary with trading signals and recommendations
    """
    try:
        tools = TechnicalAnalyzerTools()
        data = tools._get_price_data(ticker, period)
        
        if data.empty:
            return {
                "status": "error",
                "error": "No data available for signal generation"
            }
        
        # Calculate indicators
        indicators = tools._calculate_basic_indicators(data)
        
        current_price = data['Close'].iloc[-1]
        signals = []
        signal_strength = 0
        
        # RSI signals
        current_rsi = indicators['rsi'][-1] if not pd.isna(indicators['rsi'][-1]) else 50
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
        current_macd = indicators['macd'][-1] if not pd.isna(indicators['macd'][-1]) else 0
        current_macd_signal = indicators['macd_signal'][-1] if not pd.isna(indicators['macd_signal'][-1]) else 0
        current_macd_hist = indicators['macd_histogram'][-1] if not pd.isna(indicators['macd_histogram'][-1]) else 0
        
        if current_macd > current_macd_signal and current_macd_hist > 0:
            signals.append({"type": "BUY", "indicator": "MACD", "strength": "strong", "reason": "MACD above signal line"})
            signal_strength += 2
        elif current_macd < current_macd_signal and current_macd_hist < 0:
            signals.append({"type": "SELL", "indicator": "MACD", "strength": "strong", "reason": "MACD below signal line"})
            signal_strength -= 2
        
        # Moving average signals
        current_sma_20 = indicators['sma_20'][-1] if not pd.isna(indicators['sma_20'][-1]) else 0
        current_sma_50 = indicators['sma_50'][-1] if not pd.isna(indicators['sma_50'][-1]) else 0
        
        if current_price > current_sma_20 > current_sma_50:
            signals.append({"type": "BUY", "indicator": "MA", "strength": "strong", "reason": "Price above 20-day and 50-day SMAs"})
            signal_strength += 2
        elif current_price < current_sma_20 < current_sma_50:
            signals.append({"type": "SELL", "indicator": "MA", "strength": "strong", "reason": "Price below 20-day and 50-day SMAs"})
            signal_strength -= 2
        
        # Bollinger Bands signals
        current_bb_upper = indicators['bb_upper'][-1] if not pd.isna(indicators['bb_upper'][-1]) else 0
        current_bb_lower = indicators['bb_lower'][-1] if not pd.isna(indicators['bb_lower'][-1]) else 0
        
        if current_price < current_bb_lower:
            signals.append({"type": "BUY", "indicator": "BB", "strength": "moderate", "reason": "Price below lower Bollinger Band"})
            signal_strength += 1
        elif current_price > current_bb_upper:
            signals.append({"type": "SELL", "indicator": "BB", "strength": "moderate", "reason": "Price above upper Bollinger Band"})
            signal_strength -= 1
        
        # Stochastic signals
        current_stoch_k = indicators['stoch_k'][-1] if not pd.isna(indicators['stoch_k'][-1]) else 50
        current_stoch_d = indicators['stoch_d'][-1] if not pd.isna(indicators['stoch_d'][-1]) else 50
        
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
        
        return {
            "status": "success",
            "ticker": ticker,
            "current_price": current_price,
            "signals": signals,
            "signal_strength": signal_strength,
            "overall_recommendation": overall_signal,
            "total_signals": len(signals),
            "buy_signals": len([s for s in signals if s['type'] == 'BUY']),
            "sell_signals": len([s for s in signals if s['type'] == 'SELL'])
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Failed to generate trading signals: {str(e)}"
        }

@tool
def validate_technical_analysis(ticker: str, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate technical analysis results for consistency and reliability.
    
    Args:
        ticker: Stock ticker symbol
        analysis_data: Technical analysis data to validate
        
    Returns:
        Dictionary with validation results
    """
    try:
        tools = TechnicalAnalyzerTools()
        data = tools._get_price_data(ticker, "1mo")  # Recent data for validation
        
        if data.empty:
            return {
                "status": "error",
                "error": "No data available for validation"
            }
        
        validation_results = {
            "data_quality": "good",
            "indicator_consistency": "good",
            "signal_reliability": "good",
            "warnings": [],
            "recommendations": []
        }
        
        current_price = data['Close'].iloc[-1]
        
        # Validate data quality
        if len(data) < 50:
            validation_results["data_quality"] = "poor"
            validation_results["warnings"].append("Insufficient data points for reliable analysis")
        
        # Check for data gaps
        expected_days = len(data)
        actual_days = (data.index[-1] - data.index[0]).days
        if actual_days < expected_days * 0.8:
            validation_results["warnings"].append("Data gaps detected in price history")
        
        # Validate indicator consistency
        if "current_indicators" in analysis_data:
            indicators = analysis_data["current_indicators"]
            
            # Check RSI bounds
            if "rsi" in indicators and indicators["rsi"] is not None:
                if indicators["rsi"] < 0 or indicators["rsi"] > 100:
                    validation_results["indicator_consistency"] = "poor"
                    validation_results["warnings"].append("RSI value out of valid range (0-100)")
            
            # Check for conflicting signals
            buy_signals = 0
            sell_signals = 0
            
            if "signals" in analysis_data:
                for signal in analysis_data["signals"]:
                    if signal["type"] == "BUY":
                        buy_signals += 1
                    elif signal["type"] == "SELL":
                        sell_signals += 1
                
                if buy_signals > 0 and sell_signals > 0:
                    validation_results["signal_reliability"] = "poor"
                    validation_results["warnings"].append("Conflicting buy and sell signals detected")
        
        # Generate recommendations
        if validation_results["data_quality"] == "poor":
            validation_results["recommendations"].append("Collect more historical data for better analysis")
        
        if validation_results["indicator_consistency"] == "poor":
            validation_results["recommendations"].append("Verify indicator calculations and data sources")
        
        if validation_results["signal_reliability"] == "poor":
            validation_results["recommendations"].append("Review signal generation logic for conflicts")
        
        # Overall validation score
        score = 100
        if validation_results["data_quality"] == "poor":
            score -= 30
        if validation_results["indicator_consistency"] == "poor":
            score -= 30
        if validation_results["signal_reliability"] == "poor":
            score -= 40
        
        validation_results["validation_score"] = max(0, score)
        
        return {
            "status": "success",
            "ticker": ticker,
            "validation_results": validation_results,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Failed to validate technical analysis: {str(e)}"
        } 