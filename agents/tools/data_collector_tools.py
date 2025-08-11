# agents/data_collector_tools.py
"""
Data Collector Tools for Stock Prediction Workflow

This module provides tools for the data collector agent to:
- Collect historical price data with different timeframes
- Gather company information and fundamentals
- Collect market data and indices
- Calculate technical indicators
- Validate data quality and completeness
- Cache data for performance optimization
"""

import os
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import requests
import json
import time
from langchain_core.tools import tool
from dotenv import load_dotenv
import hashlib
import pickle
from pathlib import Path

# Load environment variables
load_dotenv()

class DataCollectorTools:
    """
    Tools for the data collector agent to manage comprehensive data collection.
    """
    
    def __init__(self):
        """Initialize data collector tools."""
        self.cache_dir = Path("cache/data_collector")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
    
    def _get_cache_key(self, ticker: str, data_type: str, period: str = "1d") -> str:
        """Generate cache key for data."""
        return hashlib.md5(f"{ticker}_{data_type}_{period}_{datetime.now().strftime('%Y%m%d')}".encode()).hexdigest()
    
    def _get_cached_data(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached data if available and fresh."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Check if cache is less than 1 hour old
                if datetime.now().timestamp() - cached_data.get('timestamp', 0) < 3600:
                    return cached_data.get('data')
            except Exception:
                pass
        return None
    
    def _cache_data(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Cache data with timestamp."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            cached_data = {
                'data': data,
                'timestamp': datetime.now().timestamp()
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f)
        except Exception:
            pass

@tool
def collect_price_data(ticker: str, period: str = "3mo", interval: str = "1d") -> Dict[str, Any]:
    """
    Collect comprehensive historical price data for a stock.
    
    Args:
        ticker: Stock ticker symbol
        period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
    Returns:
        Dictionary with comprehensive price data
    """
    try:
        # Check cache first
        cache_key = f"price_data_{ticker}_{period}_{interval}"
        cached_data = DataCollectorTools()._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        # Fetch fresh data
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval)
        
        if hist.empty:
            return {
                "status": "error",
                "error": f"No price data available for {ticker}",
                "ticker": ticker,
                "period": period,
                "interval": interval
            }
        
        # Calculate price metrics
        latest = hist.iloc[-1]
        prev_day = hist.iloc[-2] if len(hist) > 1 else latest
        
        # Calculate additional metrics
        hist['Returns'] = hist['Close'].pct_change()
        hist['Volatility'] = hist['Returns'].rolling(window=20).std()
        
        price_data = {
            "status": "success",
            "ticker": ticker,
            "period": period,
            "interval": interval,
            "data_points": len(hist),
            "current_price": float(latest['Close']),
            "previous_close": float(prev_day['Close']),
            "daily_change": float(latest['Close'] - prev_day['Close']),
            "daily_change_pct": float((latest['Close'] - prev_day['Close']) / prev_day['Close'] * 100),
            "volume": int(latest['Volume']),
            "high": float(latest['High']),
            "low": float(latest['Low']),
            "open": float(latest['Open']),
            "avg_volume": int(hist['Volume'].mean()),
            "price_range": {
                "min": float(hist['Low'].min()),
                "max": float(hist['High'].max()),
                "current": float(latest['Close'])
            },
            "volatility": float(hist['Volatility'].iloc[-1]) if not pd.isna(hist['Volatility'].iloc[-1]) else 0,
            "returns": {
                "daily": float(hist['Returns'].iloc[-1]) if not pd.isna(hist['Returns'].iloc[-1]) else 0,
                "weekly": float(hist['Returns'].tail(5).sum()) if len(hist) >= 5 else 0,
                "monthly": float(hist['Returns'].tail(20).sum()) if len(hist) >= 20 else 0
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache the data
        DataCollectorTools()._cache_data(cache_key, price_data)
        
        return price_data
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Failed to collect price data: {str(e)}",
            "ticker": ticker,
            "period": period,
            "interval": interval
        }

@tool
def collect_company_info(ticker: str) -> Dict[str, Any]:
    """
    Collect comprehensive company information and fundamentals.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with company information
    """
    try:
        # Check cache first
        cache_key = f"company_info_{ticker}"
        cached_data = DataCollectorTools()._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get additional data
        try:
            financials = stock.financials
            balance_sheet = stock.balance_sheet
            cashflow = stock.cashflow
        except Exception:
            financials = None
            balance_sheet = None
            cashflow = None
        
        company_info = {
            "status": "success",
            "ticker": ticker,
            "basic_info": {
                "name": info.get('longName', 'Unknown'),
                "short_name": info.get('shortName', 'Unknown'),
                "sector": info.get('sector', 'Unknown'),
                "industry": info.get('industry', 'Unknown'),
                "country": info.get('country', 'Unknown'),
                "website": info.get('website', 'Unknown'),
                "description": info.get('longBusinessSummary', 'No description available')
            },
            "market_data": {
                "market_cap": info.get('marketCap', 0),
                "enterprise_value": info.get('enterpriseValue', 0),
                "float_shares": info.get('floatShares', 0),
                "shares_outstanding": info.get('sharesOutstanding', 0),
                "shares_short": info.get('sharesShort', 0),
                "shares_short_prev_month": info.get('sharesShortPrevMonth', 0)
            },
            "valuation": {
                "pe_ratio": info.get('trailingPE', 0),
                "forward_pe": info.get('forwardPE', 0),
                "peg_ratio": info.get('pegRatio', 0),
                "price_to_book": info.get('priceToBook', 0),
                "price_to_sales": info.get('priceToSalesTrailing12Months', 0),
                "enterprise_to_revenue": info.get('enterpriseToRevenue', 0),
                "enterprise_to_ebitda": info.get('enterpriseToEbitda', 0)
            },
            "financial_metrics": {
                "beta": info.get('beta', 0),
                "dividend_yield": info.get('dividendYield', 0),
                "dividend_rate": info.get('dividendRate', 0),
                "payout_ratio": info.get('payoutRatio', 0),
                "profit_margins": info.get('profitMargins', 0),
                "operating_margins": info.get('operatingMargins', 0),
                "ebitda_margins": info.get('ebitdaMargins', 0),
                "revenue_growth": info.get('revenueGrowth', 0),
                "earnings_growth": info.get('earningsGrowth', 0),
                "revenue_per_share": info.get('revenuePerShare', 0),
                "return_on_equity": info.get('returnOnEquity', 0),
                "return_on_assets": info.get('returnOnAssets', 0),
                "debt_to_equity": info.get('debtToEquity', 0),
                "current_ratio": info.get('currentRatio', 0),
                "quick_ratio": info.get('quickRatio', 0)
            },
            "trading_info": {
                "fifty_day_average": info.get('fiftyDayAverage', 0),
                "two_hundred_day_average": info.get('twoHundredDayAverage', 0),
                "fifty_two_week_high": info.get('fiftyTwoWeekHigh', 0),
                "fifty_two_week_low": info.get('fiftyTwoWeekLow', 0),
                "fifty_two_week_change": info.get('fiftyTwoWeekChange', 0),
                "fifty_two_week_change_pct": info.get('fiftyTwoWeekChangePercent', 0)
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache the data
        DataCollectorTools()._cache_data(cache_key, company_info)
        
        return company_info
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Failed to collect company info: {str(e)}",
            "ticker": ticker
        }

@tool
def collect_market_data(ticker: str) -> Dict[str, Any]:
    """
    Collect market data including indices and sector performance.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with market data
    """
    try:
        # Check cache first
        cache_key = f"market_data_{ticker}"
        cached_data = DataCollectorTools()._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        market_data = {
            "status": "success",
            "ticker": ticker,
            "indices": {},
            "sector_performance": {},
            "market_sentiment": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Collect major indices data
        indices = {
            "sp500": "^GSPC",
            "nasdaq": "^IXIC", 
            "dow": "^DJI",
            "vix": "^VIX"
        }
        
        for index_name, index_symbol in indices.items():
            try:
                index_data = yf.Ticker(index_symbol)
                hist = index_data.history(period="5d")
                
                if not hist.empty:
                    latest = hist.iloc[-1]
                    prev = hist.iloc[-2] if len(hist) > 1 else latest
                    
                    market_data["indices"][index_name] = {
                        "current": float(latest['Close']),
                        "change": float(latest['Close'] - prev['Close']),
                        "change_pct": float((latest['Close'] - prev['Close']) / prev['Close'] * 100),
                        "volume": int(latest['Volume'])
                    }
            except Exception:
                market_data["indices"][index_name] = {
                    "current": 0,
                    "change": 0,
                    "change_pct": 0,
                    "volume": 0
                }
        
        # Determine market trend
        sp500_data = market_data["indices"].get("sp500", {})
        if sp500_data.get("change_pct", 0) > 0.5:
            market_trend = "bullish"
        elif sp500_data.get("change_pct", 0) < -0.5:
            market_trend = "bearish"
        else:
            market_trend = "neutral"
        
        market_data["market_sentiment"] = {
            "overall_trend": market_trend,
            "volatility_level": "high" if market_data["indices"].get("vix", {}).get("current", 0) > 20 else "normal",
            "market_phase": "bull_market" if sp500_data.get("change_pct", 0) > 0 else "bear_market"
        }
        
        # Cache the data
        DataCollectorTools()._cache_data(cache_key, market_data)
        
        return market_data
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Failed to collect market data: {str(e)}",
            "ticker": ticker
        }

@tool
def calculate_technical_indicators(ticker: str, period: str = "3mo") -> Dict[str, Any]:
    """
    Calculate comprehensive technical indicators for a stock.
    
    Args:
        ticker: Stock ticker symbol
        period: Data period for calculations
        
    Returns:
        Dictionary with technical indicators
    """
    try:
        # Check cache first
        cache_key = f"technical_indicators_{ticker}_{period}"
        cached_data = DataCollectorTools()._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        if hist.empty:
            return {
                "status": "error",
                "error": f"No data available for technical analysis of {ticker}",
                "ticker": ticker,
                "period": period
            }
        
        # Calculate moving averages
        hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
        hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
        hist['SMA_200'] = hist['Close'].rolling(window=200).mean()
        hist['EMA_12'] = hist['Close'].ewm(span=12).mean()
        hist['EMA_26'] = hist['Close'].ewm(span=26).mean()
        
        # Calculate MACD
        hist['MACD'] = hist['EMA_12'] - hist['EMA_26']
        hist['MACD_Signal'] = hist['MACD'].ewm(span=9).mean()
        hist['MACD_Histogram'] = hist['MACD'] - hist['MACD_Signal']
        
        # Calculate RSI
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        hist['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate Bollinger Bands
        hist['BB_Middle'] = hist['Close'].rolling(window=20).mean()
        bb_std = hist['Close'].rolling(window=20).std()
        hist['BB_Upper'] = hist['BB_Middle'] + (bb_std * 2)
        hist['BB_Lower'] = hist['BB_Middle'] - (bb_std * 2)
        hist['BB_Width'] = (hist['BB_Upper'] - hist['BB_Lower']) / hist['BB_Middle']
        
        # Calculate Stochastic Oscillator
        hist['Stoch_K'] = ((hist['Close'] - hist['Low'].rolling(window=14).min()) / 
                           (hist['High'].rolling(window=14).max() - hist['Low'].rolling(window=14).min())) * 100
        hist['Stoch_D'] = hist['Stoch_K'].rolling(window=3).mean()
        
        # Calculate Average True Range (ATR)
        hist['TR'] = np.maximum(
            hist['High'] - hist['Low'],
            np.maximum(
                abs(hist['High'] - hist['Close'].shift(1)),
                abs(hist['Low'] - hist['Close'].shift(1))
            )
        )
        hist['ATR'] = hist['TR'].rolling(window=14).mean()
        
        # Get latest values
        latest = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) > 1 else latest
        
        technical_indicators = {
            "status": "success",
            "ticker": ticker,
            "period": period,
            "moving_averages": {
                "sma_20": float(latest['SMA_20']) if not pd.isna(latest['SMA_20']) else 0,
                "sma_50": float(latest['SMA_50']) if not pd.isna(latest['SMA_50']) else 0,
                "sma_200": float(latest['SMA_200']) if not pd.isna(latest['SMA_200']) else 0,
                "ema_12": float(latest['EMA_12']) if not pd.isna(latest['EMA_12']) else 0,
                "ema_26": float(latest['EMA_26']) if not pd.isna(latest['EMA_26']) else 0
            },
            "macd": {
                "macd_line": float(latest['MACD']) if not pd.isna(latest['MACD']) else 0,
                "signal_line": float(latest['MACD_Signal']) if not pd.isna(latest['MACD_Signal']) else 0,
                "histogram": float(latest['MACD_Histogram']) if not pd.isna(latest['MACD_Histogram']) else 0,
                "signal": "bullish" if latest['MACD'] > latest['MACD_Signal'] else "bearish"
            },
            "rsi": {
                "value": float(latest['RSI']) if not pd.isna(latest['RSI']) else 50,
                "signal": "oversold" if latest['RSI'] < 30 else "overbought" if latest['RSI'] > 70 else "neutral"
            },
            "bollinger_bands": {
                "upper": float(latest['BB_Upper']) if not pd.isna(latest['BB_Upper']) else 0,
                "middle": float(latest['BB_Middle']) if not pd.isna(latest['BB_Middle']) else 0,
                "lower": float(latest['BB_Lower']) if not pd.isna(latest['BB_Lower']) else 0,
                "width": float(latest['BB_Width']) if not pd.isna(latest['BB_Width']) else 0,
                "position": "upper" if latest['Close'] > latest['BB_Upper'] else "lower" if latest['Close'] < latest['BB_Lower'] else "middle"
            },
            "stochastic": {
                "k_percent": float(latest['Stoch_K']) if not pd.isna(latest['Stoch_K']) else 50,
                "d_percent": float(latest['Stoch_D']) if not pd.isna(latest['Stoch_D']) else 50,
                "signal": "oversold" if latest['Stoch_K'] < 20 else "overbought" if latest['Stoch_K'] > 80 else "neutral"
            },
            "atr": {
                "value": float(latest['ATR']) if not pd.isna(latest['ATR']) else 0,
                "volatility": "high" if latest['ATR'] > hist['ATR'].mean() else "low"
            },
            "price_analysis": {
                "price_vs_sma20": "above" if latest['Close'] > latest['SMA_20'] else "below",
                "price_vs_sma50": "above" if latest['Close'] > latest['SMA_50'] else "below",
                "price_vs_sma200": "above" if latest['Close'] > latest['SMA_200'] else "below",
                "golden_cross": latest['SMA_50'] > latest['SMA_200'] and prev['SMA_50'] <= prev['SMA_200'],
                "death_cross": latest['SMA_50'] < latest['SMA_200'] and prev['SMA_50'] >= prev['SMA_200']
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache the data
        DataCollectorTools()._cache_data(cache_key, technical_indicators)
        
        return technical_indicators
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Failed to calculate technical indicators: {str(e)}",
            "ticker": ticker,
            "period": period
        }

@tool
def validate_data_quality(ticker: str, data_type: str = "all") -> Dict[str, Any]:
    """
    Validate the quality and completeness of collected data.
    
    Args:
        ticker: Stock ticker symbol
        data_type: Type of data to validate (price, company, market, technical, all)
        
    Returns:
        Dictionary with validation results
    """
    try:
        validation_results = {
            "status": "success",
            "ticker": ticker,
            "data_type": data_type,
            "quality_score": 0,
            "issues": [],
            "recommendations": [],
            "timestamp": datetime.now().isoformat()
        }
        
        issues = []
        recommendations = []
        quality_score = 100
        
        # Validate price data
        if data_type in ["price", "all"]:
            price_data = collect_price_data.invoke({"ticker": ticker, "period": "3mo"})
            if price_data.get("status") == "success":
                if price_data.get("data_points", 0) < 30:
                    issues.append("Insufficient price data points")
                    quality_score -= 20
                    recommendations.append("Collect more historical data")
                
                if price_data.get("current_price", 0) <= 0:
                    issues.append("Invalid current price")
                    quality_score -= 30
                    recommendations.append("Verify ticker symbol and data source")
            else:
                issues.append(f"Price data collection failed: {price_data.get('error')}")
                quality_score -= 50
        
        # Validate company info
        if data_type in ["company", "all"]:
            company_info = collect_company_info.invoke({"ticker": ticker})
            if company_info.get("status") == "success":
                basic_info = company_info.get("basic_info", {})
                if basic_info.get("name") == "Unknown":
                    issues.append("Missing company name")
                    quality_score -= 10
                
                if company_info.get("market_data", {}).get("market_cap", 0) <= 0:
                    issues.append("Missing market cap data")
                    quality_score -= 15
                    recommendations.append("Verify company is publicly traded")
            else:
                issues.append(f"Company info collection failed: {company_info.get('error')}")
                quality_score -= 25
        
        # Validate market data
        if data_type in ["market", "all"]:
            market_data = collect_market_data.invoke({"ticker": ticker})
            if market_data.get("status") == "success":
                indices = market_data.get("indices", {})
                if not indices.get("sp500"):
                    issues.append("Missing S&P 500 data")
                    quality_score -= 10
            else:
                issues.append(f"Market data collection failed: {market_data.get('error')}")
                quality_score -= 15
        
        # Validate technical indicators
        if data_type in ["technical", "all"]:
            technical_data = calculate_technical_indicators.invoke({"ticker": ticker, "period": "3mo"})
            if technical_data.get("status") == "success":
                indicators = technical_data.get("moving_averages", {})
                if indicators.get("sma_20", 0) <= 0:
                    issues.append("Missing SMA calculations")
                    quality_score -= 10
            else:
                issues.append(f"Technical indicators calculation failed: {technical_data.get('error')}")
                quality_score -= 20
        
        validation_results.update({
            "quality_score": max(0, quality_score),
            "issues": issues,
            "recommendations": recommendations,
            "data_quality": "excellent" if quality_score >= 90 else "good" if quality_score >= 70 else "fair" if quality_score >= 50 else "poor"
        })
        
        return validation_results
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Data validation failed: {str(e)}",
            "ticker": ticker,
            "data_type": data_type
        }

@tool
def collect_comprehensive_data(ticker: str, period: str = "3mo") -> Dict[str, Any]:
    """
    Collect all comprehensive data for a stock in one operation.
    
    Args:
        ticker: Stock ticker symbol
        period: Data period for analysis
        
    Returns:
        Dictionary with all collected data
    """
    try:
        print(f"📊 Collecting comprehensive data for {ticker}...")
        
        # Collect all data types
        price_data = collect_price_data.invoke({"ticker": ticker, "period": period})
        company_info = collect_company_info.invoke({"ticker": ticker})
        market_data = collect_market_data.invoke({"ticker": ticker})
        technical_indicators = calculate_technical_indicators.invoke({"ticker": ticker, "period": period})
        data_validation = validate_data_quality.invoke({"ticker": ticker, "data_type": "all"})
        
        # Compile comprehensive data
        comprehensive_data = {
            "status": "success",
            "ticker": ticker,
            "period": period,
            "collection_timestamp": datetime.now().isoformat(),
            "data_sources": ["yfinance", "technical_indicators", "market_data"],
            "price_data": price_data,
            "company_info": company_info,
            "market_data": market_data,
            "technical_indicators": technical_indicators,
            "data_validation": data_validation,
            "summary": {
                "current_price": price_data.get("current_price", 0) if price_data.get("status") == "success" else 0,
                "company_name": company_info.get("basic_info", {}).get("name", "Unknown") if company_info.get("status") == "success" else "Unknown",
                "sector": company_info.get("basic_info", {}).get("sector", "Unknown") if company_info.get("status") == "success" else "Unknown",
                "market_cap": company_info.get("market_data", {}).get("market_cap", 0) if company_info.get("status") == "success" else 0,
                "quality_score": data_validation.get("quality_score", 0) if data_validation.get("status") == "success" else 0,
                "data_points": price_data.get("data_points", 0) if price_data.get("status") == "success" else 0
            }
        }
        
        print(f"✅ Comprehensive data collection completed for {ticker}")
        print(f"   Quality Score: {comprehensive_data['summary']['quality_score']}")
        print(f"   Data Points: {comprehensive_data['summary']['data_points']}")
        print(f"   Current Price: ${comprehensive_data['summary']['current_price']:.2f}")
        
        return comprehensive_data
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Comprehensive data collection failed: {str(e)}",
            "ticker": ticker,
            "period": period
        }

# Create data collector tools instance
data_collector_tools = DataCollectorTools() 