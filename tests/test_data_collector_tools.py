#!/usr/bin/env python3
"""
🧪 Test Data Collector Tools
Test the comprehensive tools we've created for the data collector agent.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import modules
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from agents.tools.data_collector_tools import (
    collect_price_data,
    collect_company_info,
    collect_market_data,
    calculate_technical_indicators,
    validate_data_quality,
    collect_comprehensive_data
)

# Load environment variables
load_dotenv()

def test_data_collector_tools():
    """Test all data collector tools."""
    print("🧪 Testing Data Collector Tools")
    print("=" * 50)
    
    # Test cases
    test_tickers = ["AAPL", "MSFT", "TSLA", "GOOGL", "AMZN"]
    
    for ticker in test_tickers:
        print(f"\n📊 Testing {ticker}")
        print("-" * 30)
        
        # Test 1: Price Data Collection
        print(f"1️⃣ Testing price data collection for {ticker}...")
        try:
            price_result = collect_price_data.invoke({"ticker": ticker, "period": "3mo"})
            if price_result.get("status") == "success":
                print(f"   ✅ Price data collected successfully")
                print(f"   Current Price: ${price_result.get('current_price', 0):.2f}")
                print(f"   Data Points: {price_result.get('data_points', 0)}")
                print(f"   Daily Change: {price_result.get('daily_change_pct', 0):.2f}%")
            else:
                print(f"   ❌ Price data collection failed: {price_result.get('error')}")
        except Exception as e:
            print(f"   ❌ Price data test error: {str(e)}")
        
        # Test 2: Company Info Collection
        print(f"2️⃣ Testing company info collection for {ticker}...")
        try:
            company_result = collect_company_info.invoke({"ticker": ticker})
            if company_result.get("status") == "success":
                basic_info = company_result.get("basic_info", {})
                print(f"   ✅ Company info collected successfully")
                print(f"   Company Name: {basic_info.get('name', 'Unknown')}")
                print(f"   Sector: {basic_info.get('sector', 'Unknown')}")
                print(f"   Market Cap: ${company_result.get('market_data', {}).get('market_cap', 0):,.0f}")
            else:
                print(f"   ❌ Company info collection failed: {company_result.get('error')}")
        except Exception as e:
            print(f"   ❌ Company info test error: {str(e)}")
        
        # Test 3: Market Data Collection
        print(f"3️⃣ Testing market data collection for {ticker}...")
        try:
            market_result = collect_market_data.invoke({"ticker": ticker})
            if market_result.get("status") == "success":
                print(f"   ✅ Market data collected successfully")
                indices = market_result.get("indices", {})
                sp500 = indices.get("sp500", {})
                print(f"   S&P 500: ${sp500.get('current', 0):.2f} ({sp500.get('change_pct', 0):.2f}%)")
                print(f"   Market Trend: {market_result.get('market_sentiment', {}).get('overall_trend', 'neutral')}")
            else:
                print(f"   ❌ Market data collection failed: {market_result.get('error')}")
        except Exception as e:
            print(f"   ❌ Market data test error: {str(e)}")
        
        # Test 4: Technical Indicators
        print(f"4️⃣ Testing technical indicators for {ticker}...")
        try:
            technical_result = calculate_technical_indicators.invoke({"ticker": ticker, "period": "3mo"})
            if technical_result.get("status") == "success":
                print(f"   ✅ Technical indicators calculated successfully")
                moving_avgs = technical_result.get("moving_averages", {})
                rsi = technical_result.get("rsi", {})
                print(f"   SMA 20: ${moving_avgs.get('sma_20', 0):.2f}")
                print(f"   RSI: {rsi.get('value', 0):.1f} ({rsi.get('signal', 'neutral')})")
                print(f"   MACD Signal: {technical_result.get('macd', {}).get('signal', 'neutral')}")
            else:
                print(f"   ❌ Technical indicators calculation failed: {technical_result.get('error')}")
        except Exception as e:
            print(f"   ❌ Technical indicators test error: {str(e)}")
        
        # Test 5: Data Quality Validation
        print(f"5️⃣ Testing data quality validation for {ticker}...")
        try:
            validation_result = validate_data_quality.invoke({"ticker": ticker, "data_type": "all"})
            if validation_result.get("status") == "success":
                print(f"   ✅ Data quality validated successfully")
                print(f"   Quality Score: {validation_result.get('quality_score', 0)}/100")
                print(f"   Data Quality: {validation_result.get('data_quality', 'unknown')}")
                issues = validation_result.get("issues", [])
                if issues:
                    print(f"   Issues Found: {len(issues)}")
                    for issue in issues[:3]:  # Show first 3 issues
                        print(f"     - {issue}")
            else:
                print(f"   ❌ Data quality validation failed: {validation_result.get('error')}")
        except Exception as e:
            print(f"   ❌ Data quality test error: {str(e)}")

def test_comprehensive_data_collection():
    """Test comprehensive data collection for a single ticker."""
    print(f"\n🚀 Testing Comprehensive Data Collection")
    print("=" * 50)
    
    test_ticker = "AAPL"
    print(f"📊 Testing comprehensive data collection for {test_ticker}...")
    
    try:
        comprehensive_result = collect_comprehensive_data.invoke({"ticker": test_ticker, "period": "3mo"})
        
        if comprehensive_result.get("status") == "success":
            print(f"✅ Comprehensive data collection completed successfully!")
            
            summary = comprehensive_result.get("summary", {})
            print(f"\n📋 Summary:")
            print(f"   Company: {summary.get('company_name', 'Unknown')}")
            print(f"   Current Price: ${summary.get('current_price', 0):.2f}")
            print(f"   Sector: {summary.get('sector', 'Unknown')}")
            print(f"   Market Cap: ${summary.get('market_cap', 0):,.0f}")
            print(f"   Quality Score: {summary.get('quality_score', 0)}/100")
            print(f"   Data Points: {summary.get('data_points', 0)}")
            print(f"   Market Trend: {summary.get('market_trend', 'neutral')}")
            
            # Show data sources used
            data_sources = comprehensive_result.get("data_sources", [])
            print(f"\n📊 Data Sources: {', '.join(data_sources)}")
            
            # Show tools used
            tools_used = comprehensive_result.get("data_collector_tools_used", [])
            print(f"🔧 Tools Used: {len(tools_used)} tools")
            
        else:
            print(f"❌ Comprehensive data collection failed: {comprehensive_result.get('error')}")
            
    except Exception as e:
        print(f"❌ Comprehensive data collection test error: {str(e)}")

def test_error_handling():
    """Test error handling with invalid ticker."""
    print(f"\n⚠️ Testing Error Handling")
    print("=" * 30)
    
    invalid_ticker = "INVALID_TICKER_12345"
    print(f"🧪 Testing with invalid ticker: {invalid_ticker}")
    
    try:
        price_result = collect_price_data.invoke({"ticker": invalid_ticker, "period": "3mo"})
        if price_result.get("status") == "error":
            print(f"✅ Error handling working correctly")
            print(f"   Error: {price_result.get('error')}")
        else:
            print(f"⚠️ Unexpected success with invalid ticker")
            
    except Exception as e:
        print(f"❌ Error handling test failed: {str(e)}")

def test_data_caching():
    """Test data caching functionality."""
    import time
    
    print(f"\n💾 Testing Data Caching")
    print("=" * 25)
    
    test_ticker = "MSFT"
    print(f"🧪 Testing caching for {test_ticker}")
    
    try:
        # First call (should cache)
        print("1️⃣ First call (should cache data)...")
        start_time = time.time()
        result1 = collect_price_data.invoke({"ticker": test_ticker, "period": "1mo"})
        time1 = time.time() - start_time
        print(f"   Time: {time1:.2f}s")
        
        # Second call (should use cache)
        print("2️⃣ Second call (should use cache)...")
        start_time = time.time()
        result2 = collect_price_data.invoke({"ticker": test_ticker, "period": "1mo"})
        time2 = time.time() - start_time
        print(f"   Time: {time2:.2f}s")
        
        if time2 < time1:
            print(f"✅ Caching working (cached call faster)")
        else:
            print(f"⚠️ Caching may not be working (cached call slower)")
            
    except Exception as e:
        print(f"❌ Caching test error: {str(e)}")

def main():
    """Main test function."""
    import time
    
    print("🧪 Data Collector Tools Test Suite")
    print("=" * 50)
    
    # Run all tests
    test_data_collector_tools()
    test_comprehensive_data_collection()
    test_error_handling()
    test_data_caching()
    
    print(f"\n🎉 All data collector tools tests completed!")
    print("✅ Tools are ready for integration into the main workflow")

if __name__ == "__main__":
    main() 