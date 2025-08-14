#!/usr/bin/env python3
"""
Stock4U MCP Server Demo

This script demonstrates how to use the Stock4U MCP server
for stock analysis and predictions.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from mcp.client.stdio import stdio_client
except ImportError:
    print("❌ MCP client not installed. Install with: pip install 'mcp[cli]==1.12.4'")
    sys.exit(1)


async def demo_market_snapshot(client, ticker: str):
    """Demonstrate market snapshot functionality."""
    print(f"\n📊 Getting market snapshot for {ticker}...")
    
    try:
        result = await client.call_tool("get_market_snapshot", {"ticker": ticker})
        
        if result.get("status") == "success":
            data = result.get("data", {})
            print(f"✅ {ticker} Market Snapshot:")
            print(f"   Last Close: ${data.get('last_close', 'N/A'):.2f}")
            print(f"   Change: {data.get('change_pct', 'N/A'):.2f}%")
            print(f"   Volume: {data.get('volume', 'N/A'):,}")
            
            indicators = data.get("indicators", {})
            print(f"   RSI: {indicators.get('rsi', 'N/A'):.1f}")
            print(f"   SMA20: ${indicators.get('sma_20', 'N/A'):.2f}")
            print(f"   SMA50: ${indicators.get('sma_50', 'N/A'):.2f}")
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")


async def demo_stock_data(client, ticker: str, period: str = "1mo"):
    """Demonstrate stock data retrieval."""
    print(f"\n📈 Getting {period} stock data for {ticker}...")
    
    try:
        result = await client.call_tool("get_stock_data", {
            "ticker": ticker,
            "period": period
        })
        
        if result.get("status") == "success":
            data = result.get("data", {})
            rows = data.get("rows", [])
            
            if rows:
                print(f"✅ Retrieved {len(rows)} data points for {ticker}")
                
                # Show latest data
                latest = rows[-1]
                print(f"   Latest Close: ${latest.get('Close', 'N/A'):.2f}")
                print(f"   Date: {latest.get('Date', 'N/A')}")
                print(f"   Volume: {latest.get('Volume', 'N/A'):,}")
            else:
                print("⚠️ No data returned")
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")


async def demo_prediction(client, ticker: str, mode: str = "low_api"):
    """Demonstrate stock prediction functionality."""
    print(f"\n🔮 Running {mode} prediction for {ticker}...")
    
    try:
        # Configure prediction mode
        low_api = mode == "low_api"
        fast_ta = mode == "fast"
        
        result = await client.call_tool("run_stock_prediction", {
            "ticker": ticker,
            "timeframe": "1d",
            "low_api_mode": low_api,
            "fast_ta_mode": fast_ta,
            "use_ml_model": False
        })
        
        if result.get("status") == "success":
            prediction_data = result.get("result", {})
            prediction = prediction_data.get("prediction_result", {}).get("prediction", {})
            
            print(f"✅ {ticker} Prediction Results:")
            print(f"   Direction: {prediction.get('direction', 'N/A')}")
            print(f"   Confidence: {prediction.get('confidence', 'N/A'):.1f}%")
            print(f"   Timeframe: {prediction.get('timeframe', 'N/A')}")
            
            # Show risk assessment if available
            risk = prediction_data.get("risk_assessment", {})
            if risk:
                print(f"   Risk Level: {risk.get('risk_level', 'N/A')}")
                print(f"   Risk Score: {risk.get('risk_score', 'N/A'):.1f}")
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")


async def demo_caching(client):
    """Demonstrate caching functionality."""
    print(f"\n💾 Testing cache functionality...")
    
    try:
        # Test cache invalidation
        result = await client.call_tool("invalidate_cache", {"cache_key": "test_key"})
        print(f"✅ Cache invalidation: {result}")
        
        # Test getting cached result
        result = await client.call_tool("get_cached_result", {
            "cache_key": "test_key",
            "ttl_seconds": 900
        })
        print(f"✅ Cache retrieval: {result.get('status', 'unknown')}")
        
    except Exception as e:
        print(f"❌ Exception: {e}")


async def demo_health_check(client):
    """Demonstrate health check functionality."""
    print(f"\n🏥 Testing health check...")
    
    try:
        result = await client.call_tool("ping", {})
        print(f"✅ Health check: {result}")
        
    except Exception as e:
        print(f"❌ Exception: {e}")


async def demo_batch_analysis(client, tickers: list):
    """Demonstrate batch analysis of multiple stocks."""
    print(f"\n🔄 Running batch analysis for {len(tickers)} stocks...")
    
    results = {}
    
    for ticker in tickers:
        print(f"   Analyzing {ticker}...")
        try:
            # Get snapshot
            snapshot = await client.call_tool("get_market_snapshot", {"ticker": ticker})
            
            # Get prediction (low API mode for speed)
            prediction = await client.call_tool("run_stock_prediction", {
                "ticker": ticker,
                "timeframe": "1d",
                "low_api_mode": True,
                "fast_ta_mode": True
            })
            
            results[ticker] = {
                "snapshot": snapshot,
                "prediction": prediction
            }
            
        except Exception as e:
            print(f"   ❌ Error analyzing {ticker}: {e}")
    
    # Display summary
    print(f"\n📋 Batch Analysis Summary:")
    for ticker, data in results.items():
        snapshot = data.get("snapshot", {})
        prediction = data.get("prediction", {})
        
        if snapshot.get("status") == "success":
            snap_data = snapshot.get("data", {})
            print(f"   {ticker}: ${snap_data.get('last_close', 'N/A'):.2f} "
                  f"({snap_data.get('change_pct', 'N/A'):.2f}%)")
        
        if prediction.get("status") == "success":
            pred_data = prediction.get("result", {})
            pred = pred_data.get("prediction_result", {}).get("prediction", {})
            print(f"     → {pred.get('direction', 'N/A')} "
                  f"({pred.get('confidence', 'N/A'):.1f}% confidence)")


async def main():
    """Main demo function."""
    print("🚀 Stock4U MCP Server Demo")
    print("=" * 50)
    
    # Test tickers
    test_tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    
    try:
        from mcp.client.stdio import StdioServerParameters
        server_params = StdioServerParameters(command="python", args=["-m", "agents.mcp_server"])
        async with stdio_client(server_params) as client:
            print("✅ Connected to Stock4U MCP server")
            
            # Health check
            await demo_health_check(client)
            
            # Market snapshots
            for ticker in test_tickers[:3]:
                await demo_market_snapshot(client, ticker)
            
            # Stock data
            await demo_stock_data(client, "AAPL", "1mo")
            
            # Predictions (different modes)
            await demo_prediction(client, "AAPL", "low_api")
            await demo_prediction(client, "MSFT", "fast")
            
            # Caching
            await demo_caching(client)
            
            # Batch analysis
            await demo_batch_analysis(client, test_tickers[:3])
            
            print(f"\n🎉 Demo completed successfully!")
            print(f"💡 Try running individual functions or explore the MCP integration guide.")
            
    except Exception as e:
        print(f"❌ Failed to connect to MCP server: {e}")
        print(f"💡 Make sure the MCP server is running: python -m agents.mcp_server")


if __name__ == "__main__":
    asyncio.run(main())
