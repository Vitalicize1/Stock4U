#!/usr/bin/env python3
"""
Stock4U MCP Client Examples

This file contains practical examples of how to integrate with the Stock4U MCP server
using different programming languages and frameworks.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from mcp.client.stdio import stdio_client
except ImportError:
    print("❌ MCP client not installed. Install with: pip install 'mcp[cli]==1.12.4'")
    sys.exit(1)


class Stock4UClient:
    """High-level client for Stock4U MCP server."""
    
    def __init__(self):
        self.client = None
    
    async def __aenter__(self):
        from mcp.client.stdio import StdioServerParameters
        server_params = StdioServerParameters(command="python", args=["-m", "agents.mcp_server"])
        self.client = await stdio_client(server_params).__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.__aexit__(exc_type, exc_val, exc_tb)
    
    async def get_market_snapshot(self, ticker: str) -> Dict[str, Any]:
        """Get market snapshot for a ticker."""
        return await self.client.call_tool("get_market_snapshot", {"ticker": ticker})
    
    async def get_stock_data(self, ticker: str, period: str = "1mo") -> Dict[str, Any]:
        """Get historical stock data."""
        return await self.client.call_tool("get_stock_data", {
            "ticker": ticker,
            "period": period
        })
    
    async def run_prediction(self, ticker: str, timeframe: str = "1d", 
                           low_api_mode: bool = True, fast_ta_mode: bool = True) -> Dict[str, Any]:
        """Run stock prediction."""
        return await self.client.call_tool("run_stock_prediction", {
            "ticker": ticker,
            "timeframe": timeframe,
            "low_api_mode": low_api_mode,
            "fast_ta_mode": fast_ta_mode,
            "use_ml_model": False
        })
    
    async def ping(self) -> Dict[str, Any]:
        """Health check."""
        return await self.client.call_tool("ping", {})


# Example 1: Simple Stock Analysis
async def example_simple_analysis():
    """Example: Simple stock analysis workflow."""
    print("📊 Example 1: Simple Stock Analysis")
    print("-" * 40)
    
    async with Stock4UClient() as client:
        ticker = "AAPL"
        
        # Get market snapshot
        snapshot = await client.get_market_snapshot(ticker)
        if snapshot.get("status") == "success":
            data = snapshot.get("data", {})
            print(f"✅ {ticker} Market Snapshot:")
            print(f"   Price: ${data.get('last_close', 'N/A'):.2f}")
            print(f"   Change: {data.get('change_pct', 'N/A'):.2f}%")
        
        # Get prediction
        prediction = await client.run_prediction(ticker, low_api_mode=True)
        if prediction.get("status") == "success":
            pred_data = prediction.get("result", {})
            pred = pred_data.get("prediction_result", {}).get("prediction", {})
            print(f"   Prediction: {pred.get('direction', 'N/A')} "
                  f"({pred.get('confidence', 'N/A'):.1f}% confidence)")


# Example 2: Portfolio Analysis
async def example_portfolio_analysis():
    """Example: Analyze multiple stocks for portfolio insights."""
    print("\n📈 Example 2: Portfolio Analysis")
    print("-" * 40)
    
    portfolio = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    
    async with Stock4UClient() as client:
        results = {}
        
        for ticker in portfolio:
            print(f"   Analyzing {ticker}...")
            
            # Get snapshot and prediction
            snapshot = await client.get_market_snapshot(ticker)
            prediction = await client.run_prediction(ticker, low_api_mode=True)
            
            results[ticker] = {
                "snapshot": snapshot,
                "prediction": prediction
            }
        
        # Display portfolio summary
        print(f"\n📋 Portfolio Summary:")
        for ticker, data in results.items():
            snapshot = data.get("snapshot", {})
            prediction = data.get("prediction", {})
            
            if snapshot.get("status") == "success":
                snap_data = snapshot.get("data", {})
                price = snap_data.get("last_close", 0)
                change = snap_data.get("change_pct", 0)
                
                if prediction.get("status") == "success":
                    pred_data = prediction.get("result", {})
                    pred = pred_data.get("prediction_result", {}).get("prediction", {})
                    direction = pred.get("direction", "N/A")
                    confidence = pred.get("confidence", 0)
                    
                    print(f"   {ticker}: ${price:.2f} ({change:+.2f}%) → {direction} ({confidence:.1f}%)")
                else:
                    print(f"   {ticker}: ${price:.2f} ({change:+.2f}%) → Prediction failed")


# Example 3: Trading Signal Generator
async def example_trading_signals():
    """Example: Generate trading signals based on predictions."""
    print("\n🎯 Example 3: Trading Signal Generator")
    print("-" * 40)
    
    watchlist = ["AAPL", "MSFT", "GOOGL"]
    signals = []
    
    async with Stock4UClient() as client:
        for ticker in watchlist:
            # Get prediction
            prediction = await client.run_prediction(ticker, low_api_mode=True)
            
            if prediction.get("status") == "success":
                pred_data = prediction.get("result", {})
                pred = pred_data.get("prediction_result", {}).get("prediction", {})
                
                direction = pred.get("direction", "")
                confidence = pred.get("confidence", 0)
                
                # Generate signal based on confidence threshold
                if confidence > 70:
                    if direction == "UP":
                        signal = "BUY"
                    elif direction == "DOWN":
                        signal = "SELL"
                    else:
                        signal = "HOLD"
                    
                    signals.append({
                        "ticker": ticker,
                        "signal": signal,
                        "confidence": confidence,
                        "direction": direction
                    })
        
        # Display signals
        print(f"📊 Trading Signals (Confidence > 70%):")
        for signal in signals:
            print(f"   {signal['ticker']}: {signal['signal']} "
                  f"({signal['confidence']:.1f}% confidence, {signal['direction']})")


# Example 4: Market Research Report
async def example_research_report():
    """Example: Generate a comprehensive research report."""
    print("\n📋 Example 4: Market Research Report")
    print("-" * 40)
    
    ticker = "TSLA"
    
    async with Stock4UClient() as client:
        print(f"🔍 Generating research report for {ticker}...")
        
        # Get historical data
        data = await client.get_stock_data(ticker, "3mo")
        
        # Get market snapshot
        snapshot = await client.get_market_snapshot(ticker)
        
        # Get prediction
        prediction = await client.run_prediction(ticker, low_api_mode=False)
        
        # Generate report
        print(f"\n📄 Research Report: {ticker}")
        print("=" * 50)
        
        if data.get("status") == "success":
            data_rows = data.get("data", {}).get("rows", [])
            if data_rows:
                latest = data_rows[-1]
                print(f"Current Price: ${latest.get('Close', 'N/A'):.2f}")
                print(f"Data Points: {len(data_rows)} (3 months)")
        
        if snapshot.get("status") == "success":
            snap_data = snapshot.get("data", {})
            indicators = snap_data.get("indicators", {})
            print(f"Market Indicators:")
            print(f"  RSI: {indicators.get('rsi', 'N/A'):.1f}")
            print(f"  SMA20: ${indicators.get('sma_20', 'N/A'):.2f}")
            print(f"  SMA50: ${indicators.get('sma_50', 'N/A'):.2f}")
        
        if prediction.get("status") == "success":
            pred_data = prediction.get("result", {})
            pred = pred_data.get("prediction_result", {}).get("prediction", {})
            risk = pred_data.get("risk_assessment", {})
            
            print(f"Prediction Analysis:")
            print(f"  Direction: {pred.get('direction', 'N/A')}")
            print(f"  Confidence: {pred.get('confidence', 'N/A'):.1f}%")
            print(f"  Risk Level: {risk.get('risk_level', 'N/A')}")
            print(f"  Risk Score: {risk.get('risk_score', 'N/A'):.1f}")


# Example 5: Real-time Monitoring
async def example_real_time_monitoring():
    """Example: Set up real-time monitoring for multiple stocks."""
    print("\n⏰ Example 5: Real-time Monitoring")
    print("-" * 40)
    
    watchlist = ["AAPL", "MSFT", "GOOGL"]
    
    async with Stock4UClient() as client:
        print(f"🔍 Monitoring {len(watchlist)} stocks...")
        
        for ticker in watchlist:
            # Get current market data
            snapshot = await client.get_market_snapshot(ticker)
            
            if snapshot.get("status") == "success":
                data = snapshot.get("data", {})
                price = data.get("last_close", 0)
                change = data.get("change_pct", 0)
                volume = data.get("volume", 0)
                
                # Check for significant changes
                if abs(change) > 2.0:  # More than 2% change
                    alert = "🚨 SIGNIFICANT MOVE"
                elif abs(change) > 1.0:  # More than 1% change
                    alert = "⚠️ MODERATE MOVE"
                else:
                    alert = "📊 NORMAL"
                
                print(f"   {ticker}: ${price:.2f} ({change:+.2f}%) "
                      f"Vol: {volume:,} {alert}")


# Example 6: Performance Comparison
async def example_performance_comparison():
    """Example: Compare performance of different prediction modes."""
    print("\n⚡ Example 6: Performance Comparison")
    print("-" * 40)
    
    ticker = "AAPL"
    
    async with Stock4UClient() as client:
        import time
        
        # Test low API mode
        start_time = time.time()
        low_api_result = await client.run_prediction(ticker, low_api_mode=True, fast_ta_mode=True)
        low_api_time = time.time() - start_time
        
        # Test full mode
        start_time = time.time()
        full_result = await client.run_prediction(ticker, low_api_mode=False, fast_ta_mode=False)
        full_time = time.time() - start_time
        
        print(f"⏱️ Performance Comparison for {ticker}:")
        print(f"   Low API Mode: {low_api_time:.2f} seconds")
        print(f"   Full Mode: {full_time:.2f} seconds")
        print(f"   Speed Improvement: {(full_time/low_api_time):.1f}x faster")
        
        # Compare results
        if low_api_result.get("status") == "success" and full_result.get("status") == "success":
            low_pred = low_api_result.get("result", {}).get("prediction_result", {}).get("prediction", {})
            full_pred = full_result.get("result", {}).get("prediction_result", {}).get("prediction", {})
            
            print(f"\n📊 Result Comparison:")
            print(f"   Low API: {low_pred.get('direction', 'N/A')} ({low_pred.get('confidence', 'N/A'):.1f}%)")
            print(f"   Full: {full_pred.get('direction', 'N/A')} ({full_pred.get('confidence', 'N/A'):.1f}%)")


async def main():
    """Run all examples."""
    print("🚀 Stock4U MCP Client Examples")
    print("=" * 60)
    
    try:
        # Run examples
        await example_simple_analysis()
        await example_portfolio_analysis()
        await example_trading_signals()
        await example_research_report()
        await example_real_time_monitoring()
        await example_performance_comparison()
        
        print(f"\n🎉 All examples completed successfully!")
        print(f"💡 These examples demonstrate the flexibility and power of the Stock4U MCP server.")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print(f"💡 Make sure the MCP server is running: python -m agents.mcp_server")


if __name__ == "__main__":
    asyncio.run(main())
