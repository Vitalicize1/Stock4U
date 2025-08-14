#!/usr/bin/env python3
"""Test suite for Stock4U MCP server."""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from mcp.client.stdio import stdio_client
except ImportError:
    print("❌ MCP client not installed. Install with: pip install 'mcp[cli]==1.12.4'")
    sys.exit(1)


class MCPTester:
    """Test suite for Stock4U MCP server."""
    
    def __init__(self):
        self.client = None
        self.test_results = []
    
    async def __aenter__(self):
        from mcp.client.stdio import StdioServerParameters
        server_params = StdioServerParameters(command="python", args=["-m", "agents.mcp_server"])
        self.client = await stdio_client(server_params).__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.__aexit__(exc_type, exc_val, exc_tb)
    
    def log_test(self, test_name: str, success: bool, duration: float, error: str = None):
        """Log test result."""
        result = {
            "test": test_name,
            "success": success,
            "duration": duration,
            "error": error
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name} ({duration:.2f}s)")
        if error:
            print(f"   Error: {error}")
    
    async def test_ping(self):
        """Test ping functionality."""
        start_time = time.time()
        try:
            result = await self.client.call_tool("ping", {})
            duration = time.time() - start_time
            
            success = result.get("status") == "ok" and "timestamp" in result
            self.log_test("Ping", success, duration)
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Ping", False, duration, str(e))
    
    async def test_get_market_snapshot(self):
        """Test market snapshot functionality."""
        start_time = time.time()
        try:
            result = await self.client.call_tool("get_market_snapshot", {"ticker": "AAPL"})
            duration = time.time() - start_time
            
            success = (result.get("status") == "success" and 
                      "data" in result and 
                      "last_close" in result.get("data", {}))
            self.log_test("Market Snapshot", success, duration)
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Market Snapshot", False, duration, str(e))
    
    async def test_get_stock_data(self):
        """Test stock data retrieval."""
        start_time = time.time()
        try:
            result = await self.client.call_tool("get_stock_data", {
                "ticker": "MSFT",
                "period": "1mo"
            })
            duration = time.time() - start_time
            
            success = (result.get("status") == "success" and 
                      "data" in result and 
                      "rows" in result.get("data", {}))
            self.log_test("Stock Data", success, duration)
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Stock Data", False, duration, str(e))
    
    async def test_run_prediction_low_api(self):
        """Test prediction with low API mode."""
        start_time = time.time()
        try:
            result = await self.client.call_tool("run_stock_prediction", {
                "ticker": "GOOGL",
                "timeframe": "1d",
                "low_api_mode": True,
                "fast_ta_mode": True,
                "use_ml_model": False
            })
            duration = time.time() - start_time
            
            success = (result.get("status") == "success" and 
                      "result" in result and 
                      "prediction_result" in result.get("result", {}))
            self.log_test("Prediction (Low API)", success, duration)
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Prediction (Low API)", False, duration, str(e))
    
    async def test_run_prediction_full(self):
        """Test prediction with full mode."""
        start_time = time.time()
        try:
            result = await self.client.call_tool("run_stock_prediction", {
                "ticker": "TSLA",
                "timeframe": "1d",
                "low_api_mode": False,
                "fast_ta_mode": False,
                "use_ml_model": False
            })
            duration = time.time() - start_time
            
            success = (result.get("status") == "success" and 
                      "result" in result and 
                      "prediction_result" in result.get("result", {}))
            self.log_test("Prediction (Full)", success, duration)
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Prediction (Full)", False, duration, str(e))
    
    async def test_cache_operations(self):
        """Test cache operations."""
        start_time = time.time()
        try:
            # Test cache invalidation
            invalidate_result = await self.client.call_tool("invalidate_cache", {"cache_key": "test_key"})
            
            # Test cache retrieval
            get_result = await self.client.call_tool("get_cached_result", {
                "cache_key": "test_key",
                "ttl_seconds": 900
            })
            
            duration = time.time() - start_time
            
            success = (invalidate_result.get("status") == "success" and 
                      get_result.get("status") == "success")
            self.log_test("Cache Operations", success, duration)
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Cache Operations", False, duration, str(e))
    
    async def test_error_handling(self):
        """Test error handling with invalid inputs."""
        start_time = time.time()
        try:
            # Test with invalid ticker
            result = await self.client.call_tool("get_market_snapshot", {"ticker": "INVALID_TICKER_12345"})
            duration = time.time() - start_time
            
            # Should handle error gracefully
            success = result.get("status") == "error" and "error" in result
            self.log_test("Error Handling", success, duration)
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Error Handling", False, duration, str(e))
    
    async def test_performance(self):
        """Test performance benchmarks."""
        start_time = time.time()
        try:
            # Test multiple operations
            operations = [
                ("ping", {}),
                ("get_market_snapshot", {"ticker": "AAPL"}),
                ("get_stock_data", {"ticker": "MSFT", "period": "5d"}),
            ]
            
            for op_name, params in operations:
                op_start = time.time()
                await self.client.call_tool(op_name, params)
                op_duration = time.time() - op_start
                print(f"   {op_name}: {op_duration:.2f}s")
            
            duration = time.time() - start_time
            self.log_test("Performance", True, duration)
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Performance", False, duration, str(e))
    
    async def test_concurrent_operations(self):
        """Test concurrent operations."""
        start_time = time.time()
        try:
            # Run multiple operations concurrently
            tasks = [
                self.client.call_tool("get_market_snapshot", {"ticker": "AAPL"}),
                self.client.call_tool("get_market_snapshot", {"ticker": "MSFT"}),
                self.client.call_tool("get_market_snapshot", {"ticker": "GOOGL"}),
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            duration = time.time() - start_time
            
            # Check if all operations succeeded
            success = all(
                isinstance(result, dict) and result.get("status") == "success"
                for result in results
            )
            self.log_test("Concurrent Operations", success, duration)
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Concurrent Operations", False, duration, str(e))
    
    def print_summary(self):
        """Print test summary."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - passed_tests
        
        total_duration = sum(result["duration"] for result in self.test_results)
        avg_duration = total_duration / total_tests if total_tests > 0 else 0
        
        print(f"\n📊 Test Summary")
        print(f"=" * 50)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        print(f"Total Duration: {total_duration:.2f}s")
        print(f"Average Duration: {avg_duration:.2f}s")
        
        if failed_tests > 0:
            print(f"\n❌ Failed Tests:")
            for result in self.test_results:
                if not result["success"]:
                    print(f"   - {result['test']}: {result['error']}")
        
        return passed_tests == total_tests


async def main():
    """Run all MCP tests."""
    print("🧪 Stock4U MCP Server Test Suite")
    print("=" * 50)
    
    try:
        async with MCPTester() as tester:
            print("🔍 Running tests...\n")
            
            # Run all tests
            await tester.test_ping()
            await tester.test_get_market_snapshot()
            await tester.test_get_stock_data()
            await tester.test_run_prediction_low_api()
            await tester.test_run_prediction_full()
            await tester.test_cache_operations()
            await tester.test_error_handling()
            await tester.test_performance()
            await tester.test_concurrent_operations()
            
            # Print summary
            all_passed = tester.print_summary()
            
            if all_passed:
                print(f"\n🎉 All tests passed! MCP server is working correctly.")
            else:
                print(f"\n⚠️ Some tests failed. Check the errors above.")
            
            return all_passed
            
    except Exception as e:
        print(f"❌ Test suite failed to run: {e}")
        print(f"💡 Make sure the MCP server is running: python -m agents.mcp_server")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
