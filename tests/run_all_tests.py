#!/usr/bin/env python3
"""
🧪 Test Runner for Agentic Stock Predictor
Run all tests in the tests folder in an organized manner.
"""

import os
import sys
import importlib.util
from pathlib import Path
from typing import List, Dict, Any
import time
import io

# Add the parent directory to the path so we can import modules
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables from .env if present (does not override existing env)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(override=False)
except Exception:
    pass

# Ensure UTF-8 capable stdout/stderr so emoji output doesn't crash on Windows consoles
try:
    # Python 3.7+: reconfigure available
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        # Final fallback: define a safe_print wrapper
        def _safe_print(*args, **kwargs):
            text = " ".join(str(a) for a in args)
            try:
                print(text.encode("utf-8", errors="replace").decode("utf-8"), **kwargs)
            except Exception:
                pass
        print = _safe_print  # type: ignore

def run_test_module(module_name: str) -> Dict[str, Any]:
    """
    Run a specific test module and return results.
    
    Args:
        module_name: Name of the test module (without .py)
        
    Returns:
        Dictionary with test results
    """
    try:
        print(f"\n{'='*60}")
        print(f"🧪 Running {module_name}")
        print(f"{'='*60}")
        
        # Import and run the test module
        spec = importlib.util.spec_from_file_location(
            module_name, 
            f"tests/{module_name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # If the module has a main function, run it
        if hasattr(module, 'main'):
            start_time = time.time()
            module.main()
            end_time = time.time()
            
            return {
                "module": module_name,
                "status": "success",
                "duration": end_time - start_time,
                "error": None
            }
        else:
            return {
                "module": module_name,
                "status": "skipped",
                "duration": 0,
                "error": "No main() function found"
            }
            
    except Exception as e:
        return {
            "module": module_name,
            "status": "failed",
            "duration": 0,
            "error": str(e)
        }

def run_all_tests() -> None:
    """
    Run all test modules in the tests folder.
    """
    print("🚀 Agentic Stock Predictor - Test Suite")
    print("=" * 60)
    
    # Get all test files
    test_files = [
        "test_orchestrator_tools",
        "test_tavily", 
        "test_chatbot",
        "test_integrated_workflow"
    ]
    
    results = []
    total_start_time = time.time()
    
    for test_file in test_files:
        result = run_test_module(test_file)
        results.append(result)
        
        # Print result summary
        status_emoji = {
            "success": "✅",
            "failed": "❌", 
            "skipped": "⚠️"
        }
        
        emoji = status_emoji.get(result["status"], "❓")
        print(f"{emoji} {result['module']}: {result['status']}")
        
        if result["duration"] > 0:
            print(f"   Duration: {result['duration']:.2f}s")
        
        if result["error"]:
            print(f"   Error: {result['error']}")
    
    total_duration = time.time() - total_start_time
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 Test Summary")
    print(f"{'='*60}")
    
    successful = len([r for r in results if r["status"] == "success"])
    failed = len([r for r in results if r["status"] == "failed"])
    skipped = len([r for r in results if r["status"] == "skipped"])
    
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"⚠️ Skipped: {skipped}")
    print(f"⏱️ Total Duration: {total_duration:.2f}s")
    
    if failed == 0:
        print(f"\n🎉 All tests passed!")
    else:
        print(f"\n⚠️ {failed} test(s) failed. Check the output above for details.")
    
    # Print detailed results
    print(f"\n📋 Detailed Results:")
    for result in results:
        status_emoji = {
            "success": "✅",
            "failed": "❌",
            "skipped": "⚠️"
        }
        emoji = status_emoji.get(result["status"], "❓")
        print(f"   {emoji} {result['module']}: {result['status']}")

def run_specific_test(test_name: str) -> None:
    """
    Run a specific test by name.
    
    Args:
        test_name: Name of the test to run (without .py)
    """
    print(f"🎯 Running specific test: {test_name}")
    result = run_test_module(test_name)
    
    status_emoji = {
        "success": "✅",
        "failed": "❌",
        "skipped": "⚠️"
    }
    
    emoji = status_emoji.get(result["status"], "❓")
    print(f"\n{emoji} {result['module']}: {result['status']}")
    
    if result["duration"] > 0:
        print(f"Duration: {result['duration']:.2f}s")
    
    if result["error"]:
        print(f"Error: {result['error']}")

def list_available_tests() -> None:
    """
    List all available test modules.
    """
    print("📋 Available Tests:")
    print("=" * 30)
    
    test_files = [
        ("test_orchestrator_tools", "Test orchestrator tools functionality"),
        ("test_tavily", "Test Tavily search integration"),
        ("test_chatbot", "Test chatbot functionality"),
        ("test_integrated_workflow", "Test complete workflow integration")
    ]
    
    for test_file, description in test_files:
        print(f"  • {test_file}: {description}")

def main():
    """
    Main test runner function.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Run tests for Agentic Stock Predictor")
    parser.add_argument("--test", "-t", help="Run a specific test")
    parser.add_argument("--list", "-l", action="store_true", help="List available tests")
    
    args = parser.parse_args()
    
    if args.list:
        list_available_tests()
    elif args.test:
        run_specific_test(args.test)
    else:
        run_all_tests()

if __name__ == "__main__":
    main() 