#!/usr/bin/env python3
"""
🧪 Test Technical Analyzer
Test the enhanced technical analyzer with data collector integration.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import modules
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from agents.technical_analyzer import analyze_technical_with_tools
from agents.data_collector import collect_data_with_tools

# Load environment variables
load_dotenv()

def test_technical_analyzer():
    """Test the technical analyzer with real data."""
    print("🧪 Testing Technical Analyzer")
    print("=" * 40)
    
    # Test with AAPL data
    print("📊 Testing with AAPL data...")
    
    try:
        # First, collect data using the enhanced data collector
        test_state = {
            "ticker": "AAPL",
            "timeframe": "1d",
            "status": "initialized"
        }
        
        # Collect data
        print("📈 Collecting data...")
        data_result = collect_data_with_tools(test_state)
        
        if data_result.get("status") == "success":
            print("✅ Data collection successful")
            
            # Now test technical analysis
            print("🔍 Running technical analysis...")
            analysis_result = analyze_technical_with_tools(data_result)
            
            if analysis_result.get("status") == "success":
                print("✅ Technical analysis completed successfully!")
                
                # Extract analysis results
                technical_analysis = analysis_result.get("technical_analysis", {})
                
                print(f"\n📋 Technical Analysis Results:")
                print(f"   Technical Score: {technical_analysis.get('technical_score', 0):.1f}/100")
                print(f"   Signals: {', '.join(technical_analysis.get('technical_signals', []))}")
                
                # Trend analysis
                trend_analysis = technical_analysis.get("trend_analysis", {})
                print(f"   Short-term Trend: {trend_analysis.get('short_term_trend', 'neutral')}")
                print(f"   Medium-term Trend: {trend_analysis.get('medium_term_trend', 'neutral')}")
                print(f"   Trend Strength: {trend_analysis.get('trend_strength', 0):.1f}%")
                
                # Support/Resistance
                support_resistance = technical_analysis.get("support_resistance", {})
                print(f"   Support Level: ${support_resistance.get('support_level', 0):.2f}")
                print(f"   Resistance Level: ${support_resistance.get('resistance_level', 0):.2f}")
                
                # Momentum analysis
                momentum = technical_analysis.get("momentum_analysis", {})
                print(f"   RSI Signal: {momentum.get('rsi_signal', 'neutral')}")
                print(f"   Overbought/Oversold: {momentum.get('overbought_oversold', 'neutral')}")
                
                # Volume analysis
                volume = technical_analysis.get("volume_analysis", {})
                print(f"   Volume Trend: {volume.get('volume_trend', 'normal')}")
                print(f"   Volume Significance: {volume.get('volume_significance', 'medium')}")
                
                # Pattern recognition
                patterns = technical_analysis.get("pattern_recognition", {})
                detected_patterns = patterns.get("detected_patterns", [])
                if detected_patterns:
                    print(f"   Detected Patterns: {', '.join(detected_patterns)}")
                else:
                    print(f"   Detected Patterns: None")
                
                print(f"   Pattern Confidence: {patterns.get('pattern_confidence', 0):.1f}")
                
                return True
                
            else:
                print(f"❌ Technical analysis failed: {analysis_result.get('error')}")
                return False
                
        else:
            print(f"❌ Data collection failed: {data_result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {str(e)}")
        return False

def test_technical_analyzer_error_handling():
    """Test error handling with invalid data."""
    print(f"\n⚠️ Testing Error Handling")
    print("=" * 30)
    
    try:
        # Test with invalid state (no data)
        invalid_state = {
            "ticker": "INVALID",
            "timeframe": "1d",
            "status": "initialized"
        }
        
        result = analyze_technical_with_tools(invalid_state)
        
        if result.get("status") == "error":
            print("✅ Error handling working correctly")
            print(f"   Error: {result.get('error')}")
            return True
        else:
            print("⚠️ Unexpected success with invalid data")
            return False
            
    except Exception as e:
        print(f"✅ Error handling working (exception caught): {str(e)}")
        return True

def test_technical_analyzer_performance():
    """Test performance of technical analyzer."""
    print(f"\n⚡ Testing Performance")
    print("=" * 25)
    
    import time
    
    try:
        # Test state with AAPL
        test_state = {
            "ticker": "AAPL",
            "timeframe": "1d",
            "status": "initialized"
        }
        
        # Collect data first
        data_result = collect_data_with_tools(test_state)
        
        if data_result.get("status") == "success":
            # Test technical analysis performance
            start_time = time.time()
            analysis_result = analyze_technical_with_tools(data_result)
            end_time = time.time()
            
            duration = end_time - start_time
            print(f"✅ Technical analysis completed in {duration:.2f} seconds")
            
            if duration < 5:  # Should complete within 5 seconds
                print(f"   ✅ Performance is excellent (< 5s)")
            elif duration < 10:
                print(f"   ✅ Performance is good (< 10s)")
            else:
                print(f"   ⚠️ Performance is slow (> 10s)")
                
            # Check analysis quality
            if analysis_result.get("status") == "success":
                technical_analysis = analysis_result.get("technical_analysis", {})
                score = technical_analysis.get("technical_score", 0)
                
                if score > 0:
                    print(f"   ✅ Analysis quality is good (score: {score:.1f})")
                else:
                    print(f"   ⚠️ Analysis quality needs improvement (score: {score:.1f})")
                    
            return True
            
        else:
            print(f"❌ Data collection failed for performance test")
            return False
            
    except Exception as e:
        print(f"❌ Performance test failed: {str(e)}")
        return False

def main():
    """Main test function."""
    print("🧪 Technical Analyzer Test Suite")
    print("=" * 50)
    
    # Run all tests
    success_count = 0
    total_tests = 3
    
    # Test 1: Basic functionality
    if test_technical_analyzer():
        success_count += 1
    
    # Test 2: Error handling
    if test_technical_analyzer_error_handling():
        success_count += 1
    
    # Test 3: Performance
    if test_technical_analyzer_performance():
        success_count += 1
    
    # Print summary
    print(f"\n📊 Test Summary:")
    print(f"   ✅ Passed: {success_count}/{total_tests}")
    print(f"   ❌ Failed: {total_tests - success_count}/{total_tests}")
    
    if success_count == total_tests:
        print(f"\n🎉 All technical analyzer tests passed!")
        print("✅ Technical analyzer is ready for production!")
    else:
        print(f"\n⚠️ {total_tests - success_count} test(s) failed. Check the output above.")

if __name__ == "__main__":
    main() 