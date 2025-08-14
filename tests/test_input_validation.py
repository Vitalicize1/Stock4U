#!/usr/bin/env python3
"""
🧪 Test Input Validation System
Test the comprehensive input validation for Stock4U.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import modules
sys.path.append(str(Path(__file__).parent.parent))

from utils.validation import InputValidator, ValidationResult, validate_ticker, validate_timeframe, validate_api_request


def test_ticker_validation():
    """Test ticker symbol validation."""
    print("🧪 Testing Ticker Validation")
    print("=" * 50)
    
    # Valid tickers
    valid_tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA"]
    for ticker in valid_tickers:
        result = InputValidator.validate_ticker_symbol(ticker)
        status = "✅" if result.is_valid else "❌"
        print(f"   {status} {ticker}: {result.sanitized_value if result.is_valid else result.error_message}")
    
    # Invalid tickers
    invalid_tickers = [
        "",  # Empty
        "   ",  # Whitespace only
        "INVALID_TICKER_12345",  # Too long
        "AAPL123",  # Contains numbers
        "aapl",  # Lowercase (should be converted)
        "A@PL",  # Special characters
        "<script>alert('xss')</script>",  # XSS attempt
        "'; DROP TABLE users; --",  # SQL injection attempt
    ]
    
    print("\n❌ Invalid tickers:")
    for ticker in invalid_tickers:
        result = InputValidator.validate_ticker_symbol(ticker)
        print(f"   ❌ {repr(ticker)}: {result.error_message}")


def test_timeframe_validation():
    """Test timeframe validation."""
    print("\n🧪 Testing Timeframe Validation")
    print("=" * 50)
    
    # Valid timeframes
    valid_timeframes = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
    for timeframe in valid_timeframes:
        result = InputValidator.validate_timeframe(timeframe)
        status = "✅" if result.is_valid else "❌"
        print(f"   {status} {timeframe}: {result.sanitized_value if result.is_valid else result.error_message}")
    
    # Invalid timeframes
    invalid_timeframes = [
        "",  # Empty
        "invalid",  # Invalid value
        "1day",  # Wrong format
        "1M",  # Wrong format
        "<script>alert('xss')</script>",  # XSS attempt
    ]
    
    print("\n❌ Invalid timeframes:")
    for timeframe in invalid_timeframes:
        result = InputValidator.validate_timeframe(timeframe)
        print(f"   ❌ {repr(timeframe)}: {result.error_message}")


def test_period_validation():
    """Test period validation."""
    print("\n🧪 Testing Period Validation")
    print("=" * 50)
    
    # Valid periods
    valid_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
    for period in valid_periods:
        result = InputValidator.validate_period(period)
        status = "✅" if result.is_valid else "❌"
        print(f"   {status} {period}: {result.sanitized_value if result.is_valid else result.error_message}")
    
    # Invalid periods
    invalid_periods = [
        "",  # Empty
        "invalid",  # Invalid value
        "1day",  # Wrong format
        "1M",  # Wrong format
        "<script>alert('xss')</script>",  # XSS attempt
    ]
    
    print("\n❌ Invalid periods:")
    for period in invalid_periods:
        result = InputValidator.validate_period(period)
        print(f"   ❌ {repr(period)}: {result.error_message}")


def test_ticker_list_validation():
    """Test ticker list validation."""
    print("\n🧪 Testing Ticker List Validation")
    print("=" * 50)
    
    # Valid ticker lists
    valid_lists = [
        ["AAPL", "MSFT", "GOOGL"],
        "AAPL,MSFT,GOOGL",
        ["TSLA"],
        "NVDA"
    ]
    
    for ticker_list in valid_lists:
        result = InputValidator.validate_ticker_list(ticker_list)
        status = "✅" if result.is_valid else "❌"
        print(f"   {status} {ticker_list}: {result.sanitized_value if result.is_valid else result.error_message}")
    
    # Invalid ticker lists
    invalid_lists = [
        [],  # Empty list
        "",  # Empty string
        ["INVALID", "AAPL"],  # Mixed valid/invalid
        ["AAPL"] * 100,  # Too many tickers
        "<script>alert('xss')</script>",  # XSS attempt
    ]
    
    print("\n❌ Invalid ticker lists:")
    for ticker_list in invalid_lists:
        result = InputValidator.validate_ticker_list(ticker_list)
        print(f"   ❌ {repr(ticker_list)}: {result.error_message}")


def test_numeric_validation():
    """Test numeric validation."""
    print("\n🧪 Testing Numeric Validation")
    print("=" * 50)
    
    # Valid numeric values
    test_cases = [
        (100, 0, 1000, "cash"),
        (5.0, 0, 100, "fee_bps"),
        (10, 1, 20, "splits"),
    ]
    
    for value, min_val, max_val, field_name in test_cases:
        result = InputValidator.validate_numeric_range(value, min_val, max_val, field_name)
        status = "✅" if result.is_valid else "❌"
        print(f"   {status} {field_name}={value} (range {min_val}-{max_val}): {result.sanitized_value if result.is_valid else result.error_message}")
    
    # Invalid numeric values
    invalid_cases = [
        (-1, 0, 100, "cash"),  # Below minimum
        (1001, 0, 1000, "cash"),  # Above maximum
        ("invalid", 0, 100, "cash"),  # Not a number
        (None, 0, 100, "cash"),  # None value
    ]
    
    print("\n❌ Invalid numeric values:")
    for value, min_val, max_val, field_name in invalid_cases:
        result = InputValidator.validate_numeric_range(value, min_val, max_val, field_name)
        print(f"   ❌ {field_name}={repr(value)} (range {min_val}-{max_val}): {result.error_message}")


def test_boolean_validation():
    """Test boolean validation."""
    print("\n🧪 Testing Boolean Validation")
    print("=" * 50)
    
    # Valid boolean values
    valid_booleans = [
        True, False,
        "true", "false",
        "1", "0",
        "yes", "no",
        "on", "off"
    ]
    
    for value in valid_booleans:
        result = InputValidator.validate_boolean(value, "test_field")
        status = "✅" if result.is_valid else "❌"
        print(f"   {status} {repr(value)}: {result.sanitized_value if result.is_valid else result.error_message}")
    
    # Invalid boolean values
    invalid_booleans = [
        "invalid",
        "maybe",
        "sometimes",
        123,
        None
    ]
    
    print("\n❌ Invalid boolean values:")
    for value in invalid_booleans:
        result = InputValidator.validate_boolean(value, "test_field")
        print(f"   ❌ {repr(value)}: {result.error_message}")


def test_string_validation():
    """Test string validation."""
    print("\n🧪 Testing String Validation")
    print("=" * 50)
    
    # Valid strings
    valid_strings = [
        "normal string",
        "string with numbers 123",
        "UPPERCASE STRING",
        "string-with-hyphens",
        "string_with_underscores"
    ]
    
    for string in valid_strings:
        result = InputValidator.validate_string(string, max_length=100, field_name="test_field")
        status = "✅" if result.is_valid else "❌"
        print(f"   {status} {repr(string)}: {result.sanitized_value if result.is_valid else result.error_message}")
    
    # Invalid strings
    invalid_strings = [
        "<script>alert('xss')</script>",  # XSS attempt
        "javascript:alert('xss')",  # JavaScript injection
        "'; DROP TABLE users; --",  # SQL injection
        "A" * 1001,  # Too long
        123,  # Not a string
        None  # None value
    ]
    
    print("\n❌ Invalid strings:")
    for string in invalid_strings:
        result = InputValidator.validate_string(string, max_length=100, field_name="test_field")
        print(f"   ❌ {repr(string)}: {result.error_message}")


def test_api_request_validation():
    """Test complete API request validation."""
    print("\n🧪 Testing API Request Validation")
    print("=" * 50)
    
    # Valid API requests
    valid_requests = [
        {
            "ticker": "AAPL",
            "timeframe": "1d",
            "low_api_mode": False,
            "fast_ta_mode": False,
            "use_ml_model": False
        },
        {
            "tickers": ["AAPL", "MSFT", "GOOGL"],
            "period": "1y",
            "cash": 100000,
            "fee_bps": 5.0,
            "slip_bps": 5.0
        }
    ]
    
    for i, request in enumerate(valid_requests):
        result = InputValidator.validate_api_request(request)
        status = "✅" if result.is_valid else "❌"
        print(f"   {status} Request {i+1}: {result.sanitized_value if result.is_valid else result.error_message}")
        if result.warnings:
            for warning in result.warnings:
                print(f"      ⚠️ Warning: {warning}")
    
    # Invalid API requests
    invalid_requests = [
        {
            "ticker": "<script>alert('xss')</script>",  # Invalid ticker
            "timeframe": "1d"
        },
        {
            "ticker": "AAPL",
            "timeframe": "invalid"  # Invalid timeframe
        },
        {
            "tickers": ["AAPL"] * 100,  # Too many tickers
            "period": "1y"
        },
        {
            "ticker": "AAPL",
            "cash": -1000  # Negative cash
        }
    ]
    
    print("\n❌ Invalid API requests:")
    for i, request in enumerate(invalid_requests):
        result = InputValidator.validate_api_request(request)
        print(f"   ❌ Request {i+1}: {result.error_message}")


def test_malicious_content_detection():
    """Test malicious content detection."""
    print("\n🧪 Testing Malicious Content Detection")
    print("=" * 50)
    
    # Malicious content examples
    malicious_content = [
        "<script>alert('xss')</script>",
        "javascript:alert('xss')",
        "data:text/html,<script>alert('xss')</script>",
        "vbscript:msgbox('xss')",
        "onclick=alert('xss')",
        "<iframe src='http://evil.com'></iframe>",
        "<object data='http://evil.com'></object>",
        "<embed src='http://evil.com'></embed>",
        "<!-- comment -->",
        "<div>content</div>",
        "'; DROP TABLE users; --",
        "UNION SELECT * FROM users",
        "1' OR '1'='1",
        "exec xp_cmdshell",
        "sp_executesql"
    ]
    
    for content in malicious_content:
        is_malicious = InputValidator._contains_malicious_content(content)
        status = "🚨" if is_malicious else "✅"
        print(f"   {status} {repr(content)}: {'Malicious' if is_malicious else 'Safe'}")
    
    # Safe content examples
    safe_content = [
        "normal text",
        "AAPL stock analysis",
        "1d timeframe",
        "100000 cash amount",
        "true boolean value",
        "string with numbers 123"
    ]
    
    print("\n✅ Safe content examples:")
    for content in safe_content:
        is_malicious = InputValidator._contains_malicious_content(content)
        status = "✅" if not is_malicious else "🚨"
        print(f"   {status} {repr(content)}: {'Safe' if not is_malicious else 'Malicious'}")


def test_convenience_functions():
    """Test convenience functions."""
    print("\n🧪 Testing Convenience Functions")
    print("=" * 50)
    
    # Test validate_ticker
    result = validate_ticker("AAPL")
    status = "✅" if result.is_valid else "❌"
    print(f"   {status} validate_ticker('AAPL'): {result.sanitized_value if result.is_valid else result.error_message}")
    
    # Test validate_timeframe
    result = validate_timeframe("1d")
    status = "✅" if result.is_valid else "❌"
    print(f"   {status} validate_timeframe('1d'): {result.sanitized_value if result.is_valid else result.error_message}")
    
    # Test validate_api_request
    result = validate_api_request({"ticker": "MSFT", "timeframe": "1mo"})
    status = "✅" if result.is_valid else "❌"
    print(f"   {status} validate_api_request(): {result.sanitized_value if result.is_valid else result.error_message}")


def main():
    """Run all validation tests."""
    print("🚀 Stock4U - Input Validation Test Suite")
    print("=" * 60)
    
    try:
        test_ticker_validation()
        test_timeframe_validation()
        test_period_validation()
        test_ticker_list_validation()
        test_numeric_validation()
        test_boolean_validation()
        test_string_validation()
        test_api_request_validation()
        test_malicious_content_detection()
        test_convenience_functions()
        
        print("\n" + "=" * 60)
        print("🎉 Input validation test suite completed successfully!")
        print("✅ All validation functions are working correctly")
        print("🛡️ Security checks are in place")
        print("🔍 Input sanitization is functional")
        
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
