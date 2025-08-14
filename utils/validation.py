"""
Comprehensive Input Validation for Stock4U

This module provides centralized validation for all user inputs, API parameters,
and data across the Stock4U application. It includes security checks, format
validation, and business logic validation.
"""

import re
import os
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import yfinance as yf


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    is_valid: bool
    error_message: Optional[str] = None
    sanitized_value: Optional[Any] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class InputValidator:
    """Comprehensive input validator for Stock4U."""
    
    # Valid timeframes for analysis
    VALID_TIMEFRAMES = {
        "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"
    }
    
    # Valid intervals for data collection
    VALID_INTERVALS = {
        "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"
    }
    
    # Maximum allowed values for various parameters
    MAX_VALUES = {
        "ticker_length": 10,
        "timeframe_length": 10,
        "period_length": 10,
        "max_tickers_per_request": 50,
        "max_cash_amount": 1_000_000_000,  # 1 billion
        "max_fee_bps": 1000,  # 10%
        "max_slip_bps": 1000,  # 10%
        "max_string_length": 1000,
        "max_list_length": 100,
    }
    
    # Common malicious patterns to block
    MALICIOUS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"data:text/html",
        r"vbscript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>",
        r"<object[^>]*>",
        r"<embed[^>]*>",
        r"<link[^>]*>",
        r"<meta[^>]*>",
        r"<!--.*?-->",
        r"<[^>]*>",
    ]
    
    @classmethod
    def validate_ticker_symbol(cls, ticker: str) -> ValidationResult:
        """
        Validate a stock ticker symbol.
        
        Args:
            ticker: Stock ticker symbol to validate
            
        Returns:
            ValidationResult with validation status and any errors
        """
        if not ticker:
            return ValidationResult(
                is_valid=False,
                error_message="Ticker symbol is required"
            )
        
        # Convert to string and clean
        ticker_str = str(ticker).strip()
        
        # Check for malicious content
        if cls._contains_malicious_content(ticker_str):
            return ValidationResult(
                is_valid=False,
                error_message="Ticker contains invalid characters"
            )
        
        # Basic format validation
        if not re.match(r'^[A-Z]{1,10}$', ticker_str.upper()):
            return ValidationResult(
                is_valid=False,
                error_message="Ticker must be 1-10 uppercase letters only"
            )
        
        # Length check
        if len(ticker_str) > cls.MAX_VALUES["ticker_length"]:
            return ValidationResult(
                is_valid=False,
                error_message=f"Ticker too long (max {cls.MAX_VALUES['ticker_length']} characters)"
            )
        
        # Try to validate with yfinance (optional, can be disabled for performance)
        if os.getenv("VALIDATE_TICKER_WITH_YFINANCE", "true").lower() == "true":
            try:
                ticker_obj = yf.Ticker(ticker_str.upper())
                info = ticker_obj.info
                
                if not info or info.get('regularMarketPrice') is None:
                    return ValidationResult(
                        is_valid=False,
                        error_message="Ticker not found or invalid",
                        warnings=["Ticker validation failed with yfinance"]
                    )
            except Exception as e:
                return ValidationResult(
                    is_valid=False,
                    error_message="Failed to validate ticker",
                    warnings=[f"yfinance validation error: {str(e)}"]
                )
        
        return ValidationResult(
            is_valid=True,
            sanitized_value=ticker_str.upper()
        )
    
    @classmethod
    def validate_timeframe(cls, timeframe: str) -> ValidationResult:
        """
        Validate a timeframe parameter.
        
        Args:
            timeframe: Timeframe string to validate
            
        Returns:
            ValidationResult with validation status
        """
        if not timeframe:
            return ValidationResult(
                is_valid=False,
                error_message="Timeframe is required"
            )
        
        timeframe_str = str(timeframe).strip().lower()
        
        # Check for malicious content
        if cls._contains_malicious_content(timeframe_str):
            return ValidationResult(
                is_valid=False,
                error_message="Timeframe contains invalid characters"
            )
        
        # Length check
        if len(timeframe_str) > cls.MAX_VALUES["timeframe_length"]:
            return ValidationResult(
                is_valid=False,
                error_message=f"Timeframe too long (max {cls.MAX_VALUES['timeframe_length']} characters)"
            )
        
        # Validate against allowed values
        if timeframe_str not in cls.VALID_TIMEFRAMES:
            return ValidationResult(
                is_valid=False,
                error_message=f"Invalid timeframe. Must be one of: {', '.join(sorted(cls.VALID_TIMEFRAMES))}"
            )
        
        return ValidationResult(
            is_valid=True,
            sanitized_value=timeframe_str
        )
    
    @classmethod
    def validate_interval(cls, interval: str) -> ValidationResult:
        """
        Validate a data interval parameter.
        
        Args:
            interval: Interval string to validate
            
        Returns:
            ValidationResult with validation status
        """
        if not interval:
            return ValidationResult(
                is_valid=False,
                error_message="Interval is required"
            )
        
        interval_str = str(interval).strip().lower()
        
        # Check for malicious content
        if cls._contains_malicious_content(interval_str):
            return ValidationResult(
                is_valid=False,
                error_message="Interval contains invalid characters"
            )
        
        # Validate against allowed values
        if interval_str not in cls.VALID_INTERVALS:
            return ValidationResult(
                is_valid=False,
                error_message=f"Invalid interval. Must be one of: {', '.join(sorted(cls.VALID_INTERVALS))}"
            )
        
        return ValidationResult(
            is_valid=True,
            sanitized_value=interval_str
        )
    
    @classmethod
    def validate_period(cls, period: str) -> ValidationResult:
        """
        Validate a period parameter.
        
        Args:
            period: Period string to validate
            
        Returns:
            ValidationResult with validation status
        """
        if not period:
            return ValidationResult(
                is_valid=False,
                error_message="Period is required"
            )
        
        period_str = str(period).strip().lower()
        
        # Check for malicious content
        if cls._contains_malicious_content(period_str):
            return ValidationResult(
                is_valid=False,
                error_message="Period contains invalid characters"
            )
        
        # Length check
        if len(period_str) > cls.MAX_VALUES["period_length"]:
            return ValidationResult(
                is_valid=False,
                error_message=f"Period too long (max {cls.MAX_VALUES['period_length']} characters)"
            )
        
        # Basic format validation (should be like "1d", "1mo", "1y", etc.)
        if not re.match(r'^\d+[dwy]|ytd|max$', period_str) and not re.match(r'^\d+mo$', period_str):
            return ValidationResult(
                is_valid=False,
                error_message="Invalid period format. Use format like '1d', '1mo', '1y', 'ytd', or 'max'"
            )
        
        return ValidationResult(
            is_valid=True,
            sanitized_value=period_str
        )
    
    @classmethod
    def validate_ticker_list(cls, tickers: Union[str, List[str]]) -> ValidationResult:
        """
        Validate a list of ticker symbols.
        
        Args:
            tickers: List of ticker symbols or comma-separated string
            
        Returns:
            ValidationResult with validation status
        """
        if not tickers:
            return ValidationResult(
                is_valid=False,
                error_message="Ticker list is required"
            )
        
        # Convert to list if string
        if isinstance(tickers, str):
            ticker_list = [t.strip() for t in tickers.split(",") if t.strip()]
        elif isinstance(tickers, list):
            ticker_list = [str(t).strip() for t in tickers if t]
        else:
            return ValidationResult(
                is_valid=False,
                error_message="Tickers must be a string or list"
            )
        
        # Check list length
        if len(ticker_list) > cls.MAX_VALUES["max_tickers_per_request"]:
            return ValidationResult(
                is_valid=False,
                error_message=f"Too many tickers (max {cls.MAX_VALUES['max_tickers_per_request']})"
            )
        
        # Validate each ticker
        valid_tickers = []
        errors = []
        warnings = []
        
        for ticker in ticker_list:
            result = cls.validate_ticker_symbol(ticker)
            if result.is_valid:
                valid_tickers.append(result.sanitized_value)
            else:
                errors.append(f"{ticker}: {result.error_message}")
            if result.warnings:
                warnings.extend(result.warnings)
        
        if errors:
            return ValidationResult(
                is_valid=False,
                error_message=f"Invalid tickers: {'; '.join(errors)}",
                warnings=warnings
            )
        
        return ValidationResult(
            is_valid=True,
            sanitized_value=valid_tickers,
            warnings=warnings
        )
    
    @classmethod
    def validate_numeric_range(cls, value: Union[int, float, str], 
                             min_val: float = None, max_val: float = None,
                             field_name: str = "value") -> ValidationResult:
        """
        Validate a numeric value within a range.
        
        Args:
            value: Numeric value to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            field_name: Name of the field for error messages
            
        Returns:
            ValidationResult with validation status
        """
        try:
            num_val = float(value)
        except (ValueError, TypeError):
            return ValidationResult(
                is_valid=False,
                error_message=f"{field_name} must be a valid number"
            )
        
        if min_val is not None and num_val < min_val:
            return ValidationResult(
                is_valid=False,
                error_message=f"{field_name} must be at least {min_val}"
            )
        
        if max_val is not None and num_val > max_val:
            return ValidationResult(
                is_valid=False,
                error_message=f"{field_name} must be at most {max_val}"
            )
        
        return ValidationResult(
            is_valid=True,
            sanitized_value=num_val
        )
    
    @classmethod
    def validate_boolean(cls, value: Any, field_name: str = "value") -> ValidationResult:
        """
        Validate and convert a boolean value.
        
        Args:
            value: Value to validate as boolean
            field_name: Name of the field for error messages
            
        Returns:
            ValidationResult with validation status
        """
        if isinstance(value, bool):
            return ValidationResult(
                is_valid=True,
                sanitized_value=value
            )
        
        if isinstance(value, str):
            value_lower = value.lower()
            if value_lower in ('true', '1', 'yes', 'on'):
                return ValidationResult(
                    is_valid=True,
                    sanitized_value=True
                )
            elif value_lower in ('false', '0', 'no', 'off'):
                return ValidationResult(
                    is_valid=True,
                    sanitized_value=False
                )
        
        return ValidationResult(
            is_valid=False,
            error_message=f"{field_name} must be a valid boolean value"
        )
    
    @classmethod
    def validate_string(cls, value: str, max_length: int = None, 
                       field_name: str = "string") -> ValidationResult:
        """
        Validate a string value.
        
        Args:
            value: String to validate
            max_length: Maximum allowed length
            field_name: Name of the field for error messages
            
        Returns:
            ValidationResult with validation status
        """
        if not isinstance(value, str):
            return ValidationResult(
                is_valid=False,
                error_message=f"{field_name} must be a string"
            )
        
        # Check for malicious content
        if cls._contains_malicious_content(value):
            return ValidationResult(
                is_valid=False,
                error_message=f"{field_name} contains invalid characters"
            )
        
        # Length check
        max_len = max_length or cls.MAX_VALUES["max_string_length"]
        if len(value) > max_len:
            return ValidationResult(
                is_valid=False,
                error_message=f"{field_name} too long (max {max_len} characters)"
            )
        
        return ValidationResult(
            is_valid=True,
            sanitized_value=value.strip()
        )
    
    @classmethod
    def validate_api_request(cls, request_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate a complete API request.
        
        Args:
            request_data: Dictionary containing API request data
            
        Returns:
            ValidationResult with validation status
        """
        errors = []
        warnings = []
        sanitized_data = {}
        
        # Validate ticker
        if "ticker" in request_data:
            ticker_result = cls.validate_ticker_symbol(request_data["ticker"])
            if ticker_result.is_valid:
                sanitized_data["ticker"] = ticker_result.sanitized_value
            else:
                errors.append(f"ticker: {ticker_result.error_message}")
            if ticker_result.warnings:
                warnings.extend(ticker_result.warnings)
        
        # Validate timeframe
        if "timeframe" in request_data:
            timeframe_result = cls.validate_timeframe(request_data["timeframe"])
            if timeframe_result.is_valid:
                sanitized_data["timeframe"] = timeframe_result.sanitized_value
            else:
                errors.append(f"timeframe: {timeframe_result.error_message}")
        
        # Validate period
        if "period" in request_data:
            period_result = cls.validate_period(request_data["period"])
            if period_result.is_valid:
                sanitized_data["period"] = period_result.sanitized_value
            else:
                errors.append(f"period: {period_result.error_message}")
        
        # Validate tickers list
        if "tickers" in request_data:
            tickers_result = cls.validate_ticker_list(request_data["tickers"])
            if tickers_result.is_valid:
                sanitized_data["tickers"] = tickers_result.sanitized_value
            else:
                errors.append(f"tickers: {tickers_result.error_message}")
            if tickers_result.warnings:
                warnings.extend(tickers_result.warnings)
        
        # Validate numeric fields
        numeric_fields = {
            "cash": (0, cls.MAX_VALUES["max_cash_amount"]),
            "fee_bps": (0, cls.MAX_VALUES["max_fee_bps"]),
            "slip_bps": (0, cls.MAX_VALUES["max_slip_bps"]),
            "wf_splits": (1, 10)
        }
        
        for field, (min_val, max_val) in numeric_fields.items():
            if field in request_data:
                num_result = cls.validate_numeric_range(
                    request_data[field], min_val, max_val, field
                )
                if num_result.is_valid:
                    sanitized_data[field] = num_result.sanitized_value
                else:
                    errors.append(f"{field}: {num_result.error_message}")
        
        # Validate boolean fields
        boolean_fields = ["low_api_mode", "fast_ta_mode", "use_ml_model", "offline", "walk_forward", "tune_thresholds"]
        for field in boolean_fields:
            if field in request_data:
                bool_result = cls.validate_boolean(request_data[field], field)
                if bool_result.is_valid:
                    sanitized_data[field] = bool_result.sanitized_value
                else:
                    errors.append(f"{field}: {bool_result.error_message}")
        
        if errors:
            return ValidationResult(
                is_valid=False,
                error_message=f"Validation errors: {'; '.join(errors)}",
                warnings=warnings
            )
        
        return ValidationResult(
            is_valid=True,
            sanitized_value=sanitized_data,
            warnings=warnings
        )
    
    @classmethod
    def _contains_malicious_content(cls, text: str) -> bool:
        """
        Check if text contains potentially malicious content.
        
        Args:
            text: Text to check
            
        Returns:
            True if malicious content is found
        """
        if not text:
            return False
        
        text_lower = text.lower()
        
        # Check for malicious patterns
        for pattern in cls.MALICIOUS_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        
        # Check for SQL injection patterns
        sql_patterns = [
            r"(\b(union|select|insert|update|delete|drop|create|alter)\b)",
            r"(--|\b(and|or)\b\s+\d+\s*=\s*\d+)",
            r"(\b(exec|execute|xp_|sp_)\b)",
            r"(\b(union|select|insert|update|delete|drop|create|alter)\b.*\b(union|select|insert|update|delete|drop|create|alter)\b)",
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        
        return False


# Convenience functions for common validations
def validate_ticker(ticker: str) -> ValidationResult:
    """Convenience function to validate a ticker symbol."""
    return InputValidator.validate_ticker_symbol(ticker)


def validate_timeframe(timeframe: str) -> ValidationResult:
    """Convenience function to validate a timeframe."""
    return InputValidator.validate_timeframe(timeframe)


def validate_api_request(request_data: Dict[str, Any]) -> ValidationResult:
    """Convenience function to validate an API request."""
    return InputValidator.validate_api_request(request_data)


def sanitize_input(value: str) -> str:
    """
    Basic input sanitization.
    
    Args:
        value: Input value to sanitize
        
    Returns:
        Sanitized string
    """
    if not value:
        return ""
    
    # Remove null bytes and control characters
    sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', str(value))
    
    # Remove HTML tags
    sanitized = re.sub(r'<[^>]*>', '', sanitized)
    
    # Remove script tags and javascript
    sanitized = re.sub(r'<script[^>]*>.*?</script>', '', sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
    
    return sanitized.strip()
