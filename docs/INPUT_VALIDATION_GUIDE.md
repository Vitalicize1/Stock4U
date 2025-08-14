# Input Validation Guide

This guide covers the comprehensive input validation system implemented in Stock4U to ensure security, data integrity, and proper error handling.

## Overview

The Stock4U application implements a multi-layered validation system that includes:

1. **Centralized Validation Module** (`utils/validation.py`)
2. **API Request Validation** (Pydantic models with custom validators)
3. **Middleware Validation** (`utils/validation_middleware.py`)
4. **UI Validation** (Streamlit dashboard validation)
5. **Agent Tool Validation** (LangGraph agent validation)

## Security Features

### 🛡️ Malicious Content Detection

The validation system detects and blocks:

- **XSS (Cross-Site Scripting)** attempts
- **SQL Injection** patterns
- **JavaScript injection** attempts
- **HTML/XML injection** attempts
- **Command injection** patterns

### 🔍 Input Sanitization

All inputs are automatically sanitized to remove:
- Null bytes and control characters
- HTML tags
- Script tags
- JavaScript protocols

## Validation Components

### 1. Core Validation Module

Located in `utils/validation.py`, this module provides:

```python
from utils.validation import InputValidator, ValidationResult

# Validate ticker symbols
result = InputValidator.validate_ticker_symbol("AAPL")
if result.is_valid:
    ticker = result.sanitized_value
else:
    error = result.error_message

# Validate timeframes
result = InputValidator.validate_timeframe("1d")

# Validate complete API requests
result = InputValidator.validate_api_request(request_data)
```

### 2. ValidationResult Class

All validation methods return a `ValidationResult` object:

```python
@dataclass
class ValidationResult:
    is_valid: bool
    error_message: Optional[str] = None
    sanitized_value: Optional[Any] = None
    warnings: List[str] = None
```

### 3. Supported Validations

#### Ticker Symbol Validation
- **Format**: 1-10 uppercase letters only
- **Examples**: `AAPL`, `MSFT`, `GOOGL`
- **Rejected**: `AAPL123`, `A@PL`, `<script>alert('xss')</script>`

#### Timeframe Validation
- **Valid values**: `1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `10y`, `ytd`, `max`
- **Rejected**: `invalid`, `1day`, `1M`

#### Period Validation
- **Format**: `\d+[dwy]` or `\d+mo` or `ytd` or `max`
- **Examples**: `1d`, `1mo`, `1y`, `ytd`, `max`
- **Rejected**: `1day`, `1M`, `invalid`

#### Numeric Range Validation
- **Configurable**: min/max values
- **Examples**: cash (0-1B), fee_bps (0-1000), slip_bps (0-1000)

#### Boolean Validation
- **Accepted**: `true`, `false`, `1`, `0`, `yes`, `no`, `on`, `off`
- **Rejected**: `maybe`, `sometimes`, `invalid`

#### String Validation
- **Length limits**: Configurable max length
- **Sanitization**: Automatic HTML/script removal
- **Security**: Malicious content detection

## API Integration

### Pydantic Model Validation

API endpoints use Pydantic models with custom validators:

```python
class PredictRequest(BaseModel):
    ticker: str = Field(..., description="Ticker symbol")
    timeframe: str = Field("1d", description="Timeframe: 1d|1w|1m")
    
    @validator("ticker")
    def _validate_ticker(cls, v: str) -> str:
        result = InputValidator.validate_ticker_symbol(v)
        if not result.is_valid:
            raise ValueError(result.error_message)
        return result.sanitized_value
```

### Middleware Validation

Additional validation at the request level:

```python
# Automatically enabled via environment variable
ENABLE_VALIDATION_MIDDLEWARE=true
```

The middleware validates:
- Query parameters
- Path parameters
- Request body content

## Dashboard Integration

### Streamlit Validation

The dashboard implements client-side validation:

```python
# Validate inputs before processing
if submitted:
    ticker_validation = InputValidator.validate_ticker_symbol(ticker_input)
    if not ticker_validation.is_valid:
        st.error(f"❌ Invalid ticker: {ticker_validation.error_message}")
        submitted = False
```

### Real-time Feedback

Users receive immediate feedback for:
- Invalid ticker symbols
- Invalid timeframes
- Malicious input attempts
- Validation warnings

## Agent Tool Integration

### LangGraph Agent Validation

Agent tools use the validation system:

```python
@tool
def validate_ticker_symbol(ticker: str) -> Dict[str, Any]:
    # Use the centralized validation system
    from utils.validation import InputValidator
    
    validation_result = InputValidator.validate_ticker_symbol(ticker)
    if not validation_result.is_valid:
        return {
            "valid": False,
            "error": validation_result.error_message,
            "ticker": ticker
        }
```

## Configuration

### Environment Variables

```bash
# Enable/disable validation middleware
ENABLE_VALIDATION_MIDDLEWARE=true

# Enable/disable yfinance ticker validation (for performance)
VALIDATE_TICKER_WITH_YFINANCE=true

# Maximum values for validation
MAX_TICKERS_PER_REQUEST=50
MAX_CASH_AMOUNT=1000000000
MAX_FEE_BPS=1000
MAX_SLIP_BPS=1000
```

### Custom Validation Rules

You can extend the validation system by:

1. **Adding new validation methods** to `InputValidator`
2. **Modifying validation patterns** in the class constants
3. **Creating custom validators** for specific use cases

## Error Handling

### Validation Errors

All validation errors include:
- **Clear error messages** explaining the issue
- **Suggestions** for correct input format
- **Field-specific** error details

### Error Response Format

```json
{
  "error": "Validation errors: ticker: Ticker contains invalid characters; timeframe: Invalid timeframe"
}
```

### Warning System

The validation system provides warnings for:
- Performance issues
- Deprecated formats
- Non-critical validation failures

## Testing

### Validation Test Suite

Run the comprehensive test suite:

```bash
python tests/test_input_validation.py
```

The test suite covers:
- ✅ Valid input scenarios
- ❌ Invalid input scenarios
- 🛡️ Security attack attempts
- 🔍 Edge cases and boundary conditions

### Test Coverage

The validation system is tested for:
- **Ticker validation** (valid/invalid formats, malicious content)
- **Timeframe validation** (valid values, invalid formats)
- **Period validation** (format checking, edge cases)
- **Numeric validation** (range checking, type validation)
- **Boolean validation** (various true/false representations)
- **String validation** (length limits, sanitization)
- **API request validation** (complete request validation)
- **Malicious content detection** (XSS, SQL injection, etc.)

## Security Best Practices

### 1. Input Validation Layers

- **Client-side**: Immediate user feedback
- **API-level**: Request validation
- **Middleware**: Additional security checks
- **Business logic**: Domain-specific validation

### 2. Sanitization Strategy

- **Automatic sanitization** of all string inputs
- **Type conversion** with validation
- **Length limits** to prevent buffer overflow attacks
- **Pattern matching** to detect malicious content

### 3. Error Information

- **Generic error messages** to avoid information disclosure
- **Detailed logging** for debugging
- **User-friendly messages** for common errors

## Performance Considerations

### Validation Performance

- **Caching**: Validation results are cached where appropriate
- **Lazy validation**: yfinance validation can be disabled for performance
- **Efficient patterns**: Compiled regex patterns for pattern matching
- **Minimal overhead**: Validation adds minimal processing time

### Optimization Tips

1. **Disable yfinance validation** for high-throughput scenarios
2. **Use validation middleware** only when needed
3. **Cache validation results** for repeated inputs
4. **Batch validation** for multiple inputs

## Troubleshooting

### Common Issues

1. **Validation too strict**: Adjust validation patterns or limits
2. **Performance issues**: Disable optional validations
3. **False positives**: Review malicious content patterns
4. **Missing validations**: Add custom validators as needed

### Debug Mode

Enable debug logging for validation issues:

```python
import logging
logging.getLogger('utils.validation').setLevel(logging.DEBUG)
```

## Future Enhancements

### Planned Features

1. **Rate limiting**: Advanced rate limiting middleware
2. **Input profiling**: Machine learning-based input analysis
3. **Custom validators**: Domain-specific validation rules
4. **Validation metrics**: Performance and security metrics

### Extension Points

The validation system is designed for easy extension:
- **New validation types**: Add to `InputValidator` class
- **Custom patterns**: Modify pattern constants
- **Middleware extensions**: Add new middleware components
- **Integration hooks**: Connect to external validation services

## Conclusion

The Stock4U input validation system provides comprehensive security and data integrity protection through multiple layers of validation. The system is designed to be:

- **Secure**: Protects against common attack vectors
- **Performant**: Minimal overhead with caching and optimization
- **Extensible**: Easy to add new validation rules
- **User-friendly**: Clear error messages and feedback
- **Maintainable**: Well-documented and tested

For questions or issues with the validation system, refer to the test suite or contact the development team.
