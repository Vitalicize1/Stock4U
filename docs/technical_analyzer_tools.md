# Technical Analyzer Tools Documentation

## Overview

The Technical Analyzer Tools provide comprehensive technical analysis capabilities for the stock prediction system. These tools are designed to work with the Technical Analyzer Agent to perform advanced market analysis using various technical indicators, pattern recognition, and signal generation.

## Tools Overview

### 1. `calculate_advanced_indicators`

**Purpose**: Calculate comprehensive technical indicators for a stock.

**Parameters**:
- `ticker` (str): Stock ticker symbol (e.g., 'AAPL', 'MSFT')
- `period` (str): Data period for analysis (default: "6mo")

**Returns**: Dictionary containing:
- Current indicator values (RSI, MACD, Moving Averages, etc.)
- Data points analyzed
- Last updated timestamp

**Example Usage**:
```python
from agents.tools.technical_analyzer_tools import calculate_advanced_indicators

result = calculate_advanced_indicators("AAPL", "6mo")
if result["status"] == "success":
    indicators = result["current_indicators"]
    print(f"RSI: {indicators['rsi']:.2f}")
    print(f"MACD: {indicators['macd']:.4f}")
```

**Indicators Calculated**:
- Moving Averages (SMA 20, 50, 200; EMA 12, 26)
- MACD (Moving Average Convergence Divergence)
- RSI (Relative Strength Index)
- Bollinger Bands
- Stochastic Oscillator
- Volume indicators (OBV, AD)
- Williams %R
- CCI (Commodity Channel Index)
- ATR (Average True Range)
- ADX (Average Directional Index)
- Parabolic SAR

### 2. `identify_chart_patterns`

**Purpose**: Identify candlestick patterns and chart formations.

**Parameters**:
- `ticker` (str): Stock ticker symbol
- `period` (str): Data period for analysis (default: "6mo")

**Returns**: Dictionary containing:
- Candlestick patterns detected
- Trend patterns (Golden Cross, Death Cross)
- Pattern significance and signals

**Candlestick Patterns Detected**:
- Doji
- Engulfing patterns
- Hammer/Hanging Man
- Morning Star/Evening Star
- Piercing/Dark Cloud Cover
- Spinning Top
- Shooting Star

**Example Usage**:
```python
from agents.tools.technical_analyzer_tools import identify_chart_patterns

result = identify_chart_patterns("AAPL", "6mo")
if result["status"] == "success":
    patterns = result["candlestick_patterns"]
    for pattern in patterns:
        print(f"{pattern['pattern']}: {pattern['signal']} ({pattern['significance']})")
```

### 3. `analyze_support_resistance`

**Purpose**: Analyze support and resistance levels using multiple methods.

**Parameters**:
- `ticker` (str): Stock ticker symbol
- `period` (str): Data period for analysis (default: "6mo")

**Returns**: Dictionary containing:
- Pivot points
- Fibonacci retracement levels
- Moving averages as support/resistance
- Nearest support and resistance levels

**Methods Used**:
1. **Pivot Points**: Traditional pivot point calculation
2. **Recent Highs/Lows**: Rolling window analysis
3. **Fibonacci Retracements**: Based on swing high/low
4. **Moving Averages**: Key MAs as dynamic levels

**Example Usage**:
```python
from agents.tools.technical_analyzer_tools import analyze_support_resistance

result = analyze_support_resistance("AAPL", "6mo")
if result["status"] == "success":
    current_price = result["current_price"]
    nearest_support = result["nearest_support"]
    nearest_resistance = result["nearest_resistance"]
    
    print(f"Current: ${current_price:.2f}")
    print(f"Nearest Support: ${nearest_support:.2f}")
    print(f"Nearest Resistance: ${nearest_resistance:.2f}")
```

### 4. `perform_trend_analysis`

**Purpose**: Perform comprehensive trend analysis using multiple timeframes.

**Parameters**:
- `ticker` (str): Stock ticker symbol
- `period` (str): Data period for analysis (default: "6mo")

**Returns**: Dictionary containing:
- Trend analysis for short, medium, and long-term
- Trend strength calculation
- ADX strength indicator
- Price position relative to moving averages

**Timeframes Analyzed**:
- **Short-term**: 5-10 days
- **Medium-term**: 20-50 days
- **Long-term**: 200 days

**Example Usage**:
```python
from agents.tools.technical_analyzer_tools import perform_trend_analysis

result = perform_trend_analysis("AAPL", "6mo")
if result["status"] == "success":
    trends = result["trends"]
    trend_strength = result["trend_strength"]
    
    print(f"Short-term: {trends['short_term']}")
    print(f"Medium-term: {trends['medium_term']}")
    print(f"Long-term: {trends['long_term']}")
    print(f"Trend Strength: {trend_strength:.1f}%")
```

### 5. `generate_trading_signals`

**Purpose**: Generate comprehensive trading signals based on technical analysis.

**Parameters**:
- `ticker` (str): Stock ticker symbol
- `period` (str): Data period for analysis (default: "6mo")

**Returns**: Dictionary containing:
- Individual trading signals
- Overall recommendation
- Signal strength
- Buy/sell signal counts

**Signal Types**:
- **RSI Signals**: Oversold/overbought conditions
- **MACD Signals**: Crossover signals
- **Moving Average Signals**: Price vs MA relationships
- **Bollinger Bands Signals**: Price band violations
- **Stochastic Signals**: Oscillator extremes

**Overall Recommendations**:
- STRONG_BUY (signal strength ≥ 3)
- BUY (signal strength ≥ 1)
- HOLD (signal strength between -1 and 1)
- SELL (signal strength ≤ -1)
- STRONG_SELL (signal strength ≤ -3)

**Example Usage**:
```python
from agents.tools.technical_analyzer_tools import generate_trading_signals

result = generate_trading_signals("AAPL", "6mo")
if result["status"] == "success":
    recommendation = result["overall_recommendation"]
    signal_strength = result["signal_strength"]
    
    print(f"Recommendation: {recommendation}")
    print(f"Signal Strength: {signal_strength}")
    
    for signal in result["signals"]:
        print(f"- {signal['type']} ({signal['indicator']}): {signal['reason']}")
```

### 6. `validate_technical_analysis`

**Purpose**: Validate technical analysis results for consistency and reliability.

**Parameters**:
- `ticker` (str): Stock ticker symbol
- `analysis_data` (dict): Technical analysis data to validate

**Returns**: Dictionary containing:
- Data quality assessment
- Indicator consistency check
- Signal reliability validation
- Validation score (0-100)
- Warnings and recommendations

**Validation Checks**:
- Data completeness and quality
- Indicator value ranges
- Signal consistency
- Analysis reliability

**Example Usage**:
```python
from agents.tools.technical_analyzer_tools import validate_technical_analysis

# Sample analysis data
analysis_data = {
    "current_indicators": {"rsi": 65.5, "macd": 0.123},
    "signals": [{"type": "BUY", "indicator": "RSI"}]
}

result = validate_technical_analysis("AAPL", analysis_data)
if result["status"] == "success":
    validation = result["validation_results"]
    print(f"Validation Score: {validation['validation_score']}/100")
    print(f"Data Quality: {validation['data_quality']}")
    
    for warning in validation["warnings"]:
        print(f"Warning: {warning}")
```

## Integration with Technical Analyzer Agent

The tools are integrated into the `TechnicalAnalyzerAgent` class through the `analyze_technical_data_with_tools` method:

```python
from agents.technical_analyzer import TechnicalAnalyzerAgent

agent = TechnicalAnalyzerAgent()
result = agent.analyze_technical_data_with_tools({
    "ticker": "AAPL",
    "period": "6mo"
})
```

## Enhanced Technical Score Calculation

The enhanced technical score uses a sophisticated algorithm that considers:

1. **Indicator Contributions**:
   - RSI oversold/overbought conditions
   - MACD crossover signals
   - Moving average alignments

2. **Pattern Contributions**:
   - Bullish/bearish pattern detection
   - Pattern significance weighting

3. **Signal Strength**:
   - Multiple signal confirmation
   - Signal strength amplification

4. **Validation Quality**:
   - Data quality adjustment
   - Analysis reliability factor

## Error Handling

All tools include comprehensive error handling:

- **Data Availability**: Checks for sufficient data points
- **API Failures**: Handles yfinance API errors gracefully
- **Calculation Errors**: Validates indicator calculations
- **Fallback Mechanisms**: Provides alternative analysis methods

## Performance Considerations

- **Caching**: Tools can be extended with caching mechanisms
- **Data Efficiency**: Optimized for minimal API calls
- **Memory Usage**: Efficient data structure handling
- **Processing Speed**: Fast calculation algorithms

## Dependencies

The tools require the following dependencies:
- `pandas`: Data manipulation
- `numpy`: Numerical calculations
- `yfinance`: Stock data retrieval
- `TA-Lib`: Technical analysis library
- `langchain_core.tools`: Tool decorators

## Installation

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

Note: TA-Lib may require additional system dependencies on some platforms.

## Testing

Run the test script to verify all tools:

```bash
python test_technical_tools.py
```

This will test all tools with a sample ticker (AAPL) and display the results.

## Future Enhancements

Potential improvements for the tools:

1. **Additional Indicators**: More advanced technical indicators
2. **Machine Learning**: ML-based pattern recognition
3. **Real-time Data**: Live market data integration
4. **Custom Indicators**: User-defined indicator calculations
5. **Backtesting**: Historical performance validation
6. **Visualization**: Chart generation capabilities
7. **Alert System**: Real-time signal notifications
8. **Portfolio Analysis**: Multi-stock analysis tools 