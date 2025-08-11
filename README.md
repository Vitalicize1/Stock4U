# 🤖 Agentic Stock Prediction System

A sophisticated multi-agent system built with LangGraph that predicts stock movements using technical analysis, market data, and AI-powered reasoning.

## 🚀 Features

- **Multi-Agent Architecture**: Coordinated agents for data collection, technical analysis, prediction, and evaluation
- **Comprehensive Analysis**: Technical indicators, trend analysis, support/resistance levels, and risk assessment
- **AI-Powered Predictions**: LLM integration for intelligent stock movement predictions
- **Real-time Data**: Live market data using yfinance
- **Interactive Dashboard**: Beautiful Streamlit interface for visualization
- **Risk Management**: Built-in risk assessment and warnings
- **Evaluation System**: Quality assessment and optimization suggestions

## 🏗️ Architecture

The system uses a LangGraph workflow with specialized agents:

```
Orchestrator → Data Collector → Technical Analyzer → Prediction Agent → Evaluator → Elicitation
```

### Agent Roles

1. **Orchestrator Agent**: Initializes and coordinates the entire workflow
2. **Data Collector Agent**: Fetches stock data, company info, and market context
3. **Technical Analyzer Agent**: Performs technical analysis and generates signals
4. **Prediction Agent**: Uses LLMs to make final predictions
5. **Evaluator Optimizer Agent**: Assesses prediction quality and provides feedback
6. **Elicitation Agent**: Provides final summary and recommendations

## 📦 Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd agentic_stock_predictorv2
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up environment variables** (optional):
```bash
# Create a .env file for API keys
echo "OPENAI_API_KEY=your_openai_key" > .env
echo "GOOGLE_API_KEY=your_google_key" >> .env
```

## 🚀 Usage

### Command Line Interface

Run predictions for multiple stocks:
```bash
python main.py
```

### Interactive Dashboard

Launch the Streamlit dashboard:
```bash
streamlit run dashboard.py
```

### Programmatic Usage

```python
from langgraph_flow import run_prediction

# Run prediction for a single stock
result = run_prediction("AAPL", timeframe="1d")
print(result)
```

## 📊 Output Format

The system provides comprehensive analysis results:

```json
{
  "ticker": "AAPL",
  "timeframe": "1d",
  "final_summary": {
    "prediction_summary": {
      "direction": "UP",
      "confidence": 75.5,
      "reasoning": "Strong technical indicators...",
      "key_factors": ["Bullish trend", "High volume", "RSI oversold"]
    },
    "technical_summary": {
      "technical_score": 85.0,
      "technical_signals": ["STRONG_BUY"],
      "trend_analysis": {
        "short_term_trend": "bullish",
        "medium_term_trend": "bullish"
      }
    },
    "final_recommendation": {
      "action": "BUY",
      "position_size": "normal",
      "confidence": 75.5
    },
    "risk_warnings": [
      "This prediction is for informational purposes only...",
      "High volatility expected - consider using stop-loss orders"
    ]
  }
}
```

## 🔧 Configuration

### Supported Timeframes
- `1d`: One day prediction
- `1w`: One week prediction  
- `1m`: One month prediction

### Technical Indicators
- Moving Averages (20-day, 50-day)
- RSI (Relative Strength Index)
- Support/Resistance Levels
- Volume Analysis
- Trend Analysis

### Risk Assessment
- Market Risk
- Volatility Risk
- Liquidity Risk
- Sector Risk

## 🎯 Prediction Methodology

1. **Data Collection**: Fetches historical price data, company information, and market context
2. **Technical Analysis**: Calculates technical indicators and identifies patterns
3. **AI Prediction**: Uses LLMs to analyze all data and generate predictions
4. **Risk Assessment**: Evaluates various risk factors
5. **Quality Evaluation**: Assesses prediction quality and consistency
6. **Final Recommendation**: Provides actionable trading recommendations

## 📈 Dashboard Features

The Streamlit dashboard provides:

- **Real-time Analysis**: Live stock data and predictions
- **Interactive Charts**: Candlestick charts with technical indicators
- **Risk Assessment**: Visual risk breakdown and warnings
- **Technical Analysis**: Detailed technical indicator analysis
- **Prediction Details**: Comprehensive prediction reasoning and factors

## 🔮 Future Enhancements

- [ ] Sentiment analysis integration
- [ ] News and earnings data analysis
- [ ] Machine learning model integration
- [ ] Portfolio optimization
- [ ] Backtesting capabilities
- [ ] Real-time alerts and notifications
- [ ] Multi-timeframe analysis
- [ ] Sector and market correlation analysis

## ⚠️ Disclaimer

This system is for educational and informational purposes only. It should not be considered as financial advice. Always conduct your own research and consult with financial professionals before making investment decisions.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

If you encounter any issues or have questions:

1. Check the existing issues
2. Create a new issue with detailed information
3. Include error messages and system information

## 🏆 Acknowledgments

- LangGraph for the workflow framework
- yfinance for market data
- Streamlit for the dashboard interface
- OpenAI and Google for LLM capabilities
