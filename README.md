# Stock4U - AI-Powered Stock Analysis Platform

**Get AI-powered stock predictions, technical analysis, and market insights in minutes!**

Stock4U is a comprehensive multi-agent stock analysis system that combines technical analysis, sentiment analysis, and AI predictions to help you make informed investment decisions.

## Key Features

- **AI-Powered Predictions** - Multi-agent LangGraph workflow for accurate stock forecasts
- **Technical Analysis** - Charts, indicators, support/resistance levels, and trading signals
- **Real-time Market Data** - Live stock prices, volume, and key metrics
- **Risk Assessment** - Visual breakdown of potential risks and rewards
- **AI Chatbot** - Ask questions about stocks and markets in natural language
- **Beautiful Dashboard** - Modern Streamlit interface with interactive charts
- **Secure & Reliable** - Production-ready with authentication and monitoring

## Quick Start (Choose Your Setup)

### Option 1: One-Click Full Setup (Recommended)
**Everything you need in one click!**

**Windows Users:**
1. Download and extract the Stock4U folder
2. Double-click `scripts\start_stock4u.bat`
3. Wait 5-10 minutes for setup (first time only)
4. Your browser opens automatically to the dashboard

**Linux/macOS Users:**
1. Download and extract the Stock4U folder
2. Open terminal in the Stock4U directory
3. Run: `./scripts/start_stock4u.sh`
4. Your browser opens automatically to the dashboard

### Option 2: Dashboard Only (Simpler)
**Just the interface - no database setup needed**

**Windows Users:**
1. Download and extract the Stock4U folder
2. Double-click `scripts\start_dashboard.bat`
3. Wait for Python setup to complete
4. Dashboard opens in your browser

**Linux/macOS Users:**
1. Download and extract the Stock4U folder
2. Open terminal in the Stock4U directory
3. Run: `./scripts/start_dashboard.sh`
4. Dashboard opens in your browser

## What You Get

### Full Setup Includes:
- **Main Dashboard** (http://localhost:8501) - Your main interface
- **API Backend** (http://localhost:8000) - Powering all features
- **Database** - Secure data storage
- **Cache System** - Fast performance
- **Admin Tools** - Database and cache management

### Dashboard Only Includes:
- **Streamlit Dashboard** (http://localhost:8501) - Core interface
- **Local Storage** - No database required

## Your First Stock Analysis

1. **Open the Dashboard** - Your browser should open automatically
2. **Enter a Stock Symbol** - Try popular ones like AAPL, MSFT, GOOGL, TSLA
3. **Select Timeframe** - Choose from 1d, 5d, 1mo, 3mo, 1y
4. **Click "Run Prediction"** - Get AI-powered analysis in seconds!

## API Keys (Optional)

Stock4U works great out of the box with our development API keys! For enhanced features, you can add your own:

```bash
# Create a .env file in the Stock4U folder with:
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_key_here  
TAVILY_API_KEY=your_tavily_key_here
```

**Enhanced features with API keys:**
- Advanced AI chatbot responses
- News sentiment analysis
- Custom prediction models
- Learning system improvements

## System Requirements

### For Full Setup:
- **Windows 10/11** or **macOS 10.15+** or **Linux (Ubuntu 18.04+)**
- **Docker Desktop** - [Download here](https://www.docker.com/products/docker-desktop/)
- **8GB RAM** (recommended)
- **2GB free disk space**

### For Dashboard Only:
- **Python 3.8+** - [Download here](https://www.python.org/downloads/)
- **4GB RAM** (recommended)
- **1GB free disk space**

## What You Can Do

### Stock Analysis
- **Predictions**: AI-powered price forecasts with confidence levels
- **Technical Charts**: Candlestick charts with indicators
- **Support/Resistance**: Key price levels and trends
- **Risk Assessment**: Visual breakdown of potential outcomes

### AI Features
- **Chatbot**: Ask questions about stocks, markets, or analysis
- **Natural Language**: "What's the outlook for Apple stock?"
- **Sentiment Analysis**: News sentiment integration (with API key)
- **Learning System**: Autonomous model improvement

### Market Data
- **Real-time Prices**: Live stock data and metrics
- **Volume Analysis**: Trading volume and patterns
- **Market Snapshot**: Quick overview of any stock
- **Historical Data**: Price history and performance

## Stopping Stock4U

### Full Setup:
```bash
docker-compose down
```

### Dashboard Only:
Press `Ctrl+C` in the terminal where it's running

## Troubleshooting

### Common Issues:

**"Docker not running"**
- Start Docker Desktop
- Wait for it to fully load
- Try the script again

**"Port already in use"**
- Close other applications using ports 8000, 8501, 8080, 8081
- Or stop Stock4U first: `docker-compose down`

**"Python not found"**
- Install Python 3.8+ from [python.org](https://www.python.org/downloads/)
- Make sure to check "Add Python to PATH" during installation

**"Permission denied" (Linux/macOS)**
- Make scripts executable: `chmod +x scripts/*.sh`
- Or run with sudo: `sudo ./scripts/start_stock4u.sh`

**Dashboard not loading**
- Check if it's running on http://localhost:8501
- Try refreshing the browser
- Check the terminal for error messages

## Advanced Usage

### Command Line Interface
```bash
python main.py
```

### Programmatic Access
```python
from langgraph_flow import run_prediction

result = run_prediction("AAPL", timeframe="1d")
print(result)
```

### MCP Server (AI Integration)
```bash
# Start MCP server
python -m agents.mcp_server

# Test with MCP CLI
mcp dev agents.mcp_server --tool ping
```

## Development

### Running Tests
```bash
python tests/run_all_tests.py
```

### Project Structure
```
Stock4U/
├── agents/          # AI agent modules
├── dashboard/       # User interface
├── api/            # Backend services
├── scripts/        # Setup scripts
├── docs/           # Documentation
└── tests/          # Test suite
```

## Documentation

- **[Quick Start Guide](docs/QUICK_START.md)** - Detailed setup instructions
- **[API Documentation](docs/README.md)** - Technical details
- **[MCP Integration](docs/MCP_INTEGRATION_GUIDE.md)** - AI tool integration
- **[Workflow Guide](docs/PROJECT_WORKFLOW.md)** - System architecture

## Important Notes

- **Educational Purpose**: This tool is for educational and informational purposes only
- **Not Financial Advice**: Always do your own research and consult financial advisors
- **Data Sources**: Uses yfinance for market data, which may have delays
- **API Limits**: Some features may be rate-limited based on API provider policies

## Support & Community

- **Issues**: Report bugs on our repository
- **Questions**: Check our documentation or open an issue
- **Contributions**: We welcome pull requests and improvements
- **Updates**: Follow our repository for the latest features

## License

This project is open source. See the repository for license details.

---

**Ready to start analyzing stocks with AI? Choose your setup option above and get started in minutes!**

*Stock4U - Making stock analysis accessible to everyone*


