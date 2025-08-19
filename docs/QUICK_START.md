# Stock4U Quick Start Guide

Welcome to Stock4U! This guide will help you get up and running quickly with our multi-agent stock analysis system.

## 🚀 Quick Start Options

### Option 1: One-Click Setup (Recommended)

**Windows Users:**
1. Double-click `start_stock4u.bat`
2. Wait for Docker to start all services
3. Your browser will automatically open to the dashboard

**Linux/macOS Users:**
1. Open terminal in the project directory
2. Run: `./start_stock4u.sh`
3. Your browser will automatically open to the dashboard

### Option 2: Dashboard Only (Simpler)

If you just want to try the dashboard without the full backend:

**Windows Users:**
1. Double-click `start_dashboard.bat`
2. Wait for Python setup to complete
3. Dashboard will open in your browser

**Linux/macOS Users:**
1. Open terminal in the project directory
2. Run: `./start_dashboard.sh`
3. Dashboard will open in your browser

## 📋 Prerequisites

### For Full Setup (Option 1):
- **Docker Desktop** - [Download here](https://www.docker.com/products/docker-desktop/)
- **Docker Compose** - Usually included with Docker Desktop

### For Dashboard Only (Option 2):
- **Python 3.8+** - [Download here](https://www.python.org/downloads/)

## 🔧 What Gets Set Up

### Full Setup Includes:
- **Main API** (http://localhost:8000) - Backend services
- **Dashboard** (http://localhost:8501) - User interface
- **PostgreSQL Database** - Data storage
- **Redis Cache** - Performance optimization
- **PgAdmin** (http://localhost:8080) - Database management
- **Redis Commander** (http://localhost:8081) - Cache management

### Dashboard Only Includes:
- **Streamlit Dashboard** (http://localhost:8501) - Basic interface
- **Local file storage** - No database required

## 🎯 First Steps

1. **Choose a Setup Option** above
2. **Wait for services to start** (first run may take 5-10 minutes)
3. **Open the dashboard** in your browser
4. **Try a stock prediction**:
   - Enter a ticker symbol (e.g., AAPL, MSFT, GOOGL)
   - Select a timeframe (1d, 5d, 1mo)
   - Click "Run Prediction"

## 🔑 Optional API Keys

For enhanced features, you can add API keys to your `.env` file:

```bash
# OpenAI for advanced AI features
OPENAI_API_KEY=your_openai_key_here

# Google AI for alternative AI features  
GOOGLE_API_KEY=your_google_key_here

# Tavily for news sentiment analysis
TAVILY_API_KEY=your_tavily_key_here
```

## 🛠️ Troubleshooting

### Docker Issues:
- **"Docker not running"**: Start Docker Desktop
- **"Port already in use"**: Stop other services using ports 8000, 8501, 8080, 8081
- **"Permission denied"**: Run as administrator (Windows) or use `sudo` (Linux/macOS)

### Python Issues:
- **"Python not found"**: Install Python 3.8+ and add to PATH
- **"pip not found"**: Install pip or use `python -m pip`
- **"Virtual environment failed"**: Check Python installation and permissions

### Service Issues:
- **Dashboard not loading**: Check if Streamlit is running on port 8501
- **API errors**: Check if backend services are running
- **Database errors**: Check if PostgreSQL container is healthy

## 📊 What You Can Do

### Basic Features:
- **Stock Predictions**: Get AI-powered stock price predictions
- **Technical Analysis**: View charts, indicators, and signals
- **Risk Assessment**: Understand potential risks and rewards
- **Market Data**: Real-time stock data and metrics

### Advanced Features (with API keys):
- **AI Chatbot**: Ask questions about stocks and markets
- **Sentiment Analysis**: News sentiment integration
- **Learning System**: Autonomous model improvement
- **Custom Analysis**: Tailored predictions and insights

## 🛑 Stopping Stock4U

### Full Setup:
```bash
docker-compose -f ops/docker-compose.yml down
```

### Dashboard Only:
Press `Ctrl+C` in the terminal where it's running

## 📚 Next Steps

- Read the [main README](README.md) for detailed documentation
- Check out the [docs folder](docs/) for advanced guides
- Explore the [examples folder](examples/) for usage examples
- Join our community for support and updates

## 🆘 Need Help?

- Check the [troubleshooting section](#️-troubleshooting) above
- Review the [main documentation](README.md)
- Open an issue on our repository
- Contact our support team

---

**Enjoy using Stock4U! 🚀📈**
