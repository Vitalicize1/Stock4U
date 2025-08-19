# Stock4U Installation Guide

This guide will walk you through downloading and setting up Stock4U on your computer.

## Download Options

### Option 1: Download from GitHub (Recommended)

1. **Visit the Repository**
   - Go to: https://github.com/Vitalicize1/Stock4U
   - Click the green "Code" button
   - Select "Download ZIP"

2. **Extract the Files**
   - Find the downloaded `Stock4U-main.zip` file
   - Right-click and select "Extract All" (Windows) or double-click (macOS/Linux)
   - Extract to a location like `C:\Stock4U` (Windows) or `~/Stock4U` (macOS/Linux)

3. **Open the Folder**
   - Navigate to the extracted `Stock4U` folder
   - You should see files like `README.md`, `requirements.txt`, and a `scripts` folder

### Option 2: Clone with Git (Advanced Users)

```bash
git clone https://github.com/Vitalicize1/Stock4U.git
cd Stock4U
```

## Quick Setup

### Windows Users

1. **Navigate to the Stock4U folder**
   - Open File Explorer
   - Go to where you extracted the Stock4U folder

2. **Choose your setup option:**

   **For Full Features (Recommended):**
   - Double-click `scripts\start_stock4u.bat`
   - Wait 5-10 minutes for first-time setup
   - Your browser will open automatically

   **For Dashboard Only (Simpler):**
   - Double-click `scripts\start_dashboard.bat`
   - Wait for Python setup to complete
   - Dashboard will open in your browser

### macOS/Linux Users

1. **Open Terminal**
   - Press `Cmd+Space` (macOS) or `Ctrl+Alt+T` (Linux)
   - Type "Terminal" and press Enter

2. **Navigate to Stock4U folder**
   ```bash
   cd /path/to/Stock4U
   ```

3. **Make scripts executable**
   ```bash
   chmod +x scripts/*.sh
   ```

4. **Choose your setup option:**

   **For Full Features (Recommended):**
   ```bash
   ./scripts/start_stock4u.sh
   ```

   **For Dashboard Only (Simpler):**
   ```bash
   ./scripts/start_dashboard.sh
   ```

## Prerequisites

### For Full Setup (Option 1):
- **Docker Desktop** - [Download here](https://www.docker.com/products/docker-desktop/)
- **8GB RAM** (recommended)
- **2GB free disk space**

### For Dashboard Only (Option 2):
- **Python 3.8+** - [Download here](https://www.python.org/downloads/)
- **4GB RAM** (recommended)
- **1GB free disk space**

## Verification

After setup, you should see:

### Full Setup:
- **Dashboard**: http://localhost:8501 (opens automatically)
- **API**: http://localhost:8000 (backend services)
- **Database Admin**: http://localhost:8080 (optional)
- **Cache Admin**: http://localhost:8081 (optional)

### Dashboard Only:
- **Dashboard**: http://localhost:8501 (opens automatically)

## First Steps

1. **Open the Dashboard** - Your browser should open automatically
2. **Try a Stock Analysis**:
   - Enter a stock symbol (e.g., AAPL, MSFT, GOOGL)
   - Select a timeframe (1d, 5d, 1mo)
   - Click "Run Prediction"
3. **Explore Features**:
   - Check out the Chatbot tab
   - View Market Data
   - Try different stocks and timeframes

## Troubleshooting

### Common Issues:

**"Docker not running"**
- Install Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop/)
- Start Docker Desktop and wait for it to fully load
- Try the script again

**"Python not found"**
- Install Python 3.8+ from [python.org](https://www.python.org/downloads/)
- Make sure to check "Add Python to PATH" during installation
- Restart your terminal/command prompt

**"Permission denied" (macOS/Linux)**
- Make scripts executable: `chmod +x scripts/*.sh`
- Or run with sudo: `sudo ./scripts/start_stock4u.sh`

**"Port already in use"**
- Close other applications using ports 8000, 8501, 8080, 8081
- Or stop Stock4U first: `docker-compose -f ops/docker-compose.yml down`

**Dashboard not loading**
- Check if it's running on http://localhost:8501
- Try refreshing the browser
- Check the terminal for error messages

## Optional: Add API Keys

For enhanced features, create a `.env` file in the Stock4U folder:

```bash
# Create .env file with your API keys
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_key_here
TAVILY_API_KEY=your_tavily_key_here
```

**Enhanced features with API keys:**
- Advanced AI chatbot responses
- News sentiment analysis
- Custom prediction models
- Learning system improvements

## Stopping Stock4U

### Full Setup:
```bash
docker-compose -f ops/docker-compose.yml down
```

### Dashboard Only:
Press `Ctrl+C` in the terminal where it's running

## Need Help?

- Check the [main README](README.md) for detailed documentation
- Review the [troubleshooting section](#️-troubleshooting) above
- Open an issue on our GitHub repository
- Check our [Quick Start Guide](docs/QUICK_START.md)

---

**Congratulations! You're ready to start analyzing stocks with AI!**

*Stock4U - Making stock analysis accessible to everyone*
