@echo off
echo ========================================
echo    Stock4U Daily Picks Service Setup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed!
    echo Please install Python 3.8+ from: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install requirements if needed
if not exist venv\Lib\site-packages\fastapi (
    echo Installing dependencies...
    pip install -r requirements.txt
)

REM Create cache directory
if not exist cache mkdir cache

REM Set environment variables for the service
set DAILY_PICKS_PATH=cache\daily_picks.json
set DAILY_PICKS_AUTO_GENERATE=true
set DAILY_PICKS_MAX_AGE_HOURS=24
set DAILY_PICKS_PORT=8001
set DAILY_PICKS_HOST=0.0.0.0
set DAILY_PICKS_CRON=0 14 * * 1-5
set DAILY_PICKS_TIMEFRAME=1d
set DAILY_PICKS_TOP_N=3
set DAILY_PICKS_LOW_API=1
set DAILY_PICKS_FAST_TA=1

echo.
echo Starting Daily Picks Service...
echo Service will be available at: http://localhost:8001
echo Daily picks endpoint: http://localhost:8001/daily_picks
echo.
echo Press Ctrl+C to stop the service
echo.

REM Start the service
python deployment\daily_picks_service.py
