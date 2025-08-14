@echo off
echo ========================================
echo        Stock4U Dashboard Setup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed!
    echo Please install Python 3.8+ from: https://www.python.org/downloads/
    echo Then restart this script.
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment!
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\Activate.ps1
if %errorlevel% neq 0 (
    echo Trying alternative activation...
    call venv\Scripts\activate.bat
)

REM Install requirements if needed
if not exist venv\Lib\site-packages\streamlit (
    echo Installing dependencies...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install dependencies!
        pause
        exit /b 1
    )
)

REM Create .env file if it doesn't exist
if not exist .env (
    echo Creating basic .env file...
    (
        echo # Stock4U Environment Configuration
        echo # Basic setup for dashboard only
        echo.
        echo # Optional API Keys (uncomment and add your keys for full features)
        echo # OPENAI_API_KEY=your_openai_key_here
        echo # GOOGLE_API_KEY=your_google_key_here
        echo # TAVILY_API_KEY=your_tavily_key_here
    ) > .env
    echo Created .env file with basic settings.
)

echo.
echo Starting Stock4U Dashboard...
echo Dashboard will open in your browser at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the dashboard
echo.

REM Start the dashboard
streamlit run dashboard.py --server.port 8501 --server.headless false
