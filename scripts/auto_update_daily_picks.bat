@echo off
REM Automatic Daily Picks Update Script
REM This script generates and publishes fresh daily picks to GitHub Gist

echo ========================================
echo     Stock4U Daily Picks Auto Update
echo ========================================
echo %date% %time%
echo.

REM Change to Stock4U directory
cd /d "D:\Stock4U"

REM Set environment variables
set GITHUB_TOKEN=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
set PYTHONPATH=D:\Stock4U

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Generate and publish daily picks
echo 🔄 Generating and publishing daily picks...
python utils/publish_daily_picks.py

REM Log the result
echo.
echo ✅ Daily picks update completed at %date% %time%
echo ========================================

REM Optional: Add to log file
echo %date% %time% - Daily picks updated >> logs\daily_picks_auto.log
