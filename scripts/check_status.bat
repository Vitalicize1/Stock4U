@echo off
echo ========================================
echo        Stock4U Service Status
echo ========================================
echo.

echo Checking Docker containers...
docker-compose -f ops\docker-compose.yml ps

echo.
echo Checking service health...
echo.

REM Check API health
echo Testing API endpoint...
curl -s http://localhost:8000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo ✓ API is running at http://localhost:8000
) else (
    echo ✗ API is not responding
)

REM Check dashboard
echo Testing dashboard...
curl -s http://localhost:8501 >nul 2>&1
if %errorlevel% equ 0 (
    echo ✓ Dashboard is running at http://localhost:8501
) else (
    echo ✗ Dashboard is not responding
)

REM Check PgAdmin
echo Testing PgAdmin...
curl -s http://localhost:8080 >nul 2>&1
if %errorlevel% equ 0 (
    echo ✓ PgAdmin is running at http://localhost:8080
) else (
    echo ✗ PgAdmin is not responding
)

REM Check Redis Commander
echo Testing Redis Commander...
curl -s http://localhost:8081 >nul 2>&1
if %errorlevel% equ 0 (
    echo ✓ Redis Commander is running at http://localhost:8081
) else (
    echo ✗ Redis Commander is not responding
)

echo.
echo ========================================
echo Status check complete!
echo ========================================
pause
