@echo off
echo ========================================
echo           Stock4U Easy Setup
echo ========================================
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not installed or not running!
    echo Please install Docker Desktop from: https://www.docker.com/products/docker-desktop/
    echo Then restart this script.
    pause
    exit /b 1
)

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running!
    echo Please start Docker Desktop and try again.
    pause
    exit /b 1
)

echo Docker is ready! Starting Stock4U...

REM Create .env file if it doesn't exist
if not exist .env (
    echo Creating default environment configuration...
    copy .env.example .env >nul 2>&1
    if %errorlevel% neq 0 (
        echo Creating basic .env file...
        (
            echo # Stock4U Environment Configuration
            echo # Generated automatically - edit as needed
            echo.
            echo # Database Configuration
            echo POSTGRES_PASSWORD=stock4u_secure_password_123
            echo REDIS_PASSWORD=stock4u_redis_password_456
            echo.
            echo # API Configuration
            echo API_KEY=stock4u_api_key_789
            echo JWT_SECRET_KEY=your_jwt_secret_key_change_this_in_production
            echo.
            echo # Optional API Keys (uncomment and add your keys)
            echo # OPENAI_API_KEY=your_openai_key_here
            echo # GOOGLE_API_KEY=your_google_key_here
            echo # TAVILY_API_KEY=your_tavily_key_here
            echo.
            echo # Rate Limiting
            echo RATE_LIMIT_PER_MIN=60
            echo.
            echo # Learning Configuration
            echo LEARNING_SCHED_ENABLED=1
            echo LEARNING_TICKERS=AAPL,MSFT,GOOGL,AMZN,TSLA
            echo LEARNING_TIMEFRAMES=1d,5d
            echo LEARNING_PERIOD=1mo
            echo LEARNING_ITERATIONS=10
            echo LEARNING_LR=0.001
            echo LEARNING_CRON=0 2 * * *
            echo.
            echo # PgAdmin Configuration (Development)
            echo PGADMIN_EMAIL=admin@stock4u.com
            echo PGADMIN_PASSWORD=admin_password
        ) > .env
        echo Created .env file with default settings.
    )
)

REM Create necessary directories
if not exist logs mkdir logs
if not exist cache mkdir cache
if not exist logs\nginx mkdir logs\nginx

echo.
echo Starting Stock4U services...
echo This may take a few minutes on first run...
echo.

REM Start the services
docker-compose -f ops\docker-compose.yml up -d

REM Wait for services to be ready
echo.
echo Waiting for services to start...
timeout /t 10 /nobreak >nul

REM Check if services are running
echo.
echo Checking service status...
docker-compose -f ops\docker-compose.yml ps

echo.
echo ========================================
echo           Stock4U is Ready!
echo ========================================
echo.
echo Services available at:
echo - Main API: http://localhost:8000
echo - Dashboard: http://localhost:8501
echo - PgAdmin (Database): http://localhost:8080
echo - Redis Commander: http://localhost:8081
echo.
echo Default credentials:
echo - PgAdmin: admin@stock4u.com / admin_password
echo.
echo To stop Stock4U, run: docker-compose -f ops\docker-compose.yml down
echo To view logs, run: docker-compose -f ops\docker-compose.yml logs -f
echo.
echo Press any key to open the dashboard...
pause >nul

REM Open dashboard in browser
start http://localhost:8501

echo.
echo Enjoy using Stock4U!
pause
