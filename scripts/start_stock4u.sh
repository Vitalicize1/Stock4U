#!/bin/bash

echo "========================================"
echo "           Stock4U Easy Setup"
echo "========================================"
echo

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed!"
    echo "Please install Docker from: https://docs.docker.com/get-docker/"
    echo "Then restart this script."
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "ERROR: Docker is not running!"
    echo "Please start Docker and try again."
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "ERROR: Docker Compose is not installed!"
    echo "Please install Docker Compose and try again."
    exit 1
fi

echo "Docker is ready! Starting Stock4U..."

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating default environment configuration..."
    if [ -f .env.example ]; then
        cp .env.example .env
    else
        echo "Creating basic .env file..."
        cat > .env << 'EOF'
# Stock4U Environment Configuration
# Generated automatically - edit as needed

# Database Configuration
POSTGRES_PASSWORD=stock4u_secure_password_123
REDIS_PASSWORD=stock4u_redis_password_456

# API Configuration
API_KEY=stock4u_api_key_789
JWT_SECRET_KEY=your_jwt_secret_key_change_this_in_production

# Optional API Keys (uncomment and add your keys)
# OPENAI_API_KEY=your_openai_key_here
# GOOGLE_API_KEY=your_google_key_here
# TAVILY_API_KEY=your_tavily_key_here

# Rate Limiting
RATE_LIMIT_PER_MIN=60

# Learning Configuration
LEARNING_SCHED_ENABLED=1
LEARNING_TICKERS=AAPL,MSFT,GOOGL,AMZN,TSLA
LEARNING_TIMEFRAMES=1d,5d
LEARNING_PERIOD=1mo
LEARNING_ITERATIONS=10
LEARNING_LR=0.001
LEARNING_CRON=0 2 * * *

# PgAdmin Configuration (Development)
PGADMIN_EMAIL=admin@stock4u.com
PGADMIN_PASSWORD=admin_password
EOF
        echo "Created .env file with default settings."
    fi
fi

# Create necessary directories
mkdir -p logs cache logs/nginx

echo
echo "Starting Stock4U services..."
echo "This may take a few minutes on first run..."
echo

# Start the services
docker-compose up -d

# Wait for services to be ready
echo
echo "Waiting for services to start..."
sleep 10

# Check if services are running
echo
echo "Checking service status..."
docker-compose ps

echo
echo "========================================"
echo "           Stock4U is Ready!"
echo "========================================"
echo
echo "Services available at:"
echo "- Main API: http://localhost:8000"
echo "- Dashboard: http://localhost:8501"
echo "- PgAdmin (Database): http://localhost:8080"
echo "- Redis Commander: http://localhost:8081"
echo
echo "Default credentials:"
echo "- PgAdmin: admin@stock4u.com / admin_password"
echo
echo "To stop Stock4U, run: docker-compose down"
echo "To view logs, run: docker-compose logs -f"
echo

# Try to open dashboard in browser
if command -v xdg-open &> /dev/null; then
    # Linux
    xdg-open http://localhost:8501 &
elif command -v open &> /dev/null; then
    # macOS
    open http://localhost:8501 &
elif command -v start &> /dev/null; then
    # Windows (if running in WSL)
    start http://localhost:8501 &
fi

echo "Enjoy using Stock4U!"
