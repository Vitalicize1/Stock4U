#!/bin/bash

echo "========================================"
echo "        Stock4U Dashboard Setup"
echo "========================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed!"
    echo "Please install Python 3.8+ and try again."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create virtual environment!"
        exit 1
    fi
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements if needed
if [ ! -d "venv/lib/python*/site-packages/streamlit" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install dependencies!"
        exit 1
    fi
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating basic .env file..."
    cat > .env << 'EOF'
# Stock4U Environment Configuration
# Basic setup for dashboard only

# Optional API Keys (uncomment and add your keys for full features)
# OPENAI_API_KEY=your_openai_key_here
# GOOGLE_API_KEY=your_google_key_here
# TAVILY_API_KEY=your_tavily_key_here
EOF
    echo "Created .env file with basic settings."
fi

echo
echo "Starting Stock4U Dashboard..."
echo "Dashboard will open in your browser at: http://localhost:8501"
echo
echo "Press Ctrl+C to stop the dashboard"
echo

# Start the dashboard
streamlit run dashboard.py --server.port 8501 --server.headless false
