# Stock4U User-Friendly Scripts

This directory contains easy-to-use scripts to get Stock4U up and running quickly.

## 🚀 Quick Start Scripts

### `start_stock4u.bat` / `start_stock4u.sh`
**One-click setup for the full Stock4U experience**

- Starts all services (API, Dashboard, Database, Cache)
- Automatically creates environment configuration
- Opens dashboard in your browser
- Perfect for first-time users

**Usage:**
- **Windows**: Double-click `start_stock4u.bat`
- **Linux/macOS**: Run `./start_stock4u.sh`

### `start_dashboard.bat` / `start_dashboard.sh`
**Simpler setup for dashboard only**

- Sets up Python environment
- Installs dependencies
- Starts Streamlit dashboard
- No database required
- Good for testing or basic use

**Usage:**
- **Windows**: Double-click `start_dashboard.bat`
- **Linux/macOS**: Run `./start_dashboard.sh`

## 🛑 Stop Scripts

### `stop_stock4u.bat` / `stop_stock4u.sh`
**Cleanly stop all Stock4U services**

- Stops all Docker containers
- Cleans up resources
- Frees up system resources

**Usage:**
- **Windows**: Double-click `stop_stock4u.bat`
- **Linux/macOS**: Run `./stop_stock4u.sh`

## 📊 Status Scripts

### `check_status.bat`
**Check if all services are running properly**

- Shows Docker container status
- Tests service endpoints
- Provides health check results

**Usage:**
- **Windows**: Double-click `check_status.bat`

## 📁 Configuration Files

### `env.example`
**Template for environment configuration**

- Copy to `.env` and customize
- Contains all available settings
- Well-documented options

## 🔧 What Each Script Does

### Full Setup Scripts (`start_stock4u.*`)
1. **Check Prerequisites**: Docker, Docker Compose
2. **Create Environment**: Generate `.env` file with defaults
3. **Create Directories**: logs, cache, etc.
4. **Start Services**: All Docker containers
5. **Health Check**: Verify services are running
6. **Open Browser**: Launch dashboard automatically

### Dashboard Only Scripts (`start_dashboard.*`)
1. **Check Prerequisites**: Python 3.8+
2. **Setup Environment**: Virtual environment
3. **Install Dependencies**: Python packages
4. **Create Config**: Basic `.env` file
5. **Start Dashboard**: Streamlit application

### Stop Scripts (`stop_stock4u.*`)
1. **Stop Containers**: `docker-compose down`
2. **Clean Resources**: `docker system prune`
3. **Confirm Completion**: Status message

## 🎯 When to Use Each Option

### Use Full Setup When:
- You want all features (API, database, cache)
- You're doing development or production work
- You need persistent data storage
- You want to explore all capabilities

### Use Dashboard Only When:
- You just want to try the interface
- You don't need advanced features
- You want a quick demo
- You're on a system with limited resources

## 🛠️ Troubleshooting

### Script Won't Run:
- **Windows**: Right-click → "Run as administrator"
- **Linux/macOS**: `chmod +x script_name.sh`

### Services Won't Start:
- Check if Docker is running
- Ensure ports 8000, 8501, 8080, 8081 are free
- Verify internet connection for downloads

### Dashboard Issues:
- Check if Python is installed
- Ensure virtual environment is created
- Verify dependencies are installed

## 📞 Support

If you encounter issues:
1. Check the [main README](README.md)
2. Review the [Quick Start Guide](QUICK_START.md)
3. Look at the troubleshooting section
4. Open an issue on our repository

---

**These scripts make Stock4U accessible to everyone! 🚀**
