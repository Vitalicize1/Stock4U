# Stock4U Daily Picks - Permanent Setup Guide

This guide sets up a centralized daily picks service that all your Stock4U user servers can access.

## Quick Setup (Recommended)

### Option 1: Standalone Service (Windows)
```bash
# 1. Navigate to your Stock4U directory
cd D:\Stock4U

# 2. Run the daily picks service
deployment\start_daily_picks_service.bat

# 3. Service will be available at:
# http://localhost:8001/daily_picks
```

### Option 2: Docker (Production)
```bash
# 1. Start the containerized service
docker-compose -f deployment/docker-compose.daily-picks.yml up -d

# 2. Service will be available at:
# http://localhost:8001/daily_picks
```

## Configure All User Servers

Once the service is running, configure all your Streamlit instances:

### Environment Variable
Set this on every user server (Streamlit Cloud, local, etc.):
```bash
DAILY_PICKS_URL=http://YOUR_DEV_SERVER_IP:8001/daily_picks
```

### Examples:
- **Local development**: `DAILY_PICKS_URL=http://localhost:8001/daily_picks`
- **Remote server**: `DAILY_PICKS_URL=http://192.168.1.100:8001/daily_picks`
- **Domain**: `DAILY_PICKS_URL=https://picks.yourstock4u.com/daily_picks`

## Service Features

### Endpoints
- `GET /daily_picks` - Get current daily picks (main endpoint)
- `GET /health` - Service health check
- `POST /generate` - Manually trigger picks generation
- `GET /picks/status` - Detailed status information

### Auto-Generation
- Automatically generates fresh picks when data is stale (>24h old)
- Configurable via `DAILY_PICKS_AUTO_GENERATE=true`
- Manual generation available via `/generate` endpoint

### Configuration
Environment variables for the service:

```bash
# Core settings
DAILY_PICKS_PATH=cache/daily_picks.json
DAILY_PICKS_AUTO_GENERATE=true
DAILY_PICKS_MAX_AGE_HOURS=24
DAILY_PICKS_PORT=8001

# Generation settings
DAILY_PICKS_TIMEFRAME=1d
DAILY_PICKS_TOP_N=3
DAILY_PICKS_LOW_API=1      # Use low-API mode for background jobs
DAILY_PICKS_FAST_TA=1      # Use fast technical analysis
```

## Deployment Options

### 1. Development Server
- Run `deployment\start_daily_picks_service.bat`
- Service runs on port 8001
- All user servers point to this URL

### 2. Docker Production
- Use `deployment/docker-compose.daily-picks.yml`
- Includes optional Nginx proxy with caching
- SSL support available

### 3. Cloud Deployment
- Deploy `deployment/daily_picks_service.py` to any cloud platform
- Set environment variables as needed
- Point all user servers to the deployed URL

## Testing

### Verify Service
```bash
# Check health
curl http://localhost:8001/health

# Get picks
curl http://localhost:8001/daily_picks

# Generate fresh picks
curl -X POST http://localhost:8001/generate
```

### Verify User Server
1. Set `DAILY_PICKS_URL` in user server environment
2. Restart Streamlit app
3. Go to "Daily Picks & Analysis" tab
4. Should show the same picks as the service endpoint

## Troubleshooting

### Service Not Starting
- Check Python version (3.8+ required)
- Ensure all dependencies installed: `pip install -r requirements.txt`
- Check port 8001 is available

### User Servers Not Updating
- Verify `DAILY_PICKS_URL` is set correctly
- Check network connectivity to the service
- Restart Streamlit apps after setting environment variable

### Stale Data
- Service auto-generates when data is >24h old
- Manually trigger: `curl -X POST http://localhost:8001/generate`
- Check service logs for errors

## Security Notes

- Service allows CORS from all origins for public access
- No authentication required for read-only endpoints
- Consider using HTTPS and rate limiting for production
- The included Nginx config provides basic security headers

## Monitoring

### Health Checks
- `GET /health` - Basic service health
- `GET /picks/status` - Detailed picks information
- Docker health checks included

### Logs
- Service logs to stdout/stderr
- Docker logs: `docker-compose logs daily-picks-service`
- Windows: Check console output

## Production Recommendations

1. **Use Docker** for consistent deployment
2. **Set up SSL** with proper domain name
3. **Enable monitoring** and alerting
4. **Use reverse proxy** (Nginx) for caching and security
5. **Regular backups** of picks data if needed
6. **Monitor service uptime** and auto-restart on failure

## Support

If you encounter issues:
1. Check service health endpoint
2. Verify environment variables
3. Check network connectivity
4. Review service logs
5. Test manual generation endpoint

The service is designed to be robust and self-healing, automatically generating fresh picks when needed.
