# Stock4U Monitoring Guide

## 🎯 Overview

This guide covers setting up comprehensive monitoring for your Stock4U production deployment using the industry-standard **Grafana + Prometheus** stack.

## 🚀 Quick Start

### 1. Start Monitoring Stack

```powershell
# Windows PowerShell
.\scripts\start_monitoring.ps1

# Or manually:
docker-compose -f ops/docker-compose.monitoring.yml up -d
```

### 2. Access Dashboards

- **Grafana**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **AlertManager**: http://localhost:9093

## 📊 What's Monitored

### System Metrics
- **API Health**: `stock4u_up` (1 = healthy, 0 = down)
- **Memory Usage**: `stock4u_process_memory_bytes`
- **CPU Usage**: `stock4u_process_cpu_percent`
- **Thread Count**: `stock4u_process_threads`

### Business Metrics
- **Learning Job Duration**: `stock4u_learning_last_elapsed_seconds`
- **Cache Size**: `stock4u_cache_size_bytes`
- **Database Connection**: `stock4u_database_connected`

### API Metrics
- **Request Rate**: `api_request_total`
- **Response Time**: `api_request_ms`
- **Error Rate**: By endpoint and status

## 🔧 Configuration

### Environment Variables

```bash
# API Token for monitoring
API_TOKEN=your-secure-token

# Rate limiting
RATE_LIMIT_PER_MIN=60

# Learning configuration
LEARNING_SCHED_ENABLED=1
LEARNING_TICKERS=AAPL,MSFT,GOOGL,NVDA,AMZN
LEARNING_TIMEFRAMES=1d,1w
LEARNING_PERIOD=1y
LEARNING_ITERATIONS=120
LEARNING_LR=0.08
LEARNING_CRON="30 2 * * *"
```

### Prometheus Configuration

The Prometheus configuration (`ops/prometheus/prometheus.yml`) includes:
- 15-second scrape intervals
- Bearer token authentication
- Alerting rules integration
- AlertManager integration

### Alerting Rules

Key alerts configured:
- **API Down**: Triggers when `stock4u_up == 0`
- **Learning Job Slow**: Triggers when learning takes >1 hour
- **High Response Time**: Triggers when API response >5 seconds

## 📈 Grafana Dashboards

### Default Dashboard

The Stock4U dashboard includes:
1. **API Health Panel**: Shows system status
2. **Learning Job Duration**: Tracks learning performance
3. **API Request Rate**: Monitors traffic patterns
4. **API Response Time**: Tracks performance

### Custom Queries

Useful PromQL queries for custom dashboards:

```promql
# Request rate by endpoint
rate(api_request_total[5m])

# Error rate
rate(api_request_total{status="error"}[5m])

# 95th percentile response time
histogram_quantile(0.95, rate(api_request_ms_bucket[5m]))

# Memory usage trend
stock4u_process_memory_bytes

# Learning job frequency
rate(stock4u_learning_last_timestamp[1h])
```

## 🚨 Alerting

### AlertManager Configuration

Alerts are routed to:
- **Webhook**: http://127.0.0.1:5001/ (configurable)
- **Email**: Configure in `ops/alertmanager/alertmanager.yml`
- **Slack**: Add webhook URL for team notifications

### Alert Severity Levels

- **Critical**: API down, database disconnected
- **Warning**: High response time, learning job slow
- **Info**: Cache size warnings, performance degradation

## 🔍 Troubleshooting

### Common Issues

1. **Prometheus can't scrape metrics**
   - Check API token in `ops/prometheus/token.txt`
   - Verify API is running on correct port
   - Check network connectivity

2. **Grafana can't connect to Prometheus**
   - Verify Prometheus is running: `docker ps`
   - Check datasource configuration
   - Restart Grafana container

3. **No metrics appearing**
   - Check API `/metrics` endpoint directly
   - Verify authentication is working
   - Check logs: `docker logs stock4u-prometheus`

### Useful Commands

```bash
# Check service status
docker-compose -f ops/docker-compose.monitoring.yml ps

# View logs
docker logs stock4u-prometheus
docker logs stock4u-grafana
docker logs stock4u-alertmanager

# Test metrics endpoint
curl -H "Authorization: Bearer YOUR_TOKEN" http://localhost:8000/metrics

# Restart services
docker-compose -f ops/docker-compose.monitoring.yml restart
```

## 📋 Production Checklist

### Before Going Live

- [ ] Set secure API token
- [ ] Configure email/Slack alerts
- [ ] Set up backup for monitoring data
- [ ] Configure log retention policies
- [ ] Test alert notifications
- [ ] Set up dashboard access for team
- [ ] Configure SSL/TLS for external access

### Regular Maintenance

- [ ] Review alert thresholds monthly
- [ ] Clean up old metrics data
- [ ] Update Grafana dashboards
- [ ] Monitor disk usage
- [ ] Review and tune Prometheus retention

## 🔐 Security Considerations

### Access Control
- Use strong API tokens
- Restrict network access to monitoring ports
- Consider VPN for remote access
- Implement user authentication in Grafana

### Data Protection
- Encrypt sensitive metrics data
- Implement log rotation
- Regular security updates
- Monitor for suspicious activity

## 📚 Advanced Topics

### Custom Metrics

Add custom business metrics:

```python
from utils.logger import log_metric, increment

# Track prediction accuracy
log_metric("prediction_accuracy", 0.75, {"ticker": "AAPL"})

# Count successful predictions
increment("predictions_successful", 1, {"timeframe": "1d"})
```

### External Monitoring

Integrate with external services:
- **Uptime Robot**: For external health checks
- **PagerDuty**: For incident management
- **DataDog**: For advanced APM
- **New Relic**: For application performance

### Scaling Monitoring

For high-traffic deployments:
- Use Prometheus federation
- Implement metrics aggregation
- Consider Thanos for long-term storage
- Use Grafana Enterprise for advanced features

## 🆘 Support

For monitoring issues:
1. Check the troubleshooting section above
2. Review container logs
3. Verify configuration files
4. Test individual components
5. Check network connectivity

---

**Remember**: Good monitoring is the foundation of reliable production systems. Start simple and expand as your needs grow!
