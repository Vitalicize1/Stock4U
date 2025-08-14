# Database Implementation Guide

## Overview

Stock4U now includes a comprehensive database system using PostgreSQL and Redis for improved performance, data persistence, and analytics capabilities. The system provides:

- **PostgreSQL**: Primary database for structured data storage
- **Redis**: High-performance caching layer
- **SQLite**: Fallback database for development/testing
- **Unified API**: Consistent interface across all storage backends

## Architecture

### Database Layers

1. **PostgreSQL (Primary)**
   - Predictions and analysis results
   - User sessions and authentication
   - System metrics and monitoring
   - Daily picks and recommendations
   - Cache persistence

2. **Redis (Cache Layer)**
   - Fast access to frequently used data
   - Session storage
   - Real-time metrics
   - Temporary data storage

3. **SQLite (Fallback)**
   - Development and testing
   - Offline operation
   - Backup storage

### Key Components

- `utils/database.py`: Database connection management
- `models/database_models.py`: Database schema and models
- `utils/database_cache.py`: Unified cache system
- `utils/database_logger.py`: Database-backed logging
- `utils/init_database.py`: Database initialization utilities

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Using Docker Compose (Recommended)

```bash
# Start database services
docker-compose up -d postgres redis

# Initialize database
python utils/init_database.py
```

### 3. Manual Setup

#### PostgreSQL Setup

```bash
# Install PostgreSQL (Ubuntu/Debian)
sudo apt-get install postgresql postgresql-contrib

# Create database and user
sudo -u postgres psql
CREATE DATABASE stock4u;
CREATE USER stock4u_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE stock4u TO stock4u_user;
\q
```

#### Redis Setup

```bash
# Install Redis (Ubuntu/Debian)
sudo apt-get install redis-server

# Start Redis
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

### 4. Configuration

Create a `.env` file in the project root:

```env
# PostgreSQL Settings
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=stock4u
POSTGRES_USER=stock4u_user
POSTGRES_PASSWORD=your_password
POSTGRES_SSL_MODE=prefer

# Redis Settings
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=
REDIS_SSL=false

# Connection Pool Settings
POSTGRES_POOL_SIZE=10
POSTGRES_MAX_OVERFLOW=20
POSTGRES_POOL_TIMEOUT=30

# Feature Flags
USE_POSTGRES=true
USE_REDIS=true
FALLBACK_TO_SQLITE=true

# Development Settings
SQL_ECHO=false
```

## Database Models

### Prediction Model

Stores prediction results and analysis data:

```python
class Prediction(BaseModel):
    id = UUIDField(primary_key=True)
    ticker = CharField(max_length=10, index=True)
    timeframe = CharField(max_length=10, default="1d")
    timestamp = DateTimeField(default=datetime.utcnow, index=True)
    
    # Prediction results
    direction = CharField(max_length=10, null=True)  # UP, DOWN, NEUTRAL
    confidence = FloatField(null=True)
    predicted_price = FloatField(null=True)
    current_price = FloatField(null=True)
    
    # Analysis data (JSON)
    technical_analysis = TextField(null=True)
    sentiment_analysis = TextField(null=True)
    market_data = TextField(null=True)
    company_info = TextField(null=True)
    
    # Metadata
    prediction_engine = CharField(max_length=50, null=True)
    confidence_metrics = TextField(null=True)
    recommendation = TextField(null=True)
    
    # Actual results
    actual_direction = CharField(max_length=10, null=True)
    actual_price = FloatField(null=True)
    is_correct = BooleanField(null=True)
    
    # Performance tracking
    processing_time_ms = IntegerField(null=True)
    api_calls_count = IntegerField(default=0)
    cache_hit = BooleanField(default=False)
```

### CacheEntry Model

Manages cache entries with TTL support:

```python
class CacheEntry(BaseModel):
    id = UUIDField(primary_key=True)
    cache_key = CharField(max_length=255, unique=True, index=True)
    data = TextField()  # JSON string
    ttl_seconds = IntegerField(default=900)
    created_at = DateTimeField(default=datetime.utcnow, index=True)
    accessed_at = DateTimeField(default=datetime.utcnow)
    access_count = IntegerField(default=0)
```

### UserSession Model

Handles user sessions and authentication:

```python
class UserSession(BaseModel):
    id = UUIDField(primary_key=True)
    session_id = CharField(max_length=255, unique=True, index=True)
    user_id = CharField(max_length=255, null=True, index=True)
    ip_address = CharField(max_length=45, null=True)
    user_agent = TextField(null=True)
    is_active = BooleanField(default=True)
    expires_at = DateTimeField(null=True)
    session_data = TextField(null=True)  # JSON string
```

### SystemMetric Model

Stores system metrics and monitoring data:

```python
class SystemMetric(BaseModel):
    id = UUIDField(primary_key=True)
    metric_name = CharField(max_length=100, index=True)
    metric_value = FloatField()
    metric_unit = CharField(max_length=20, null=True)
    timestamp = DateTimeField(default=datetime.utcnow, index=True)
    context = TextField(null=True)  # JSON string
    tags = TextField(null=True)  # JSON string
```

### DailyPick Model

Manages daily stock picks and recommendations:

```python
class DailyPick(BaseModel):
    id = UUIDField(primary_key=True)
    date = DateField(index=True)
    ticker = CharField(max_length=10, index=True)
    recommendation = CharField(max_length=20)  # BUY, SELL, HOLD
    confidence = FloatField(null=True)
    reasoning = TextField(null=True)
    target_price = FloatField(null=True)
    stop_loss = FloatField(null=True)
    
    # Performance tracking
    actual_return = FloatField(null=True)
    hit_target = BooleanField(null=True)
    hit_stop_loss = BooleanField(null=True)
    
    # Metadata
    source = CharField(max_length=50, default="system")
    tags = TextField(null=True)  # JSON string
```

## Usage Examples

### Database Connection

```python
from utils.database import initialize_databases, get_postgres_session, get_redis_client

# Initialize databases
db_status = initialize_databases()
print(f"Database status: {db_status}")

# Use PostgreSQL session
with get_postgres_session() as session:
    result = session.execute("SELECT COUNT(*) FROM predictions")
    count = result.scalar()
    print(f"Total predictions: {count}")

# Use Redis client
redis_client = get_redis_client()
redis_client.set("test_key", "test_value", ex=3600)
value = redis_client.get("test_key")
print(f"Redis value: {value}")
```

### Cache Operations

```python
from utils.database_cache import db_cache

# Set cache
db_cache.set("prediction:AAPL:1d", {"direction": "UP", "confidence": 0.75}, ttl_seconds=900)

# Get cache
cached_data = db_cache.get("prediction:AAPL:1d")
if cached_data:
    print(f"Cached prediction: {cached_data}")

# Delete cache
db_cache.delete("prediction:AAPL:1d")

# Get cache statistics
stats = db_cache.get_stats()
print(f"Cache stats: {stats}")
```

### Prediction Logging

```python
from utils.database_logger import db_logger

# Log a prediction
prediction_id = db_logger.log_prediction("AAPL", {
    "prediction_result": {"direction": "UP", "confidence": 0.75},
    "technical_analysis": {...},
    "sentiment_analysis": {...}
}, timeframe="1d")

# Update with actual results
db_logger.update_prediction_result(prediction_id, "UP", 150.25)

# Get accuracy statistics
accuracy = db_logger.get_prediction_accuracy(days=30)
print(f"30-day accuracy: {accuracy['accuracy']}%")

# Get recent predictions
recent = db_logger.get_recent_predictions(ticker="AAPL", limit=10)
for pred in recent:
    print(f"{pred['timestamp']}: {pred['direction']} ({pred['confidence']})")
```

### Model Operations

```python
from models.database_models import Prediction, DailyPick, SystemMetric

# Create prediction from workflow result
prediction = Prediction.create_from_result("AAPL", workflow_result, "1d")

# Update prediction with actual results
prediction.update_actual_result("UP", 150.25)

# Get analysis data
analysis_data = prediction.get_analysis_data()

# Create daily pick
pick = DailyPick.create_pick(
    date=datetime.now(),
    ticker="AAPL",
    recommendation="BUY",
    confidence=0.75,
    reasoning="Strong technical indicators",
    target_price=155.0,
    stop_loss=145.0
)

# Record system metric
SystemMetric.record_metric(
    "prediction_processing_time_ms",
    1250.0,
    context={"ticker": "AAPL", "engine": "ensemble"}
)
```

## Database Management

### Initialization Script

```bash
# Full initialization
python utils/init_database.py

# Specific operations
python utils/init_database.py --create-tables
python utils/init_database.py --test-connections
python utils/init_database.py --show-stats
python utils/init_database.py --validate-config
```

### Table Management

```python
from models.database_models import create_tables, drop_tables

# Create tables
create_tables()

# Drop tables (use with caution!)
drop_tables()
```

### Data Cleanup

```python
from utils.database_logger import cleanup_old_data

# Clean up data older than 90 days
cleanup_stats = cleanup_old_data(days=90)
print(f"Cleaned up: {cleanup_stats}")
```

## Performance Optimization

### Connection Pooling

PostgreSQL connection pooling is configured with:

- Pool size: 10 connections
- Max overflow: 20 connections
- Pool timeout: 30 seconds
- Pre-ping: Enabled for connection health

### Caching Strategy

1. **Redis First**: Fast access for frequently used data
2. **PostgreSQL Fallback**: Persistent storage for all data
3. **Automatic Sync**: Data retrieved from PostgreSQL is cached in Redis
4. **TTL Management**: Automatic expiration of cached data

### Indexing

Key indexes are created for:

- `predictions`: (ticker, timestamp), (timestamp), (direction)
- `cache_entries`: (cache_key), (created_at), (accessed_at)
- `user_sessions`: (session_id), (user_id), (expires_at)
- `system_metrics`: (metric_name, timestamp), (timestamp)
- `daily_picks`: (date, ticker), (date), (ticker), (recommendation)

## Monitoring and Analytics

### System Metrics

The system automatically records metrics for:

- Prediction processing time
- Cache hit/miss rates
- Prediction accuracy
- API call counts
- User session activity

### Query Examples

```sql
-- Get prediction accuracy by engine
SELECT 
    prediction_engine,
    COUNT(*) as total_predictions,
    AVG(CASE WHEN is_correct THEN 1.0 ELSE 0.0 END) as accuracy
FROM predictions 
WHERE is_correct IS NOT NULL
GROUP BY prediction_engine;

-- Get cache performance
SELECT 
    metric_name,
    AVG(metric_value) as avg_value,
    COUNT(*) as count
FROM system_metrics 
WHERE metric_name LIKE 'cache_%'
GROUP BY metric_name;

-- Get daily picks performance
SELECT 
    date,
    COUNT(*) as total_picks,
    AVG(actual_return) as avg_return
FROM daily_picks 
WHERE actual_return IS NOT NULL
GROUP BY date
ORDER BY date DESC;
```

## Troubleshooting

### Common Issues

1. **Connection Errors**
   - Check database credentials in `.env`
   - Verify database services are running
   - Check network connectivity

2. **Import Errors**
   - Install required packages: `pip install psycopg2-binary redis sqlalchemy peewee`
   - Check Python path and imports

3. **Performance Issues**
   - Monitor connection pool usage
   - Check Redis memory usage
   - Review query performance with indexes

### Debug Mode

Enable SQL logging for debugging:

```env
SQL_ECHO=true
```

### Health Checks

```python
from utils.database import initialize_databases
from utils.database_cache import initialize_cache

# Check database status
db_status = initialize_databases()
cache_status = initialize_cache()

print(f"Database: {db_status}")
print(f"Cache: {cache_status}")
```

## Migration from File-Based Storage

The new database system is backward compatible with existing file-based storage:

1. **Cache**: Automatically falls back to file-based cache if databases unavailable
2. **Logging**: Maintains compatibility with existing prediction logger
3. **Gradual Migration**: Can run both systems simultaneously

### Migration Steps

1. Install database dependencies
2. Configure database connections
3. Initialize database tables
4. Run application (automatic fallback to file storage)
5. Verify database functionality
6. Gradually migrate data as needed

## Security Considerations

1. **Connection Security**
   - Use SSL for PostgreSQL connections
   - Secure Redis with authentication
   - Limit database access to application only

2. **Data Protection**
   - Encrypt sensitive data at rest
   - Implement proper access controls
   - Regular backup procedures

3. **Environment Variables**
   - Never commit credentials to version control
   - Use secure environment variable management
   - Rotate passwords regularly

## Backup and Recovery

### PostgreSQL Backup

```bash
# Create backup
pg_dump -h localhost -U postgres -d stock4u > backup.sql

# Restore backup
psql -h localhost -U postgres -d stock4u < backup.sql
```

### Redis Backup

```bash
# Redis automatically creates AOF files
# Manual backup
redis-cli BGSAVE
```

### Application-Level Backup

```python
from utils.database_logger import export_predictions_to_csv

# Export data for backup
export_predictions_to_csv("backup/predictions.csv", days=365)
```

## Future Enhancements

1. **Data Archiving**: Automatic archiving of old data
2. **Replication**: PostgreSQL read replicas for scaling
3. **Sharding**: Horizontal partitioning for large datasets
4. **Analytics**: Advanced analytics and reporting features
5. **Real-time Streaming**: Real-time data processing pipelines
