"""
Database Models for Stock4U

This module defines the database schema using Peewee ORM for:
- Predictions and analysis results
- Cache management
- User sessions and authentication
- System metrics and monitoring
- Historical data storage
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from peewee import *
# Optional Postgres extensions; safe to skip when not installed
try:
    from playhouse.postgres_ext import *  # type: ignore
except Exception:  # pragma: no cover - not required for SQLite
    pass
from utils.database import get_peewee_db

# Database instance - will be initialized lazily
_database = None

def get_database():
    """Get database instance with lazy initialization."""
    global _database
    if _database is None:
        try:
            _database = get_peewee_db()
        except Exception:
            # Fallback to file-based SQLite to persist locally
            from peewee import SqliteDatabase
            from pathlib import Path
            Path('cache').mkdir(exist_ok=True)
            _database = SqliteDatabase('cache/stock4u.db')
    return _database


class BaseModel(Model):
    """Base model with common fields and methods."""
    
    class Meta:
        database = get_database()
    
    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)
    
    def save(self, *args, **kwargs):
        """Override save to update the updated_at timestamp."""
        self.updated_at = datetime.utcnow()
        return super().save(*args, **kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model instance to dictionary."""
        data = {}
        for field in self._meta.fields:
            value = getattr(self, field.name)
            if isinstance(value, datetime):
                value = value.isoformat()
            elif isinstance(value, (dict, list)):
                value = json.dumps(value) if value else None
            data[field.name] = value
        return data


class User(BaseModel):
    """User accounts for dashboard authentication."""

    id = UUIDField(primary_key=True, default=uuid.uuid4)
    username = CharField(max_length=50, unique=True, index=True)
    password_hash = CharField(max_length=255)
    email = CharField(max_length=255, null=True, unique=True)
    role = CharField(max_length=20, default="user")
    is_active = BooleanField(default=True)
    last_login = DateTimeField(null=True)

    class Meta:
        table_name = 'users'

    @classmethod
    def get_by_username(cls, username: str) -> Optional['User']:
        try:
            return cls.get(cls.username == username)
        except cls.DoesNotExist:
            return None

    @classmethod
    def get_by_email(cls, email: str) -> Optional['User']:
        try:
            return cls.get((cls.email.is_null(False)) & (cls.email == email))
        except cls.DoesNotExist:
            return None

class Prediction(BaseModel):
    """Model for storing prediction results and analysis."""
    
    id = UUIDField(primary_key=True, default=uuid.uuid4)
    ticker = CharField(max_length=10, index=True)
    timeframe = CharField(max_length=10, default="1d")
    timestamp = DateTimeField(default=datetime.utcnow, index=True)
    
    # Prediction results
    direction = CharField(max_length=10, null=True)  # UP, DOWN, NEUTRAL
    confidence = FloatField(null=True)
    predicted_price = FloatField(null=True)
    current_price = FloatField(null=True)
    
    # Analysis data (stored as JSON)
    technical_analysis = TextField(null=True)  # JSON string
    sentiment_analysis = TextField(null=True)  # JSON string
    market_data = TextField(null=True)  # JSON string
    company_info = TextField(null=True)  # JSON string
    
    # Prediction metadata
    prediction_engine = CharField(max_length=50, null=True)  # ensemble, ml, llm, rule
    confidence_metrics = TextField(null=True)  # JSON string
    recommendation = TextField(null=True)  # JSON string
    
    # Actual results (updated later)
    actual_direction = CharField(max_length=10, null=True)
    actual_price = FloatField(null=True)
    is_correct = BooleanField(null=True)
    
    # Performance tracking
    processing_time_ms = IntegerField(null=True)
    api_calls_count = IntegerField(default=0)
    cache_hit = BooleanField(default=False)
    
    class Meta:
        table_name = 'predictions'
        indexes = (
            (('ticker', 'timestamp'), False),  # Non-unique index
            (('timestamp',), False),
            (('direction',), False),
        )
    
    @classmethod
    def create_from_result(cls, ticker: str, result: Dict[str, Any], timeframe: str = "1d") -> 'Prediction':
        """Create a prediction record from workflow result."""
        prediction_data = result.get("prediction_result", {})
        
        # Extract prediction details
        direction = prediction_data.get("direction")
        confidence = prediction_data.get("confidence")
        predicted_price = prediction_data.get("predicted_price")
        current_price = prediction_data.get("current_price")
        
        # Store analysis as JSON
        technical_analysis = json.dumps(result.get("technical_analysis", {}))
        sentiment_analysis = json.dumps(result.get("sentiment_analysis", {}))
        market_data = json.dumps(result.get("data", {}).get("market_data", {}))
        company_info = json.dumps(result.get("data", {}).get("company_info", {}))
        
        # Extract confidence metrics and recommendation
        confidence_metrics = json.dumps(result.get("confidence_metrics", {}))
        recommendation = json.dumps(result.get("recommendation", {}))
        
        # Determine prediction engine
        prediction_engine = "ensemble"  # Default
        if result.get("use_ml_model"):
            prediction_engine = "ml"
        elif result.get("low_api_mode"):
            prediction_engine = "rule"
        
        return cls.create(
            ticker=ticker,
            timeframe=timeframe,
            direction=direction,
            confidence=confidence,
            predicted_price=predicted_price,
            current_price=current_price,
            technical_analysis=technical_analysis,
            sentiment_analysis=sentiment_analysis,
            market_data=market_data,
            company_info=company_info,
            confidence_metrics=confidence_metrics,
            recommendation=recommendation,
            prediction_engine=prediction_engine,
            processing_time_ms=result.get("timings", {}).get("total_s", 0) * 1000,
            cache_hit=result.get("cache_hit", False)
        )
    
    def update_actual_result(self, actual_direction: str, actual_price: float) -> None:
        """Update prediction with actual market results."""
        self.actual_direction = actual_direction
        self.actual_price = actual_price
        
        # Determine if prediction was correct
        if self.direction and actual_direction:
            self.is_correct = self.direction.upper() == actual_direction.upper()
        
        self.save()
    
    def get_analysis_data(self) -> Dict[str, Any]:
        """Get analysis data as dictionary."""
        data = {}
        for field in ['technical_analysis', 'sentiment_analysis', 'market_data', 'company_info', 'confidence_metrics', 'recommendation']:
            value = getattr(self, field)
            if value:
                try:
                    data[field] = json.loads(value)
                except json.JSONDecodeError:
                    data[field] = None
            else:
                data[field] = None
        return data


class CacheEntry(BaseModel):
    """Model for storing cache entries with TTL support."""
    
    id = UUIDField(primary_key=True, default=uuid.uuid4)
    cache_key = CharField(max_length=255, unique=True, index=True)
    data = TextField()  # JSON string
    ttl_seconds = IntegerField(default=900)  # 15 minutes default
    created_at = DateTimeField(default=datetime.utcnow, index=True)
    accessed_at = DateTimeField(default=datetime.utcnow)
    access_count = IntegerField(default=0)
    
    class Meta:
        table_name = 'cache_entries'
        indexes = (
            (('created_at',), False),
            (('accessed_at',), False),
        )
    
    @classmethod
    def is_valid(cls, cache_key: str, ttl_seconds: int = 900) -> bool:
        """Check if a cache entry is still valid."""
        try:
            entry = cls.get(cls.cache_key == cache_key)
            if not entry:
                return False
            
            # Check if expired
            expiry_time = entry.created_at + timedelta(seconds=entry.ttl_seconds)
            return datetime.utcnow() < expiry_time
        except cls.DoesNotExist:
            return False
    
    @classmethod
    def get_data(cls, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached data if valid."""
        try:
            entry = cls.get(cls.cache_key == cache_key)
            
            # Check if expired
            expiry_time = entry.created_at + timedelta(seconds=entry.ttl_seconds)
            if datetime.utcnow() >= expiry_time:
                entry.delete_instance()
                return None
            
            # Update access info
            entry.accessed_at = datetime.utcnow()
            entry.access_count += 1
            entry.save()
            
            return json.loads(entry.data)
        except (cls.DoesNotExist, json.JSONDecodeError):
            return None
    
    @classmethod
    def set_data(cls, cache_key: str, data: Dict[str, Any], ttl_seconds: int = 900) -> None:
        """Set cached data."""
        data_json = json.dumps(data)
        
        # Use upsert to handle existing entries
        cls.insert(
            cache_key=cache_key,
            data=data_json,
            ttl_seconds=ttl_seconds,
            created_at=datetime.utcnow(),
            accessed_at=datetime.utcnow(),
            access_count=1
        ).on_conflict('replace').execute()
    
    @classmethod
    def cleanup_expired(cls) -> int:
        """Remove expired cache entries. Returns count of removed entries."""
        now = datetime.utcnow()
        expired_entries = cls.select().where(
            cls.created_at + (cls.ttl_seconds * 1000000) < now  # Convert to microseconds
        )
        count = expired_entries.count()
        expired_entries.delete_instance()
        return count


class UserSession(BaseModel):
    """Model for storing user sessions and authentication."""
    
    id = UUIDField(primary_key=True, default=uuid.uuid4)
    session_id = CharField(max_length=255, unique=True, index=True)
    user_id = CharField(max_length=255, null=True, index=True)
    ip_address = CharField(max_length=45, null=True)  # IPv6 compatible
    user_agent = TextField(null=True)
    is_active = BooleanField(default=True)
    expires_at = DateTimeField(null=True)
    
    # Session data
    session_data = TextField(null=True)  # JSON string
    
    class Meta:
        table_name = 'user_sessions'
        indexes = (
            (('session_id',), True),
            (('user_id',), False),
            (('expires_at',), False),
        )
    
    @classmethod
    def create_session(cls, session_id: str, user_id: Optional[str] = None, 
                      ip_address: Optional[str] = None, user_agent: Optional[str] = None,
                      expires_in_hours: int = 24) -> 'UserSession':
        """Create a new user session."""
        expires_at = datetime.utcnow() + timedelta(hours=expires_in_hours)
        
        return cls.create(
            session_id=session_id,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            expires_at=expires_at
        )
    
    @classmethod
    def get_valid_session(cls, session_id: str) -> Optional['UserSession']:
        """Get a valid session by ID."""
        try:
            session = cls.get(
                (cls.session_id == session_id) & 
                (cls.is_active == True) & 
                ((cls.expires_at.is_null()) | (cls.expires_at > datetime.utcnow()))
            )
            return session
        except cls.DoesNotExist:
            return None
    
    def update_session_data(self, data: Dict[str, Any]) -> None:
        """Update session data."""
        self.session_data = json.dumps(data)
        self.save()
    
    def get_session_data(self) -> Dict[str, Any]:
        """Get session data as dictionary."""
        if self.session_data:
            try:
                return json.loads(self.session_data)
            except json.JSONDecodeError:
                return {}
        return {}
    
    @classmethod
    def cleanup_expired(cls) -> int:
        """Remove expired sessions. Returns count of removed sessions."""
        expired_sessions = cls.select().where(
            (cls.expires_at.is_null(False)) & (cls.expires_at < datetime.utcnow())
        )
        count = expired_sessions.count()
        expired_sessions.delete_instance()
        return count


class SystemMetric(BaseModel):
    """Model for storing system metrics and monitoring data."""
    
    id = UUIDField(primary_key=True, default=uuid.uuid4)
    metric_name = CharField(max_length=100, index=True)
    metric_value = FloatField()
    metric_unit = CharField(max_length=20, null=True)
    timestamp = DateTimeField(default=datetime.utcnow, index=True)
    
    # Additional context
    context = TextField(null=True)  # JSON string
    tags = TextField(null=True)  # JSON string for key-value pairs
    
    class Meta:
        table_name = 'system_metrics'
        indexes = (
            (('metric_name', 'timestamp'), False),
            (('timestamp',), False),
        )
    
    @classmethod
    def record_metric(cls, name: str, value: float, unit: Optional[str] = None,
                     context: Optional[Dict[str, Any]] = None, tags: Optional[Dict[str, str]] = None) -> 'SystemMetric':
        """Record a system metric."""
        return cls.create(
            metric_name=name,
            metric_value=value,
            metric_unit=unit,
            context=json.dumps(context) if context else None,
            tags=json.dumps(tags) if tags else None
        )
    
    @classmethod
    def get_metrics(cls, name: str, hours: int = 24) -> List['SystemMetric']:
        """Get metrics for a specific name within the last N hours."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return cls.select().where(
            (cls.metric_name == name) & (cls.timestamp >= cutoff_time)
        ).order_by(cls.timestamp.desc())
    
    @classmethod
    def get_latest_metric(cls, name: str) -> Optional['SystemMetric']:
        """Get the latest metric for a specific name."""
        try:
            return cls.select().where(cls.metric_name == name).order_by(cls.timestamp.desc()).first()
        except cls.DoesNotExist:
            return None
    
    @classmethod
    def cleanup_old_metrics(cls, days: int = 30) -> int:
        """Remove metrics older than N days. Returns count of removed metrics."""
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        old_metrics = cls.select().where(cls.timestamp < cutoff_time)
        count = old_metrics.count()
        old_metrics.delete_instance()
        return count


class DailyPick(BaseModel):
    """Model for storing daily stock picks and recommendations."""
    
    id = UUIDField(primary_key=True, default=uuid.uuid4)
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
    source = CharField(max_length=50, default="system")  # system, user, external
    tags = TextField(null=True)  # JSON string
    
    class Meta:
        table_name = 'daily_picks'
        indexes = (
            (('date', 'ticker'), True),  # Unique constraint
            (('date',), False),
            (('ticker',), False),
            (('recommendation',), False),
        )
    
    @classmethod
    def create_pick(cls, date: datetime, ticker: str, recommendation: str,
                   confidence: Optional[float] = None, reasoning: Optional[str] = None,
                   target_price: Optional[float] = None, stop_loss: Optional[float] = None,
                   source: str = "system", tags: Optional[Dict[str, str]] = None) -> 'DailyPick':
        """Create a daily pick."""
        return cls.create(
            date=date.date() if isinstance(date, datetime) else date,
            ticker=ticker,
            recommendation=recommendation,
            confidence=confidence,
            reasoning=reasoning,
            target_price=target_price,
            stop_loss=stop_loss,
            source=source,
            tags=json.dumps(tags) if tags else None
        )
    
    @classmethod
    def get_picks_for_date(cls, date: datetime) -> List['DailyPick']:
        """Get all picks for a specific date."""
        return cls.select().where(cls.date == date.date()).order_by(cls.ticker)
    
    @classmethod
    def get_picks_for_ticker(cls, ticker: str, days: int = 30) -> List['DailyPick']:
        """Get picks for a specific ticker within the last N days."""
        cutoff_date = datetime.utcnow().date() - timedelta(days=days)
        return cls.select().where(
            (cls.ticker == ticker) & (cls.date >= cutoff_date)
        ).order_by(cls.date.desc())
    
    def update_performance(self, actual_return: float, hit_target: Optional[bool] = None,
                          hit_stop_loss: Optional[bool] = None) -> None:
        """Update pick with performance data."""
        self.actual_return = actual_return
        if hit_target is not None:
            self.hit_target = hit_target
        if hit_stop_loss is not None:
            self.hit_stop_loss = hit_stop_loss
        self.save()


# Create tables
def create_tables():
    """Create all database tables using the active database (Postgres or SQLite)."""
    tables = [Prediction, CacheEntry, UserSession, SystemMetric, DailyPick, User]
    db = get_database()
    # Ensure models are bound to the active database
    for m in tables:
        m._meta.set_database(db)
    with db:
        db.create_tables(tables, safe=True)
    try:
        print("✅ Database tables created successfully")
    except Exception:
        pass


def drop_tables():
    """Drop all database tables (use with caution!)."""
    tables = [Prediction, CacheEntry, UserSession, SystemMetric, DailyPick, User]
    db = get_database()
    with db:
        db.drop_tables(tables, safe=True)
    try:
        print("🗑️ Database tables dropped successfully")
    except Exception:
        pass


# Initialize tables when module is imported
if __name__ == "__main__":
    create_tables()
