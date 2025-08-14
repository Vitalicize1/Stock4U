"""
Database Configuration and Connection Management

This module provides centralized database configuration for PostgreSQL and Redis,
with support for environment-based configuration and connection pooling.
"""

import os
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager
import redis
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from peewee import PostgresqlDatabase, SqliteDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class DatabaseConfig:
    """Centralized database configuration."""
    
    # PostgreSQL Configuration
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB = os.getenv("POSTGRES_DB", "stock4u")
    POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
    POSTGRES_SSL_MODE = os.getenv("POSTGRES_SSL_MODE", "prefer")
    
    # Redis Configuration
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
    REDIS_SSL = os.getenv("REDIS_SSL", "false").lower() == "true"
    
    # Connection Pool Settings
    POSTGRES_POOL_SIZE = int(os.getenv("POSTGRES_POOL_SIZE", "10"))
    POSTGRES_MAX_OVERFLOW = int(os.getenv("POSTGRES_MAX_OVERFLOW", "20"))
    POSTGRES_POOL_TIMEOUT = int(os.getenv("POSTGRES_POOL_TIMEOUT", "30"))
    
    # Feature Flags
    USE_POSTGRES = os.getenv("USE_POSTGRES", "true").lower() == "true"
    USE_REDIS = os.getenv("USE_REDIS", "true").lower() == "true"
    FALLBACK_TO_SQLITE = os.getenv("FALLBACK_TO_SQLITE", "true").lower() == "true"
    
    @classmethod
    def get_postgres_url(cls) -> str:
        """Get PostgreSQL connection URL."""
        if cls.POSTGRES_PASSWORD:
            return f"postgresql://{cls.POSTGRES_USER}:{cls.POSTGRES_PASSWORD}@{cls.POSTGRES_HOST}:{cls.POSTGRES_PORT}/{cls.POSTGRES_DB}?sslmode={cls.POSTGRES_SSL_MODE}"
        else:
            return f"postgresql://{cls.POSTGRES_USER}@{cls.POSTGRES_HOST}:{cls.POSTGRES_PORT}/{cls.POSTGRES_DB}?sslmode={cls.POSTGRES_SSL_MODE}"
    
    @classmethod
    def get_redis_url(cls) -> str:
        """Get Redis connection URL."""
        if cls.REDIS_PASSWORD:
            return f"redis://:{cls.REDIS_PASSWORD}@{cls.REDIS_HOST}:{cls.REDIS_PORT}/{cls.REDIS_DB}"
        else:
            return f"redis://{cls.REDIS_HOST}:{cls.REDIS_PORT}/{cls.REDIS_DB}"
    
    @classmethod
    def validate_config(cls) -> Dict[str, bool]:
        """Validate database configuration and return status."""
        status = {
            "postgres_configured": bool(cls.POSTGRES_DB and cls.POSTGRES_USER),
            "redis_configured": bool(cls.REDIS_HOST),
            "use_postgres": cls.USE_POSTGRES,
            "use_redis": cls.USE_REDIS,
            "fallback_sqlite": cls.FALLBACK_TO_SQLITE
        }
        return status


class DatabaseManager:
    """Manages database connections and provides connection pooling."""
    
    def __init__(self):
        self._postgres_engine: Optional[Engine] = None
        self._postgres_session_factory: Optional[sessionmaker] = None
        self._redis_client: Optional[redis.Redis] = None
        self._peewee_db: Optional[PostgresqlDatabase] = None
        self._sqlite_db: Optional[SqliteDatabase] = None
        self._config = DatabaseConfig()
        
    def initialize_databases(self) -> Dict[str, bool]:
        """Initialize all database connections."""
        results = {}
        
        # Initialize PostgreSQL
        if self._config.USE_POSTGRES:
            results["postgres"] = self._initialize_postgres()
        
        # Initialize Redis
        if self._config.USE_REDIS:
            results["redis"] = self._initialize_redis()
        
        # Initialize SQLite fallback if needed
        if self._config.FALLBACK_TO_SQLITE and not results.get("postgres", False):
            results["sqlite"] = self._initialize_sqlite()
        
        return results
    
    def _initialize_postgres(self) -> bool:
        """Initialize PostgreSQL connection."""
        try:
            if not self._config.POSTGRES_DB or not self._config.POSTGRES_USER:
                logger.warning("PostgreSQL not configured - missing database or user")
                return False
            
            # SQLAlchemy engine
            self._postgres_engine = create_engine(
                self._config.get_postgres_url(),
                poolclass=QueuePool,
                pool_size=self._config.POSTGRES_POOL_SIZE,
                max_overflow=self._config.POSTGRES_MAX_OVERFLOW,
                pool_timeout=self._config.POSTGRES_POOL_TIMEOUT,
                pool_pre_ping=True,
                echo=os.getenv("SQL_ECHO", "false").lower() == "true"
            )
            
            # SQLAlchemy session factory
            self._postgres_session_factory = sessionmaker(
                bind=self._postgres_engine,
                autocommit=False,
                autoflush=False
            )
            
            # Peewee database (for ORM operations)
            self._peewee_db = PostgresqlDatabase(
                self._config.POSTGRES_DB,
                user=self._config.POSTGRES_USER,
                password=self._config.POSTGRES_PASSWORD,
                host=self._config.POSTGRES_HOST,
                port=self._config.POSTGRES_PORT,
                sslmode=self._config.POSTGRES_SSL_MODE
            )
            
            # Test connection
            with self._postgres_engine.connect() as conn:
                conn.execute("SELECT 1")
            
            logger.info("✅ PostgreSQL connection established")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize PostgreSQL: {e}")
            return False
    
    def _initialize_redis(self) -> bool:
        """Initialize Redis connection."""
        try:
            self._redis_client = redis.Redis(
                host=self._config.REDIS_HOST,
                port=self._config.REDIS_PORT,
                db=self._config.REDIS_DB,
                password=self._config.REDIS_PASSWORD or None,
                ssl=self._config.REDIS_SSL,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            
            # Test connection
            self._redis_client.ping()
            
            logger.info("✅ Redis connection established")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Redis: {e}")
            return False
    
    def _initialize_sqlite(self) -> bool:
        """Initialize SQLite fallback database."""
        try:
            sqlite_path = os.path.join("cache", "stock4u.db")
            os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
            
            self._sqlite_db = SqliteDatabase(sqlite_path)
            
            logger.info(f"✅ SQLite fallback database initialized: {sqlite_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize SQLite: {e}")
            return False
    
    @contextmanager
    def get_postgres_session(self):
        """Get a PostgreSQL session with automatic cleanup."""
        if not self._postgres_session_factory:
            raise RuntimeError("PostgreSQL not initialized")
        
        session = self._postgres_session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def get_redis_client(self) -> redis.Redis:
        """Get Redis client."""
        if not self._redis_client:
            raise RuntimeError("Redis not initialized")
        return self._redis_client
    
    def get_peewee_db(self) -> PostgresqlDatabase:
        """Get Peewee database instance."""
        if self._peewee_db:
            return self._peewee_db
        elif self._sqlite_db:
            return self._sqlite_db
        else:
            raise RuntimeError("No database available")
    
    def get_sqlalchemy_engine(self) -> Engine:
        """Get SQLAlchemy engine."""
        if not self._postgres_engine:
            raise RuntimeError("PostgreSQL not initialized")
        return self._postgres_engine
    
    def close_connections(self):
        """Close all database connections."""
        if self._postgres_engine:
            self._postgres_engine.dispose()
            logger.info("PostgreSQL connections closed")
        
        if self._redis_client:
            self._redis_client.close()
            logger.info("Redis connection closed")
        
        if self._peewee_db:
            self._peewee_db.close()
            logger.info("Peewee database connection closed")
        
        if self._sqlite_db:
            self._sqlite_db.close()
            logger.info("SQLite database connection closed")


# Global database manager instance
db_manager = DatabaseManager()


def initialize_databases() -> Dict[str, bool]:
    """Initialize all databases and return status."""
    return db_manager.initialize_databases()


def get_postgres_session():
    """Get PostgreSQL session context manager."""
    return db_manager.get_postgres_session()


def get_redis_client() -> redis.Redis:
    """Get Redis client."""
    return db_manager.get_redis_client()


def get_peewee_db():
    """Get Peewee database instance."""
    return db_manager.get_peewee_db()


def get_sqlalchemy_engine() -> Engine:
    """Get SQLAlchemy engine."""
    return db_manager.get_sqlalchemy_engine()


def close_database_connections():
    """Close all database connections."""
    db_manager.close_connections()
