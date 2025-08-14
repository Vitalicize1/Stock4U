"""
Database-Backed Cache System

This module provides a unified cache interface that uses both Redis (for fast access)
and PostgreSQL (for persistence) to store cached results and data.
"""

import json
import time
import hashlib
import logging
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta
from functools import wraps

from utils.database import get_redis_client, get_postgres_session
from models.database_models import CacheEntry, SystemMetric

logger = logging.getLogger(__name__)


class DatabaseCache:
    """Unified cache system using Redis and PostgreSQL."""
    
    def __init__(self):
        self.redis_client = None
        self._redis_available = False
        self._postgres_available = False
        
        # Initialize connections
        self._initialize_connections()
    
    def _initialize_connections(self):
        """Initialize Redis and PostgreSQL connections."""
        try:
            self.redis_client = get_redis_client()
            self._redis_available = True
            logger.info("✅ Redis cache initialized")
        except Exception as e:
            logger.warning(f"⚠️ Redis not available: {e}")
            self._redis_available = False
        
        try:
            # Test PostgreSQL connection
            with get_postgres_session() as session:
                session.execute("SELECT 1")
            self._postgres_available = True
            logger.info("✅ PostgreSQL cache initialized")
        except Exception as e:
            logger.warning(f"⚠️ PostgreSQL not available: {e}")
            self._postgres_available = False
    
    def _generate_cache_key(self, key: str) -> str:
        """Generate a standardized cache key."""
        # Hash long keys to avoid Redis key length issues
        if len(key) > 100:
            return f"hash:{hashlib.md5(key.encode()).hexdigest()}"
        return key
    
    def get(self, key: str, ttl_seconds: int = 900) -> Optional[Dict[str, Any]]:
        """
        Get cached data with fallback strategy:
        1. Try Redis first (fastest)
        2. Try PostgreSQL if Redis fails
        3. Return None if not found
        """
        cache_key = self._generate_cache_key(key)
        
        # Try Redis first
        if self._redis_available:
            try:
                cached_data = self._get_from_redis(cache_key)
                if cached_data:
                    self._record_cache_hit("redis")
                    return cached_data
            except Exception as e:
                logger.warning(f"Redis get failed: {e}")
        
        # Try PostgreSQL
        if self._postgres_available:
            try:
                cached_data = self._get_from_postgres(key, ttl_seconds)
                if cached_data:
                    # Store in Redis for future fast access
                    if self._redis_available:
                        try:
                            self._set_in_redis(cache_key, cached_data, ttl_seconds)
                        except Exception:
                            pass  # Don't fail if Redis storage fails
                    
                    self._record_cache_hit("postgres")
                    return cached_data
            except Exception as e:
                logger.warning(f"PostgreSQL get failed: {e}")
        
        self._record_cache_miss()
        return None
    
    def set(self, key: str, data: Dict[str, Any], ttl_seconds: int = 900) -> bool:
        """
        Set cached data with dual storage:
        1. Store in Redis (fast access)
        2. Store in PostgreSQL (persistence)
        """
        cache_key = self._generate_cache_key(key)
        success = False
        
        # Store in Redis
        if self._redis_available:
            try:
                self._set_in_redis(cache_key, data, ttl_seconds)
                success = True
            except Exception as e:
                logger.warning(f"Redis set failed: {e}")
        
        # Store in PostgreSQL
        if self._postgres_available:
            try:
                self._set_in_postgres(key, data, ttl_seconds)
                success = True
            except Exception as e:
                logger.warning(f"PostgreSQL set failed: {e}")
        
        if success:
            self._record_cache_set()
        
        return success
    
    def delete(self, key: str) -> bool:
        """Delete cached data from both Redis and PostgreSQL."""
        cache_key = self._generate_cache_key(key)
        success = False
        
        # Delete from Redis
        if self._redis_available:
            try:
                self.redis_client.delete(cache_key)
                success = True
            except Exception as e:
                logger.warning(f"Redis delete failed: {e}")
        
        # Delete from PostgreSQL
        if self._postgres_available:
            try:
                CacheEntry.delete().where(CacheEntry.cache_key == key).execute()
                success = True
            except Exception as e:
                logger.warning(f"PostgreSQL delete failed: {e}")
        
        return success
    
    def exists(self, key: str) -> bool:
        """Check if a key exists in cache."""
        cache_key = self._generate_cache_key(key)
        
        # Check Redis first
        if self._redis_available:
            try:
                if self.redis_client.exists(cache_key):
                    return True
            except Exception:
                pass
        
        # Check PostgreSQL
        if self._postgres_available:
            try:
                return CacheEntry.is_valid(key)
            except Exception:
                pass
        
        return False
    
    def clear(self, pattern: Optional[str] = None) -> int:
        """Clear cache entries. Returns count of cleared entries."""
        cleared_count = 0
        
        # Clear Redis
        if self._redis_available:
            try:
                if pattern:
                    keys = self.redis_client.keys(pattern)
                    if keys:
                        cleared_count += self.redis_client.delete(*keys)
                else:
                    self.redis_client.flushdb()
                    cleared_count += 1  # Count as 1 operation
            except Exception as e:
                logger.warning(f"Redis clear failed: {e}")
        
        # Clear PostgreSQL
        if self._postgres_available:
            try:
                if pattern:
                    # PostgreSQL doesn't support pattern matching like Redis
                    # This is a simplified implementation
                    cleared_count += CacheEntry.delete().where(
                        CacheEntry.cache_key.contains(pattern)
                    ).execute()
                else:
                    cleared_count += CacheEntry.delete().execute()
            except Exception as e:
                logger.warning(f"PostgreSQL clear failed: {e}")
        
        return cleared_count
    
    def cleanup_expired(self) -> int:
        """Remove expired cache entries. Returns count of removed entries."""
        removed_count = 0
        
        # Cleanup Redis (Redis handles TTL automatically)
        if self._redis_available:
            try:
                # Redis automatically expires keys, but we can trigger cleanup
                self.redis_client.execute_command("MEMORY", "PURGE")
            except Exception as e:
                logger.warning(f"Redis cleanup failed: {e}")
        
        # Cleanup PostgreSQL
        if self._postgres_available:
            try:
                removed_count = CacheEntry.cleanup_expired()
            except Exception as e:
                logger.warning(f"PostgreSQL cleanup failed: {e}")
        
        return removed_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "redis_available": self._redis_available,
            "postgres_available": self._postgres_available,
            "total_entries": 0,
            "redis_entries": 0,
            "postgres_entries": 0,
        }
        
        # Redis stats
        if self._redis_available:
            try:
                stats["redis_entries"] = self.redis_client.dbsize()
                stats["total_entries"] += stats["redis_entries"]
            except Exception:
                pass
        
        # PostgreSQL stats
        if self._postgres_available:
            try:
                stats["postgres_entries"] = CacheEntry.select().count()
                stats["total_entries"] += stats["postgres_entries"]
            except Exception:
                pass
        
        return stats
    
    def _get_from_redis(self, key: str) -> Optional[Dict[str, Any]]:
        """Get data from Redis."""
        try:
            data = self.redis_client.get(key)
            if data:
                return json.loads(data)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Redis get error: {e}")
        return None
    
    def _set_in_redis(self, key: str, data: Dict[str, Any], ttl_seconds: int) -> None:
        """Set data in Redis."""
        try:
            data_json = json.dumps(data)
            self.redis_client.setex(key, ttl_seconds, data_json)
        except Exception as e:
            logger.warning(f"Redis set error: {e}")
            raise
    
    def _get_from_postgres(self, key: str, ttl_seconds: int) -> Optional[Dict[str, Any]]:
        """Get data from PostgreSQL."""
        try:
            return CacheEntry.get_data(key)
        except Exception as e:
            logger.warning(f"PostgreSQL get error: {e}")
        return None
    
    def _set_in_postgres(self, key: str, data: Dict[str, Any], ttl_seconds: int) -> None:
        """Set data in PostgreSQL."""
        try:
            CacheEntry.set_data(key, data, ttl_seconds)
        except Exception as e:
            logger.warning(f"PostgreSQL set error: {e}")
            raise
    
    def _record_cache_hit(self, source: str):
        """Record cache hit metric."""
        try:
            SystemMetric.record_metric(
                "cache_hit",
                1.0,
                context={"source": source},
                tags={"cache_type": "hit"}
            )
        except Exception:
            pass  # Don't fail if metrics recording fails
    
    def _record_cache_miss(self):
        """Record cache miss metric."""
        try:
            SystemMetric.record_metric(
                "cache_miss",
                1.0,
                tags={"cache_type": "miss"}
            )
        except Exception:
            pass
    
    def _record_cache_set(self):
        """Record cache set metric."""
        try:
            SystemMetric.record_metric(
                "cache_set",
                1.0,
                tags={"cache_type": "set"}
            )
        except Exception:
            pass


# Global cache instance
db_cache = DatabaseCache()


# Convenience functions for backward compatibility
def get_cached_result(key: str, ttl_seconds: int = 900) -> Optional[Dict[str, Any]]:
    """Get cached result (backward compatibility)."""
    return db_cache.get(key, ttl_seconds)


def set_cached_result(key: str, data: Dict[str, Any], ttl_seconds: int = 900) -> bool:
    """Set cached result (backward compatibility)."""
    return db_cache.set(key, data, ttl_seconds)


def invalidate_cached_result(key: str) -> bool:
    """Invalidate cached result (backward compatibility)."""
    return db_cache.delete(key)


# Cache decorator for functions
def cached(ttl_seconds: int = 900, key_prefix: str = ""):
    """
    Decorator to cache function results.
    
    Args:
        ttl_seconds: Time to live in seconds
        key_prefix: Prefix for cache key
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            cache_key = f"{key_prefix}:{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            cached_result = db_cache.get(cache_key, ttl_seconds)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            db_cache.set(cache_key, result, ttl_seconds)
            
            return result
        return wrapper
    return decorator


# Cache management functions
def clear_cache(pattern: Optional[str] = None) -> int:
    """Clear cache entries."""
    return db_cache.clear(pattern)


def cleanup_expired_cache() -> int:
    """Clean up expired cache entries."""
    return db_cache.cleanup_expired()


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    return db_cache.get_stats()


def initialize_cache() -> Dict[str, bool]:
    """Initialize cache system."""
    return {
        "redis": db_cache._redis_available,
        "postgres": db_cache._postgres_available
    }
