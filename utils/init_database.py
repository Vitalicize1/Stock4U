"""
Database Initialization Script for Stock4U

This script initializes the PostgreSQL and Redis databases, creates tables,
and provides utilities for database management.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.database import initialize_databases, DatabaseConfig
from models.database_models import create_tables, drop_tables
from utils.database_cache import initialize_cache
from utils.database_logger import initialize_database_logger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_env_file():
    """Create a .env file with database configuration template."""
    env_content = """# Database Configuration
# PostgreSQL Settings
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=stock4u
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password_here
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
"""
    
    env_file = project_root / ".env"
    if not env_file.exists():
        with open(env_file, "w") as f:
            f.write(env_content)
        logger.info(f"✅ Created .env file at {env_file}")
        logger.info("📝 Please update the database credentials in .env file")
    else:
        logger.info("ℹ️ .env file already exists")


def check_database_requirements():
    """Check if required database packages are installed."""
    required_packages = [
        "psycopg2-binary",
        "redis",
        "sqlalchemy",
        "peewee"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"❌ Missing required packages: {', '.join(missing_packages)}")
        logger.info("💡 Install with: pip install " + " ".join(missing_packages))
        return False
    
    logger.info("✅ All required packages are installed")
    return True


def validate_configuration():
    """Validate database configuration."""
    config = DatabaseConfig()
    status = config.validate_config()
    
    logger.info("📊 Configuration Status:")
    for key, value in status.items():
        status_icon = "✅" if value else "❌"
        logger.info(f"  {status_icon} {key}: {value}")
    
    return status


def initialize_database_systems():
    """Initialize all database systems."""
    logger.info("🚀 Initializing database systems...")
    
    # Initialize databases
    db_status = initialize_databases()
    logger.info(f"📊 Database Status: {db_status}")
    
    # Initialize cache
    cache_status = initialize_cache()
    logger.info(f"💾 Cache Status: {cache_status}")
    
    # Initialize logger
    logger_status = initialize_database_logger()
    logger.info(f"📝 Logger Status: {logger_status}")
    
    return db_status, cache_status, logger_status


def create_database_tables():
    """Create database tables."""
    logger.info("🗄️ Creating database tables...")
    
    try:
        create_tables()
        logger.info("✅ Database tables created successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create tables: {e}")
        return False


def drop_database_tables():
    """Drop database tables (use with caution!)."""
    logger.warning("⚠️ This will delete all data in the database!")
    response = input("Are you sure you want to drop all tables? (yes/no): ")
    
    if response.lower() == "yes":
        try:
            drop_tables()
            logger.info("🗑️ Database tables dropped successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to drop tables: {e}")
            return False
    else:
        logger.info("❌ Operation cancelled")
        return False


def test_database_connections():
    """Test database connections."""
    logger.info("🔍 Testing database connections...")
    
    # Test PostgreSQL
    try:
        from utils.database import get_postgres_session
        with get_postgres_session() as session:
            result = session.execute("SELECT 1 as test")
            logger.info("✅ PostgreSQL connection successful")
    except Exception as e:
        logger.error(f"❌ PostgreSQL connection failed: {e}")
    
    # Test Redis
    try:
        from utils.database import get_redis_client
        redis_client = get_redis_client()
        redis_client.ping()
        logger.info("✅ Redis connection successful")
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")


def show_database_stats():
    """Show database statistics."""
    logger.info("📈 Database Statistics:")
    
    try:
        from utils.database_cache import get_cache_stats
        cache_stats = get_cache_stats()
        logger.info(f"  💾 Cache: {cache_stats}")
    except Exception as e:
        logger.warning(f"⚠️ Could not get cache stats: {e}")
    
    try:
        from utils.database_logger import get_accuracy_summary
        accuracy_stats = get_accuracy_summary()
        logger.info(f"  📊 Predictions: {accuracy_stats}")
    except Exception as e:
        logger.warning(f"⚠️ Could not get accuracy stats: {e}")


def main():
    """Main initialization function."""
    logger.info("🎯 Stock4U Database Initialization")
    logger.info("=" * 50)
    
    # Check requirements
    if not check_database_requirements():
        return False
    
    # Create .env file if needed
    create_env_file()
    
    # Validate configuration
    config_status = validate_configuration()
    
    # Initialize systems
    db_status, cache_status, logger_status = initialize_database_systems()
    
    # Create tables if PostgreSQL is available
    if db_status.get("postgres", False):
        create_database_tables()
        test_database_connections()
        show_database_stats()
    else:
        logger.warning("⚠️ PostgreSQL not available, using SQLite fallback")
        create_database_tables()
    
    logger.info("✅ Database initialization complete!")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Stock4U Database Initialization")
    parser.add_argument("--create-tables", action="store_true", help="Create database tables")
    parser.add_argument("--drop-tables", action="store_true", help="Drop database tables")
    parser.add_argument("--test-connections", action="store_true", help="Test database connections")
    parser.add_argument("--show-stats", action="store_true", help="Show database statistics")
    parser.add_argument("--validate-config", action="store_true", help="Validate configuration")
    
    args = parser.parse_args()
    
    if args.create_tables:
        create_database_tables()
    elif args.drop_tables:
        drop_database_tables()
    elif args.test_connections:
        test_database_connections()
    elif args.show_stats:
        show_database_stats()
    elif args.validate_config:
        validate_configuration()
    else:
        main()
