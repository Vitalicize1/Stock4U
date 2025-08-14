"""
Security utilities for Stock4U production deployment.
Includes JWT authentication, encryption, rate limiting, and security middleware.
"""

import os
import time
import hashlib
import secrets
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Union
from functools import wraps
from fastapi import HTTPException, Depends, Header, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import bcrypt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Security configuration
class SecurityConfig:
    """Security configuration class."""
    
    def __init__(self):
        self.jwt_secret = os.getenv("JWT_SECRET_KEY", self._generate_secret_key())
        self.jwt_algorithm = "HS256"
        self.jwt_expire_minutes = int(os.getenv("JWT_EXPIRE_MINUTES", "30"))
        self.jwt_refresh_expire_days = int(os.getenv("JWT_REFRESH_EXPIRE_DAYS", "7"))
        
        # Rate limiting
        self.rate_limit_per_minute = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))
        self.rate_limit_burst = int(os.getenv("RATE_LIMIT_BURST", "10"))
        
        # Encryption
        self.encryption_key = os.getenv("ENCRYPTION_KEY", self._generate_encryption_key())
        
        # API token requirements
        self.api_token_min_length = int(os.getenv("API_TOKEN_MIN_LENGTH", "32"))
        self.api_token_rotation_days = int(os.getenv("API_TOKEN_ROTATION_DAYS", "90"))
    
    def _generate_secret_key(self) -> str:
        """Generate a secure secret key for JWT."""
        return secrets.token_urlsafe(32)
    
    def _generate_encryption_key(self) -> str:
        """Generate a secure encryption key."""
        return base64.urlsafe_b64encode(Fernet.generate_key()).decode()

# Global security config
security_config = SecurityConfig()

# Rate limiting storage
_rate_limit_store: Dict[str, list] = {}

class JWTAuth:
    """JWT authentication handler."""
    
    @staticmethod
    def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=security_config.jwt_expire_minutes)
        
        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(to_encode, security_config.jwt_secret, algorithm=security_config.jwt_algorithm)
        return encoded_jwt
    
    @staticmethod
    def create_refresh_token(data: Dict[str, Any]) -> str:
        """Create a JWT refresh token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=security_config.jwt_refresh_expire_days)
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, security_config.jwt_secret, algorithm=security_config.jwt_algorithm)
        return encoded_jwt
    
    @staticmethod
    def verify_token(token: str) -> Dict[str, Any]:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(token, security_config.jwt_secret, algorithms=[security_config.jwt_algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")

class Encryption:
    """Data encryption utilities."""
    
    def __init__(self):
        self.cipher_suite = Fernet(security_config.encryption_key.encode())
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        return self.cipher_suite.encrypt(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        return self.cipher_suite.decrypt(encrypted_data.encode()).decode()
    
    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt."""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode(), salt).decode()
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify a password against its hash."""
        return bcrypt.checkpw(password.encode(), hashed.encode())

class RateLimiter:
    """Rate limiting implementation."""
    
    @staticmethod
    def check_rate_limit(identifier: str, limit: int, window: int = 60) -> bool:
        """Check if request is within rate limit."""
        now = time.time()
        window_start = now - window
        
        # Clean old entries
        if identifier in _rate_limit_store:
            _rate_limit_store[identifier] = [t for t in _rate_limit_store[identifier] if t >= window_start]
        else:
            _rate_limit_store[identifier] = []
        
        # Check limit
        if len(_rate_limit_store[identifier]) >= limit:
            return False
        
        # Add current request
        _rate_limit_store[identifier].append(now)
        return True
    
    @staticmethod
    def get_identifier(request: Request, token: Optional[str] = None) -> str:
        """Get rate limiting identifier."""
        if token:
            return f"token:{hashlib.sha256(token.encode()).hexdigest()[:16]}"
        
        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"

class SecurityMiddleware:
    """Security middleware for FastAPI."""
    
    @staticmethod
    def validate_api_token(token: str) -> bool:
        """Validate API token strength and format."""
        if len(token) < security_config.api_token_min_length:
            return False
        
        # Check for special characters
        if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in token):
            return False
        
        return True
    
    @staticmethod
    def sanitize_input(data: str) -> str:
        """Sanitize user input to prevent injection attacks."""
        # Basic XSS prevention
        dangerous_chars = ['<', '>', '"', "'", '&']
        for char in dangerous_chars:
            data = data.replace(char, f'&#{ord(char)};')
        return data
    
    @staticmethod
    def mask_sensitive_data(data: str, data_type: str = "email") -> str:
        """Mask sensitive data for logging."""
        if data_type == "email":
            if "@" in data:
                username, domain = data.split("@")
                return f"{username[:2]}***@{domain}"
        elif data_type == "ip":
            parts = data.split(".")
            return f"{parts[0]}.{parts[1]}.***.***"
        elif data_type == "token":
            return f"{data[:8]}...{data[-4:]}"
        
        return "***MASKED***"

# Security decorators and dependencies
def require_auth(func):
    """Decorator to require JWT authentication."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # This will be implemented with FastAPI dependencies
        return await func(*args, **kwargs)
    return wrapper

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())) -> Dict[str, Any]:
    """Get current authenticated user from JWT token."""
    try:
        payload = JWTAuth.verify_token(credentials.credentials)
        return payload
    except HTTPException:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def enhanced_auth_guard(
    request: Request,
    authorization: Optional[str] = Header(None),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> None:
    """Enhanced authentication guard with rate limiting and security checks."""
    
    # Rate limiting
    identifier = RateLimiter.get_identifier(request, authorization)
    if not RateLimiter.check_rate_limit(identifier, security_config.rate_limit_per_minute):
        logger.warning(f"Rate limit exceeded for {identifier}")
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # Log security event
    logger.info(f"API access: {request.method} {request.url.path} from {identifier}")
    
    # Additional security checks can be added here
    # - IP whitelisting
    # - User agent validation
    # - Request signature verification
    
    return current_user

# Utility functions
def generate_secure_token(length: int = 32) -> str:
    """Generate a secure random token."""
    return secrets.token_urlsafe(length)

def validate_password_strength(password: str) -> Dict[str, Any]:
    """Validate password strength."""
    errors = []
    
    if len(password) < 8:
        errors.append("Password must be at least 8 characters long")
    
    if not any(c.isupper() for c in password):
        errors.append("Password must contain at least one uppercase letter")
    
    if not any(c.islower() for c in password):
        errors.append("Password must contain at least one lowercase letter")
    
    if not any(c.isdigit() for c in password):
        errors.append("Password must contain at least one digit")
    
    if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
        errors.append("Password must contain at least one special character")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "strength": "strong" if len(errors) == 0 else "weak"
    }

def audit_log(event_type: str, details: Dict[str, Any], user_id: Optional[str] = None):
    """Log security audit events."""
    audit_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": event_type,
        "user_id": user_id,
        "details": details,
        "ip_address": "***MASKED***"  # Mask IP for privacy
    }
    
    logger.info(f"AUDIT: {audit_data}")

# Initialize encryption
encryption = Encryption()
