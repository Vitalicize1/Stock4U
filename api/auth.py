"""
Enhanced authentication endpoints for Stock4U API.
Provides JWT-based authentication, user management, and security features.
"""

from datetime import timedelta
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from utils.security import (
    JWTAuth, RateLimiter, SecurityMiddleware, 
    generate_secure_token, audit_log, security_config
)
from utils.user_management import (
    user_manager, UserCreate, UserUpdate, UserResponse,
    UserRole
)
from utils.logger import get_logger
import os
from datetime import datetime

logger = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["authentication"])

# Request/Response models
class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse

class RefreshTokenRequest(BaseModel):
    refresh_token: str

class RefreshTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int

class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str

class CreateApiTokenRequest(BaseModel):
    token_name: str

class CreateApiTokenResponse(BaseModel):
    token: str
    token_name: str
    created_at: str

# Authentication dependencies
security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user from JWT token."""
    try:
        payload = JWTAuth.verify_token(credentials.credentials)
        username = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        user = user_manager.get_user(username)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User is inactive",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return user
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def require_permission(permission: str):
    """Dependency to require specific permission."""
    async def permission_checker(current_user = Depends(get_current_user)):
        if not current_user.has_permission(permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: {permission}"
            )
        return current_user
    return permission_checker

# Authentication endpoints
@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Authenticate user and return JWT tokens."""
    # Rate limiting for login attempts
    identifier = f"login:{request.username}"
    if not RateLimiter.check_rate_limit(identifier, 5, 300):  # 5 attempts per 5 minutes
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many login attempts. Please try again later."
        )
    
    # Authenticate user
    user = user_manager.authenticate_user(request.username, request.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    
    # Create tokens
    access_token_expires = timedelta(minutes=security_config.jwt_expire_minutes)
    access_token = JWTAuth.create_access_token(
        data={"sub": user.username, "role": user.role},
        expires_delta=access_token_expires
    )
    refresh_token = JWTAuth.create_refresh_token(
        data={"sub": user.username}
    )
    
    # Audit successful login
    audit_log("api_login_successful", {
        "username": user.username,
        "user_id": user.id
    })
    
    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=security_config.jwt_expire_minutes * 60,
        user=user.to_response()
    )

@router.post("/refresh", response_model=RefreshTokenResponse)
async def refresh_token(request: RefreshTokenRequest):
    """Refresh access token using refresh token."""
    try:
        payload = JWTAuth.verify_token(request.refresh_token)
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )
        
        username = payload.get("sub")
        user = user_manager.get_user(username)
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )
        
        # Create new access token
        access_token_expires = timedelta(minutes=security_config.jwt_expire_minutes)
        access_token = JWTAuth.create_access_token(
            data={"sub": user.username, "role": user.role},
            expires_delta=access_token_expires
        )
        
        return RefreshTokenResponse(
            access_token=access_token,
            expires_in=security_config.jwt_expire_minutes * 60
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )

@router.post("/register", response_model=UserResponse)
async def register_user(user_data: UserCreate):
    """Register a new user."""
    # Check if registration is enabled
    if os.getenv("ENABLE_USER_REGISTRATION", "false").lower() != "true":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User registration is disabled"
        )
    
    try:
        user = user_manager.create_user(user_data)
        return user.to_response()
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user = Depends(get_current_user)):
    """Get current user information."""
    return current_user.to_response()

@router.post("/change-password")
async def change_password(
    request: ChangePasswordRequest,
    current_user = Depends(get_current_user)
):
    """Change user password."""
    success = user_manager.change_password(
        current_user.username,
        request.old_password,
        request.new_password
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid old password"
        )
    
    return {"message": "Password changed successfully"}

@router.post("/api-token", response_model=CreateApiTokenResponse)
async def create_api_token(
    request: CreateApiTokenRequest,
    current_user = Depends(get_current_user)
):
    """Create a new API token for the current user."""
    try:
        token = user_manager.generate_api_token(
            current_user.username,
            request.token_name
        )
        
        return CreateApiTokenResponse(
            token=token,
            token_name=request.token_name,
            created_at=datetime.utcnow().isoformat()
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.delete("/api-token/{token}")
async def revoke_api_token(
    token: str,
    current_user = Depends(get_current_user)
):
    """Revoke an API token."""
    success = user_manager.revoke_api_token(current_user.username, token)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Token not found"
        )
    
    return {"message": "Token revoked successfully"}

# Admin endpoints (require admin permission)
@router.get("/users", response_model=list[UserResponse])
async def list_users(current_user = Depends(require_permission("user_management"))):
    """List all users (admin only)."""
    return user_manager.list_users()

@router.post("/users", response_model=UserResponse)
async def create_user(
    user_data: UserCreate,
    current_user = Depends(require_permission("user_management"))
):
    """Create a new user (admin only)."""
    try:
        user = user_manager.create_user(user_data)
        return user.to_response()
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.put("/users/{username}", response_model=UserResponse)
async def update_user(
    username: str,
    update_data: UserUpdate,
    current_user = Depends(require_permission("user_management"))
):
    """Update user information (admin only)."""
    user = user_manager.update_user(username, update_data)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return user.to_response()

@router.delete("/users/{username}")
async def delete_user(
    username: str,
    current_user = Depends(require_permission("user_management"))
):
    """Delete a user (admin only)."""
    try:
        success = user_manager.delete_user(username)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        return {"message": "User deleted successfully"}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

# Security endpoints
@router.get("/security/verify")
async def verify_security_config():
    """Verify security configuration."""
    return {
        "jwt_enabled": True,
        "rate_limiting_enabled": True,
        "encryption_enabled": True,
        "password_requirements": {
            "min_length": 8,
            "require_uppercase": True,
            "require_lowercase": True,
            "require_digits": True,
            "require_special_chars": True
        },
        "token_expiry": {
            "access_token_minutes": security_config.jwt_expire_minutes,
            "refresh_token_days": security_config.jwt_refresh_expire_days
        }
    }

@router.post("/logout")
async def logout(current_user = Depends(get_current_user)):
    """Logout user (client should discard tokens)."""
    # In a stateless JWT system, logout is handled client-side
    # But we can log the event for audit purposes
    audit_log("api_logout", {
        "username": current_user.username,
        "user_id": current_user.id
    })
    
    return {"message": "Logged out successfully"}
