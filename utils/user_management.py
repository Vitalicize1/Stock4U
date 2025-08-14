"""
User management system for Stock4U.
Handles user authentication, authorization, and role-based access control.
"""

import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
from pydantic import BaseModel, EmailStr, validator
from utils.security import (
    JWTAuth, Encryption, validate_password_strength, 
    generate_secure_token, audit_log, security_config
)
from utils.logger import get_logger

logger = get_logger(__name__)

# User roles and permissions
class UserRole:
    """User roles and their permissions."""
    
    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"
    API_USER = "api_user"
    
    # Role permissions mapping
    PERMISSIONS = {
        ADMIN: [
            "read", "write", "delete", "admin", "user_management",
            "system_config", "monitoring", "audit_logs"
        ],
        USER: [
            "read", "write", "predictions", "backtesting", "learning"
        ],
        READONLY: [
            "read", "predictions"
        ],
        API_USER: [
            "read", "predictions", "backtesting"
        ]
    }
    
    @classmethod
    def has_permission(cls, role: str, permission: str) -> bool:
        """Check if a role has a specific permission."""
        return permission in cls.PERMISSIONS.get(role, [])

# User models
class UserCreate(BaseModel):
    """Model for creating a new user."""
    username: str
    email: EmailStr
    password: str
    role: str = UserRole.USER
    full_name: Optional[str] = None
    
    @validator('username')
    def validate_username(cls, v):
        if len(v) < 3:
            raise ValueError('Username must be at least 3 characters long')
        if not v.isalnum():
            raise ValueError('Username must contain only alphanumeric characters')
        return v.lower()
    
    @validator('password')
    def validate_password(cls, v):
        validation = validate_password_strength(v)
        if not validation['valid']:
            raise ValueError(f"Password validation failed: {', '.join(validation['errors'])}")
        return v
    
    @validator('role')
    def validate_role(cls, v):
        if v not in UserRole.PERMISSIONS:
            raise ValueError(f'Invalid role. Must be one of: {list(UserRole.PERMISSIONS.keys())}')
        return v

class UserUpdate(BaseModel):
    """Model for updating user information."""
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None
    
    @validator('role')
    def validate_role(cls, v):
        if v is not None and v not in UserRole.PERMISSIONS:
            raise ValueError(f'Invalid role. Must be one of: {list(UserRole.PERMISSIONS.keys())}')
        return v

class UserResponse(BaseModel):
    """Model for user response (without sensitive data)."""
    id: str
    username: str
    email: str
    role: str
    full_name: Optional[str]
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime]
    
    class Config:
        from_attributes = True

class User:
    """User entity class."""
    
    def __init__(self, user_data: Dict[str, Any]):
        self.id = user_data['id']
        self.username = user_data['username']
        self.email = user_data['email']
        self.password_hash = user_data['password_hash']
        self.role = user_data['role']
        self.full_name = user_data.get('full_name')
        self.is_active = user_data.get('is_active', True)
        self.created_at = datetime.fromisoformat(user_data['created_at'])
        self.last_login = datetime.fromisoformat(user_data['last_login']) if user_data.get('last_login') else None
        self.api_tokens = user_data.get('api_tokens', [])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary."""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'password_hash': self.password_hash,
            'role': self.role,
            'full_name': self.full_name,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat(),
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'api_tokens': self.api_tokens
        }
    
    def to_response(self) -> UserResponse:
        """Convert to response model."""
        return UserResponse(
            id=self.id,
            username=self.username,
            email=self.email,
            role=self.role,
            full_name=self.full_name,
            is_active=self.is_active,
            created_at=self.created_at,
            last_login=self.last_login
        )
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has a specific permission."""
        return UserRole.has_permission(self.role, permission)

class UserManager:
    """User management system."""
    
    def __init__(self, users_file: str = "cache/users.json"):
        self.users_file = Path(users_file)
        self.users_file.parent.mkdir(parents=True, exist_ok=True)
        self.encryption = Encryption()
        self._load_users()
    
    def _load_users(self):
        """Load users from file."""
        if self.users_file.exists():
            try:
                with open(self.users_file, 'r') as f:
                    encrypted_data = f.read()
                    decrypted_data = self.encryption.decrypt_data(encrypted_data)
                    self.users = json.loads(decrypted_data)
            except Exception as e:
                logger.error(f"Failed to load users: {e}")
                self.users = {}
        else:
            self.users = {}
            self._create_default_admin()
    
    def _save_users(self):
        """Save users to file."""
        try:
            data = json.dumps(self.users, indent=2)
            encrypted_data = self.encryption.encrypt_data(data)
            with open(self.users_file, 'w') as f:
                f.write(encrypted_data)
        except Exception as e:
            logger.error(f"Failed to save users: {e}")
            raise
    
    def _create_default_admin(self):
        """Create default admin user if no users exist."""
        admin_password = os.getenv("ADMIN_PASSWORD", "Admin123!")
        admin_user = UserCreate(
            username="admin",
            email="admin@stock4u.com",
            password=admin_password,
            role=UserRole.ADMIN,
            full_name="System Administrator"
        )
        self.create_user(admin_user)
        logger.info("Created default admin user")
    
    def create_user(self, user_data: UserCreate) -> User:
        """Create a new user."""
        # Check if username already exists
        if user_data.username in self.users:
            raise ValueError("Username already exists")
        
        # Check if email already exists
        for user in self.users.values():
            if user['email'] == user_data.email:
                raise ValueError("Email already exists")
        
        # Create user
        user_id = generate_secure_token(16)
        password_hash = self.encryption.hash_password(user_data.password)
        
        user = {
            'id': user_id,
            'username': user_data.username,
            'email': user_data.email,
            'password_hash': password_hash,
            'role': user_data.role,
            'full_name': user_data.full_name,
            'is_active': True,
            'created_at': datetime.utcnow().isoformat(),
            'last_login': None,
            'api_tokens': []
        }
        
        self.users[user_data.username] = user
        self._save_users()
        
        # Audit log
        audit_log("user_created", {
            "username": user_data.username,
            "email": user_data.email,
            "role": user_data.role
        })
        
        logger.info(f"Created user: {user_data.username}")
        return User(user)
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate a user with username and password."""
        if username not in self.users:
            return None
        
        user_data = self.users[username]
        if not user_data['is_active']:
            return None
        
        if not self.encryption.verify_password(password, user_data['password_hash']):
            # Audit failed login
            audit_log("login_failed", {
                "username": username,
                "reason": "invalid_password"
            })
            return None
        
        # Update last login
        user_data['last_login'] = datetime.utcnow().isoformat()
        self._save_users()
        
        # Audit successful login
        audit_log("login_successful", {
            "username": username,
            "user_id": user_data['id']
        })
        
        return User(user_data)
    
    def authenticate_api_token(self, token: str) -> Optional[User]:
        """Authenticate a user with API token."""
        for username, user_data in self.users.items():
            for api_token in user_data.get('api_tokens', []):
                if api_token['token'] == token and api_token['active']:
                    return User(user_data)
        return None
    
    def generate_api_token(self, username: str, token_name: str) -> str:
        """Generate a new API token for a user."""
        if username not in self.users:
            raise ValueError("User not found")
        
        user_data = self.users[username]
        token = generate_secure_token(32)
        
        api_token = {
            'name': token_name,
            'token': token,
            'created_at': datetime.utcnow().isoformat(),
            'active': True
        }
        
        user_data['api_tokens'].append(api_token)
        self._save_users()
        
        # Audit token creation
        audit_log("api_token_created", {
            "username": username,
            "token_name": token_name
        })
        
        return token
    
    def revoke_api_token(self, username: str, token: str) -> bool:
        """Revoke an API token."""
        if username not in self.users:
            return False
        
        user_data = self.users[username]
        for api_token in user_data['api_tokens']:
            if api_token['token'] == token:
                api_token['active'] = False
                self._save_users()
                
                # Audit token revocation
                audit_log("api_token_revoked", {
                    "username": username,
                    "token_name": api_token['name']
                })
                
                return True
        
        return False
    
    def get_user(self, username: str) -> Optional[User]:
        """Get a user by username."""
        if username not in self.users:
            return None
        return User(self.users[username])
    
    def update_user(self, username: str, update_data: UserUpdate) -> Optional[User]:
        """Update user information."""
        if username not in self.users:
            return None
        
        user_data = self.users[username]
        
        if update_data.email is not None:
            user_data['email'] = update_data.email
        if update_data.full_name is not None:
            user_data['full_name'] = update_data.full_name
        if update_data.role is not None:
            user_data['role'] = update_data.role
        if update_data.is_active is not None:
            user_data['is_active'] = update_data.is_active
        
        self._save_users()
        
        # Audit user update
        audit_log("user_updated", {
            "username": username,
            "updated_fields": update_data.dict(exclude_unset=True)
        })
        
        return User(user_data)
    
    def delete_user(self, username: str) -> bool:
        """Delete a user."""
        if username not in self.users:
            return False
        
        # Don't allow deletion of the last admin
        if self.users[username]['role'] == UserRole.ADMIN:
            admin_count = sum(1 for u in self.users.values() if u['role'] == UserRole.ADMIN)
            if admin_count <= 1:
                raise ValueError("Cannot delete the last admin user")
        
        del self.users[username]
        self._save_users()
        
        # Audit user deletion
        audit_log("user_deleted", {
            "username": username
        })
        
        return True
    
    def list_users(self) -> List[UserResponse]:
        """List all users."""
        return [User(user_data).to_response() for user_data in self.users.values()]
    
    def change_password(self, username: str, old_password: str, new_password: str) -> bool:
        """Change user password."""
        if username not in self.users:
            return False
        
        user_data = self.users[username]
        
        # Verify old password
        if not self.encryption.verify_password(old_password, user_data['password_hash']):
            return False
        
        # Validate new password
        validation = validate_password_strength(new_password)
        if not validation['valid']:
            raise ValueError(f"Password validation failed: {', '.join(validation['errors'])}")
        
        # Update password
        user_data['password_hash'] = self.encryption.hash_password(new_password)
        self._save_users()
        
        # Audit password change
        audit_log("password_changed", {
            "username": username
        })
        
        return True

# Global user manager instance
user_manager = UserManager()
