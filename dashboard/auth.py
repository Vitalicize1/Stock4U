"""
Dashboard Authentication Module

Provides user authentication for the Stock4U dashboard using Streamlit session state.
Supports user registration, login, password reset, and session management.
"""

import os
import hashlib
import hmac
import json
import uuid
import streamlit as st
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from pathlib import Path


class DashboardAuth:
    """Authentication manager for the Stock4U dashboard."""
    
    def __init__(self):
        """Initialize the authentication system."""
        self.users_file = Path("cache/users.json")
        self.users = self._load_users()
        self.session_timeout = timedelta(hours=8)  # 8 hour session timeout
        self.password_reset_tokens = {}  # Store reset tokens temporarily
    
    def _load_users(self) -> Dict[str, Dict[str, Any]]:
        """Load users from file or create default admin user."""
        users = {}
        
        # Try to load existing users from file
        if self.users_file.exists():
            try:
                with open(self.users_file, 'r', encoding='utf-8') as f:
                    users = json.load(f)
            except Exception as e:
                st.error(f"Error loading users: {e}")
        
        # If no users exist, create default admin
        if not users:
            admin_username = os.getenv("DASHBOARD_ADMIN_USER", "admin")
            admin_password = os.getenv("DASHBOARD_ADMIN_PASSWORD", "stock4u2024")
            
            # Hash the password
            hashed_password = self._hash_password(admin_password)
            
            users[admin_username] = {
                "password_hash": hashed_password,
                "role": "admin",
                "created_at": datetime.now().isoformat(),
                "last_login": None,
                "email": "admin@stock4u.com"
            }
            
            # Save to file
            self._save_users(users)
        
        return users
    
    def _save_users(self, users: Dict[str, Dict[str, Any]]) -> None:
        """Save users to file."""
        try:
            # Ensure cache directory exists
            self.users_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.users_file, 'w', encoding='utf-8') as f:
                json.dump(users, f, indent=2, ensure_ascii=False)
        except Exception as e:
            st.error(f"Error saving users: {e}")
    
    def _hash_password(self, password: str) -> str:
        """Hash a password using SHA-256."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password against its hash."""
        return hmac.compare_digest(self._hash_password(password), password_hash)
    
    def register_user(self, username: str, password: str, email: str) -> Dict[str, Any]:
        """Register a new user."""
        result = {"success": False, "message": ""}
        
        # Validate input
        if not username or not password or not email:
            result["message"] = "All fields are required."
            return result
        
        if len(username) < 3:
            result["message"] = "Username must be at least 3 characters long."
            return result
        
        if len(password) < 6:
            result["message"] = "Password must be at least 6 characters long."
            return result
        
        if username in self.users:
            result["message"] = "Username already exists."
            return result
        
        # Check if email is already used
        for user in self.users.values():
            if user.get("email") == email:
                result["message"] = "Email already registered."
                return result
        
        # Create new user
        hashed_password = self._hash_password(password)
        self.users[username] = {
            "password_hash": hashed_password,
            "role": "user",
            "created_at": datetime.now().isoformat(),
            "last_login": None,
            "email": email
        }
        
        # Save to file
        self._save_users(self.users)
        
        result["success"] = True
        result["message"] = "User registered successfully!"
        return result
    
    def login(self, username: str, password: str) -> bool:
        """Authenticate a user with username and password."""
        if username not in self.users:
            return False
        
        user = self.users[username]
        if not self._verify_password(password, user["password_hash"]):
            return False
        
        # Update last login
        user["last_login"] = datetime.now().isoformat()
        
        # Store user info in session state
        st.session_state["authenticated"] = True
        st.session_state["username"] = username
        st.session_state["user_role"] = user["role"]
        st.session_state["login_time"] = datetime.now().isoformat()
        
        return True
    
    def request_password_reset(self, email: str) -> Dict[str, Any]:
        """Request a password reset for a user."""
        result = {"success": False, "message": ""}
        
        # Find user by email
        user_found = None
        for username, user in self.users.items():
            if user.get("email") == email:
                user_found = username
                break
        
        if not user_found:
            result["message"] = "If the email exists, a reset link will be sent."
            return result  # Don't reveal if email exists
        
        # Generate reset token
        reset_token = str(uuid.uuid4())
        self.password_reset_tokens[reset_token] = {
            "username": user_found,
            "email": email,
            "created_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(hours=1)).isoformat()
        }
        
        result["success"] = True
        result["message"] = "Password reset link sent to your email."
        result["token"] = reset_token  # In a real app, this would be sent via email
        return result
    
    def reset_password(self, token: str, new_password: str) -> Dict[str, Any]:
        """Reset password using a valid token."""
        result = {"success": False, "message": ""}
        
        # Validate token
        if token not in self.password_reset_tokens:
            result["message"] = "Invalid or expired reset token."
            return result
        
        reset_data = self.password_reset_tokens[token]
        
        # Check if token is expired
        try:
            expires_at = datetime.fromisoformat(reset_data["expires_at"])
            if datetime.now() > expires_at:
                del self.password_reset_tokens[token]
                result["message"] = "Reset token has expired."
                return result
        except ValueError:
            result["message"] = "Invalid reset token."
            return result
        
        # Validate new password
        if len(new_password) < 6:
            result["message"] = "Password must be at least 6 characters long."
            return result
        
        # Update password
        username = reset_data["username"]
        if username in self.users:
            self.users[username]["password_hash"] = self._hash_password(new_password)
            self._save_users(self.users)
            
            # Remove used token
            del self.password_reset_tokens[token]
            
            result["success"] = True
            result["message"] = "Password reset successfully!"
        else:
            result["message"] = "User not found."
        
        return result
    
    def logout(self):
        """Log out the current user."""
        # Clear session state
        for key in ["authenticated", "username", "user_role", "login_time"]:
            if key in st.session_state:
                del st.session_state[key]
    
    def is_authenticated(self) -> bool:
        """Check if user is currently authenticated."""
        if not st.session_state.get("authenticated", False):
            return False
        
        # Check session timeout
        login_time_str = st.session_state.get("login_time")
        if not login_time_str:
            return False
        
        try:
            login_time = datetime.fromisoformat(login_time_str)
            if datetime.now() - login_time > self.session_timeout:
                self.logout()
                return False
        except ValueError:
            self.logout()
            return False
        
        return True
    
    def get_current_user(self) -> Optional[Dict[str, Any]]:
        """Get current user information."""
        if not self.is_authenticated():
            return None
        
        username = st.session_state.get("username")
        if username and username in self.users:
            user_info = self.users[username].copy()
            user_info["username"] = username
            return user_info
        
        return None
    
    def require_auth(self):
        """Decorator to require authentication for dashboard pages."""
        if not self.is_authenticated():
            st.error("Please log in to access this page.")
            st.stop()


def show_login_page() -> bool:
    """Display the login page with registration and forgot password options."""
    st.title("Stock4U Authentication")
    
    # Initialize auth
    auth = DashboardAuth()
    
    # Create tabs for login, register, and forgot password
    tab1, tab2, tab3 = st.tabs(["Login", "Register", "Forgot Password"])
    
    with tab1:
        st.subheader("Login")
        st.markdown("Enter your credentials to access the dashboard.")
        
        # Login form
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button("Login")
            
            if submitted:
                if auth.login(username, password):
                    st.success("Login successful! Redirecting...")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
    
    with tab2:
        st.subheader("Create Account")
        st.markdown("Register a new account to access the dashboard.")
        
        # Registration form
        with st.form("register_form"):
            new_username = st.text_input("Username", placeholder="Choose a username (min 3 chars)")
            new_email = st.text_input("Email", placeholder="Enter your email address")
            new_password = st.text_input("Password", type="password", placeholder="Choose a password (min 6 chars)")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
            register_submitted = st.form_submit_button("Register")
            
            if register_submitted:
                if new_password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    result = auth.register_user(new_username, new_password, new_email)
                    if result["success"]:
                        st.success(result["message"])
                        st.info("You can now login with your new account.")
                    else:
                        st.error(result["message"])
    
    with tab3:
        st.subheader("Reset Password")
        st.markdown("Enter your email to receive a password reset link.")
        
        # Forgot password form
        with st.form("forgot_password_form"):
            reset_email = st.text_input("Email", placeholder="Enter your registered email")
            reset_submitted = st.form_submit_button("Send Reset Link")
            
            if reset_submitted:
                result = auth.request_password_reset(reset_email)
                if result["success"]:
                    st.success(result["message"])
                    # In a real app, this would be sent via email
                    st.info(f"Reset token: {result['token']}")
                    st.warning("In a production environment, this token would be sent to your email.")
                else:
                    st.info(result["message"])
        
        # Password reset form (if token is provided)
        st.markdown("---")
        st.markdown("**Reset Password with Token**")
        
        with st.form("reset_password_form"):
            reset_token = st.text_input("Reset Token", placeholder="Enter the reset token from your email")
            new_password_reset = st.text_input("New Password", type="password", placeholder="Enter new password")
            confirm_password_reset = st.text_input("Confirm New Password", type="password", placeholder="Confirm new password")
            reset_password_submitted = st.form_submit_button("Reset Password")
            
            if reset_password_submitted:
                if new_password_reset != confirm_password_reset:
                    st.error("Passwords do not match.")
                else:
                    result = auth.reset_password(reset_token, new_password_reset)
                    if result["success"]:
                        st.success(result["message"])
                    else:
                        st.error(result["message"])
    
    return False


def show_logout_button():
    """Display a logout button in the sidebar."""
    if st.sidebar.button("Logout"):
        auth = DashboardAuth()
        auth.logout()
        st.rerun()


def show_user_info():
    """Display current user information in the sidebar."""
    auth = DashboardAuth()
    user = auth.get_current_user()
    
    if user:
        st.sidebar.markdown("---")
        st.sidebar.subheader("User Info")
        st.sidebar.markdown(f"**Username:** {user['username']}")
        
        if user.get("email"):
            st.sidebar.markdown(f"**Email:** {user['email']}")
        
        if user.get("last_login"):
            try:
                last_login = datetime.fromisoformat(user["last_login"])
                st.sidebar.markdown(f"**Last Login:** {last_login.strftime('%Y-%m-%d %H:%M')}")
            except ValueError:
                pass


def init_auth():
    """Initialize authentication and return auth instance."""
    return DashboardAuth()
