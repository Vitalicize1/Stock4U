"""
Tests for dashboard authentication system.
"""

import pytest
import os
from unittest.mock import patch, MagicMock
from dashboard.auth import DashboardAuth


class TestDashboardAuth:
    """Test cases for DashboardAuth class."""
    
    def test_init_with_default_credentials(self):
        """Test initialization with default credentials."""
        # Mock the file operations
        with patch('pathlib.Path.exists', return_value=False):
            auth = DashboardAuth()
            assert "admin" in auth.users
            assert auth.users["admin"]["role"] == "admin"
    
    @patch.dict(os.environ, {
        "DASHBOARD_ADMIN_USER": "testuser",
        "DASHBOARD_ADMIN_PASSWORD": "testpass"
    })
    def test_init_with_custom_credentials(self):
        """Test initialization with custom environment variables."""
        # Mock the file operations
        with patch('pathlib.Path.exists', return_value=False):
            auth = DashboardAuth()
            assert "testuser" in auth.users
            assert "admin" not in auth.users
    
    def test_password_hashing(self):
        """Test password hashing and verification."""
        auth = DashboardAuth()
        password = "testpassword"
        hashed = auth._hash_password(password)
        
        # Should be different from original
        assert hashed != password
        
        # Should verify correctly
        assert auth._verify_password(password, hashed)
        assert not auth._verify_password("wrongpassword", hashed)
    
    def test_login_success(self):
        """Test successful login."""
        # Mock the file operations
        with patch('pathlib.Path.exists', return_value=False):
            auth = DashboardAuth()
            
            # Mock session state
            with patch('streamlit.session_state', {}):
                result = auth.login("admin", "stock4u2024")
                assert result is True
                assert auth.is_authenticated() is True
    
    def test_login_failure(self):
        """Test failed login."""
        # Mock the file operations
        with patch('pathlib.Path.exists', return_value=False):
            auth = DashboardAuth()
            
            # Mock session state
            with patch('streamlit.session_state', {}):
                result = auth.login("admin", "wrongpassword")
                assert result is False
                assert auth.is_authenticated() is False
    
    def test_logout(self):
        """Test logout functionality."""
        # Mock the file operations
        with patch('pathlib.Path.exists', return_value=False):
            auth = DashboardAuth()
            
            # Mock session state with authenticated user
            mock_session = {
                "authenticated": True,
                "username": "admin",
                "user_role": "admin",
                "login_time": "2024-12-01T12:00:00"
            }
            
            with patch('streamlit.session_state', mock_session):
                auth.logout()
                assert "authenticated" not in mock_session
                assert "username" not in mock_session
    
    def test_get_current_user(self):
        """Test getting current user information."""
        # Mock the file operations
        with patch('pathlib.Path.exists', return_value=False):
            auth = DashboardAuth()
            
            # Ensure admin user exists in auth.users
            assert "admin" in auth.users
            
            # Mock session state with authenticated user
            mock_session = {
                "authenticated": True,
                "username": "admin",
                "user_role": "admin",
                "login_time": "2024-12-01T12:00:00"
            }
            
            # Mock the is_authenticated method to return True for this test
            with patch('streamlit.session_state', mock_session):
                with patch.object(auth, 'is_authenticated', return_value=True):
                    user = auth.get_current_user()
                    assert user is not None
                    assert user["username"] == "admin"
                    assert user["role"] == "admin"
    
    def test_get_current_user_not_authenticated(self):
        """Test getting current user when not authenticated."""
        # Mock the file operations
        with patch('pathlib.Path.exists', return_value=False):
            auth = DashboardAuth()
            
            # Mock empty session state
            with patch('streamlit.session_state', {}):
                user = auth.get_current_user()
                assert user is None
    
    def test_session_timeout(self):
        """Test session timeout functionality."""
        # Mock the file operations
        with patch('pathlib.Path.exists', return_value=False):
            auth = DashboardAuth()
            
            # Mock session state with expired login time
            expired_time = "2024-01-01T00:00:00"  # Old date
            mock_session = {
                "authenticated": True,
                "username": "admin",
                "user_role": "admin",
                "login_time": expired_time
            }
            
            with patch('streamlit.session_state', mock_session):
                # Should not be authenticated due to expired session
                assert not auth.is_authenticated()
                # Session should be cleared
                assert "authenticated" not in mock_session
    
    def test_session_valid(self):
        """Test valid session functionality."""
        # Mock the file operations
        with patch('pathlib.Path.exists', return_value=False):
            auth = DashboardAuth()
            
            # Mock session state with recent login time
            from datetime import datetime, timedelta
            recent_time = (datetime.now() - timedelta(hours=1)).isoformat()
            mock_session = {
                "authenticated": True,
                "username": "admin",
                "user_role": "admin",
                "login_time": recent_time
            }
            
            with patch('streamlit.session_state', mock_session):
                # Should be authenticated with recent session
                assert auth.is_authenticated()
    
    def test_register_user_success(self):
        """Test successful user registration."""
        # Mock the file operations
        with patch('pathlib.Path.exists', return_value=False):
            auth = DashboardAuth()
            
            result = auth.register_user("testuser", "password123", "test@example.com")
            assert result["success"] is True
            assert "testuser" in auth.users
            assert auth.users["testuser"]["role"] == "user"
            assert auth.users["testuser"]["email"] == "test@example.com"
    
    def test_register_user_duplicate_username(self):
        """Test user registration with duplicate username."""
        # Mock the file operations
        with patch('pathlib.Path.exists', return_value=False):
            auth = DashboardAuth()
            
            # First registration should succeed
            result1 = auth.register_user("testuser", "password123", "test1@example.com")
            assert result1["success"] is True
            
            # Second registration with same username should fail
            result2 = auth.register_user("testuser", "password456", "test2@example.com")
            assert result2["success"] is False
            assert "already exists" in result2["message"]
    
    def test_register_user_duplicate_email(self):
        """Test user registration with duplicate email."""
        # Mock the file operations
        with patch('pathlib.Path.exists', return_value=False):
            auth = DashboardAuth()
            
            # First registration should succeed
            result1 = auth.register_user("user1", "password123", "test@example.com")
            assert result1["success"] is True
            
            # Second registration with same email should fail
            result2 = auth.register_user("user2", "password456", "test@example.com")
            assert result2["success"] is False
            assert "already registered" in result2["message"]
    
    def test_request_password_reset(self):
        """Test password reset request."""
        # Mock the file operations
        with patch('pathlib.Path.exists', return_value=False):
            auth = DashboardAuth()
            
            # Register a user first
            auth.register_user("testuser", "password123", "test@example.com")
            
            # Request password reset
            result = auth.request_password_reset("test@example.com")
            assert result["success"] is True
            assert "token" in result
            
            # Check that token is stored
            assert result["token"] in auth.password_reset_tokens
    
    def test_reset_password_success(self):
        """Test successful password reset."""
        # Mock the file operations
        with patch('pathlib.Path.exists', return_value=False):
            auth = DashboardAuth()
            
            # Register a user first
            auth.register_user("testuser", "password123", "test@example.com")
            
            # Request password reset
            reset_result = auth.request_password_reset("test@example.com")
            token = reset_result["token"]
            
            # Reset password
            result = auth.reset_password(token, "newpassword123")
            assert result["success"] is True
            
            # Verify password was changed
            assert auth.login("testuser", "newpassword123") is True
            assert auth.login("testuser", "password123") is False


if __name__ == "__main__":
    pytest.main([__file__])
