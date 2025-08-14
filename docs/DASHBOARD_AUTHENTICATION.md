# 🔐 Dashboard Authentication Guide

## Overview

The Stock4U dashboard includes a comprehensive authentication system that protects access to the application. Users must register and log in with valid credentials before accessing any dashboard features.

## Features

- **User Registration**: Create new accounts with email verification
- **Secure Login**: Username/password authentication with SHA-256 password hashing
- **Password Reset**: Forgot password functionality with secure tokens
- **Session Management**: 8-hour session timeout with automatic logout
- **User Roles**: Support for admin and user roles
- **Persistent Storage**: User data stored in JSON file
- **Environment Configuration**: Customizable default admin credentials

## Default Credentials

- **Username**: `admin`
- **Password**: `stock4u2024`

## Environment Variables

You can customize the default admin user by setting these environment variables:

```bash
DASHBOARD_ADMIN_USER=your_admin_username
DASHBOARD_ADMIN_PASSWORD=your_secure_password
```

## User Management

### User Registration
- Users can register new accounts through the dashboard
- Username must be at least 3 characters long
- Password must be at least 6 characters long
- Email addresses must be unique
- All users start with "user" role

### Password Reset
- Users can request password reset via email
- Reset tokens expire after 1 hour
- Secure token generation using UUID
- Password validation on reset

### User Storage
- User data is stored in `cache/users.json`
- Passwords are hashed using SHA-256
- User sessions managed via Streamlit session state

## Security Features

### Password Security
- Passwords are hashed using SHA-256
- Secure comparison using `hmac.compare_digest()` to prevent timing attacks
- No plain text passwords stored in memory

### Session Security
- Session timeout after 8 hours of inactivity
- Automatic logout on session expiration
- Session state cleared on logout

### User Management
- User information stored in Streamlit session state
- Last login tracking
- Role-based access control (admin/user)

## Usage

### For Users
1. **Registration**: Click "Register" tab to create a new account
2. **Login**: Use the "Login" tab with your credentials
3. **Password Reset**: Use "Forgot Password" tab if you forget your password
4. **Dashboard Access**: Once logged in, access all dashboard features
5. **Logout**: Use the logout button in the sidebar when done

### Registration Process
1. Choose a username (minimum 3 characters)
2. Enter your email address
3. Create a password (minimum 6 characters)
4. Confirm your password
5. Click "Register"

### Password Reset Process
1. Enter your registered email address
2. Click "Send Reset Link"
3. Copy the reset token (in production, this would be emailed)
4. Enter the token and new password
5. Click "Reset Password"

### For Developers
```python
from dashboard.auth import DashboardAuth

# Initialize authentication
auth = DashboardAuth()

# Check if user is authenticated
if not auth.is_authenticated():
    # Show login page
    show_login_page()
    return

# Get current user info
user = auth.get_current_user()
print(f"Logged in as: {user['username']} ({user['role']})")
```

## Testing

Run the authentication tests:
```bash
python -m pytest tests/test_dashboard_auth.py -v
```

## Security Best Practices

1. **Change Default Credentials**: Always change the default admin password
2. **Use Strong Passwords**: Use complex passwords with mixed characters
3. **Environment Variables**: Store credentials in environment variables, not in code
4. **Regular Updates**: Keep the authentication system updated
5. **Monitor Access**: Check login logs for suspicious activity

## Troubleshooting

### Common Issues

1. **Login Fails**: Verify username and password are correct
2. **Session Expired**: Re-login after 8 hours of inactivity
3. **Environment Variables**: Ensure environment variables are set correctly

### Debug Mode

To enable debug logging, set:
```bash
STREAMLIT_LOG_LEVEL=debug
```

## Future Enhancements

- [ ] Email integration for password reset
- [ ] Multi-factor authentication (MFA)
- [ ] Database-backed user management
- [ ] User profile management
- [ ] Audit logging
- [ ] LDAP/Active Directory integration
- [ ] Social login (Google, GitHub, etc.)
- [ ] Email verification for new registrations
