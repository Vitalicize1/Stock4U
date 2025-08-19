# Streamlit Cloud User Persistence

## The Problem
In Streamlit Cloud, the file system and database are ephemeral - they get wiped on every app restart. This means users created through registration get deleted when the app reboots.

## The Solution
We've implemented a cloud-aware authentication system that can persist users using Streamlit Secrets.

## Quick Setup (Recommended)

### Option 1: Use Default Admin Account
The app automatically creates a default admin account:
- **Username**: `admin`
- **Password**: `stock4u2024`

This admin account persists across restarts and can be used immediately.

### Option 2: Set Up Persistent Users via Streamlit Secrets

1. **Go to your Streamlit Cloud dashboard**
2. **Click on your app settings**
3. **Go to the "Secrets" section**
4. **Add the following configuration**:

```toml
[users.admin]
password_hash = "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3"  # "hello"
role = "admin"
created_at = "2024-01-01T00:00:00"
last_login = ""
email = "admin@stock4u.com"

[users.demo]
password_hash = "ef92b778bafe771e89245b89ecbc08a44a4e166c06659911881f383d4473e94f"  # "demo123"
role = "user"
created_at = "2024-01-01T00:00:00"
last_login = ""
email = "demo@stock4u.com"
```

5. **Save the secrets**
6. **Restart your app**

Now you'll have persistent users:
- **Admin**: username `admin`, password `hello`
- **Demo User**: username `demo`, password `demo123`

## Custom Users

To add your own users, you need to generate password hashes. You can use this Python code:

```python
import hashlib

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Generate hash for your password
your_password = "your_secure_password"
password_hash = hash_password(your_password)
print(f"Password hash: {password_hash}")
```

Then add to your Streamlit secrets:

```toml
[users.your_username]
password_hash = "your_generated_hash_here"
role = "user"  # or "admin"
created_at = "2024-01-01T00:00:00"
last_login = ""
email = "your.email@example.com"
```

## Environment Variables

The system automatically detects cloud environments via the `STOCK4U_CLOUD=1` environment variable (set in `streamlit_app.py`).

## How It Works

1. **Cloud Detection**: The system checks if `STOCK4U_CLOUD=1`
2. **Load from Secrets**: In cloud mode, it loads users from Streamlit secrets
3. **Fallback to Default**: If no secrets are configured, it creates the default admin user
4. **Session Persistence**: User sessions persist for 8 hours
5. **No Database**: Cloud mode doesn't use the database to avoid ephemeral storage issues

## Benefits

- ✅ **Users persist across app restarts**
- ✅ **No database setup required**
- ✅ **Secure password storage**
- ✅ **Easy user management via Streamlit dashboard**
- ✅ **Automatic fallback to default admin**

## Troubleshooting

**Q: Users still get deleted on restart**
A: Make sure `STOCK4U_CLOUD=1` is set and users are configured in Streamlit secrets

**Q: Can't login with default admin**
A: Try username `admin` and password `stock4u2024`

**Q: How to add more users?**
A: Either use the registration form (temporary) or add them to Streamlit secrets (persistent)

**Q: Registration not working**
A: In cloud mode, registration creates temporary users. For persistent users, use Streamlit secrets.
