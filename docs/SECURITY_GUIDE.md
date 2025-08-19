# Stock4U Security Guide

This document outlines the security measures implemented in the Stock4U Docker setup and provides guidance for secure deployment.

## 🔐 Security Overview

Stock4U implements a multi-layered security approach with the following components:

- **Container Security**: Non-root users, read-only filesystems, security options
- **Network Security**: SSL/TLS encryption, rate limiting, IP restrictions
- **Authentication**: JWT tokens, API keys, secure password policies
- **Monitoring**: Security event logging, audit trails
- **Data Protection**: Encryption, secure storage, PII protection

## 🚀 Quick Start

### 1. Environment Setup

Create a `.env` file with secure credentials:

```bash
# Database Configuration
POSTGRES_PASSWORD=your_secure_postgres_password_here
POSTGRES_USER=stock4u_user

# Redis Configuration
REDIS_PASSWORD=your_secure_redis_password_here

# JWT Configuration
JWT_SECRET_KEY=your_very_long_random_jwt_secret_key_here_min_32_chars

# API Configuration
API_KEY=your_secure_api_key_here_min_32_chars

# PgAdmin Configuration (Development only)
PGADMIN_EMAIL=admin@stock4u.com
PGADMIN_PASSWORD=your_secure_pgadmin_password_here
```

### 2. Generate SSL Certificates

```powershell
# Generate self-signed certificates for development
.\start_secure.ps1 -GenerateSSL

# Or manually
.\ops\security\generate_ssl_cert.ps1
```

### 3. Start Services

```powershell
# Production mode
.\start_secure.ps1

# Development mode (includes monitoring)
.\start_secure.ps1 -Dev

# Security check only
.\start_secure.ps1 -CheckSecurity
```

## 🛡️ Security Features

### Container Security

#### Dockerfile Security
- **Non-root user**: Application runs as `stock4u` user instead of root
- **Security updates**: Base image updated with latest security patches
- **Minimal packages**: Only essential packages installed
- **Read-only filesystem**: Application filesystem mounted as read-only
- **Temporary filesystems**: `/tmp` and `/var/tmp` mounted as tmpfs
- **Security options**: `no-new-privileges` prevents privilege escalation

#### Docker Compose Security
- **Network isolation**: Custom bridge network with defined subnet
- **Port binding**: Services bound to localhost where possible
- **Health checks**: All services have health checks
- **Resource limits**: Memory and CPU limits configured
- **Restart policies**: Appropriate restart policies for each service

### Network Security

#### SSL/TLS Configuration
- **Strong ciphers**: ECDHE-RSA-AES256-GCM-SHA512 and similar
- **TLS 1.2+**: Only modern TLS versions supported
- **Certificate management**: Self-signed for dev, CA-signed for production
- **HSTS**: HTTP Strict Transport Security enabled
- **Perfect Forward Secrecy**: ECDHE key exchange

#### Nginx Security Headers
```nginx
# Security Headers
add_header X-Frame-Options "SAMEORIGIN" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;
add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' data:; connect-src 'self' https:; frame-ancestors 'self';" always;
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
add_header Permissions-Policy "geolocation=(), microphone=(), camera=()" always;
```

#### Rate Limiting
- **API endpoints**: 10 requests/second with burst of 20
- **Login endpoints**: 5 requests/minute with burst of 3
- **IP-based limiting**: Rate limits applied per IP address

### Authentication & Authorization

#### JWT Configuration
- **Algorithm**: HS256 (HMAC with SHA-256)
- **Token expiration**: 30 minutes for access tokens
- **Refresh tokens**: 7 days with rotation
- **Secret key**: Minimum 32 characters, stored in environment

#### API Key Security
- **Minimum length**: 32 characters
- **Special characters**: Required for complexity
- **Rotation**: 90-day rotation policy
- **Storage**: Environment variables only

### Database Security

#### PostgreSQL Security
- **Custom user**: `stock4u_user` instead of default `postgres`
- **Strong passwords**: Environment variable-based passwords
- **Network access**: Bound to localhost only
- **SSL connections**: Enforced for all connections
- **Connection limits**: Configured connection pooling

#### Redis Security
- **Authentication**: Password-protected Redis instance
- **Network binding**: Bound to localhost only
- **Memory limits**: 256MB with LRU eviction
- **Protected mode**: Enabled for additional security

### Monitoring Security

#### Prometheus Security
- **Authentication**: Required for metrics access
- **Network restrictions**: Only internal network access
- **Scrape intervals**: 15-second intervals with 10-second timeout
- **Token-based auth**: Bearer token authentication

#### Grafana Security
- **Admin password**: Minimum 12 characters
- **Session timeout**: 8-hour sessions
- **Signup disabled**: Manual user creation only
- **Email verification**: Required for new accounts
- **Gravatar disabled**: Privacy protection

## 🔍 Security Monitoring

### Audit Logging
- **Log level**: INFO and above
- **Format**: JSON structured logging
- **Destination**: File-based logging
- **Retention**: 90 days for logs, 365 days for metrics

### Security Events Tracked
- **Failed logins**: All authentication failures
- **API access**: All API endpoint access
- **Configuration changes**: System configuration modifications
- **Data access**: Sensitive data access patterns

### Alerting
- **Failed authentication**: Multiple failed login attempts
- **Rate limit violations**: Excessive API usage
- **System health**: Service availability issues
- **Security incidents**: Unusual access patterns

## 🚨 Security Best Practices

### Password Security
1. **Use strong passwords**: Minimum 12 characters with complexity
2. **Unique passwords**: Different passwords for each service
3. **Regular rotation**: Change passwords every 90 days
4. **Secure storage**: Use environment variables, never hardcode

### SSL/TLS Best Practices
1. **Use CA-signed certificates**: Self-signed only for development
2. **Regular renewal**: Monitor certificate expiration dates
3. **Strong ciphers**: Use only modern, secure cipher suites
4. **HSTS**: Enable HTTP Strict Transport Security

### Network Security
1. **Firewall rules**: Restrict access to necessary ports only
2. **VPN access**: Use VPN for remote access
3. **IP whitelisting**: Restrict access to known IP ranges
4. **Regular updates**: Keep all software updated

### Container Security
1. **Image scanning**: Scan images for vulnerabilities
2. **Base image updates**: Regularly update base images
3. **Resource limits**: Set appropriate CPU and memory limits
4. **Security scanning**: Use tools like Trivy or Snyk

## 🔧 Development vs Production

### Development Environment
- **Self-signed certificates**: Acceptable for local development
- **Debug mode**: Enabled for troubleshooting
- **Admin tools**: PgAdmin and Redis Commander available
- **Localhost binding**: Services bound to localhost only

### Production Environment
- **CA-signed certificates**: Required for production
- **Debug disabled**: Security through obscurity
- **Admin tools disabled**: No development tools in production
- **Load balancer**: Use proper load balancer configuration
- **Backup strategy**: Implement regular backups
- **Monitoring**: Full monitoring and alerting setup

## 🛠️ Troubleshooting

### Common Security Issues

#### SSL Certificate Errors
```bash
# Regenerate certificates
.\start_secure.ps1 -GenerateSSL

# Check certificate validity
openssl x509 -in ops/security/ssl/cert.pem -text -noout
```

#### Authentication Failures
```bash
# Check JWT configuration
echo $JWT_SECRET_KEY

# Verify API key
echo $API_KEY

# Check Redis authentication
redis-cli -a $REDIS_PASSWORD ping
```

#### Network Access Issues
```bash
# Check service status
docker-compose -f ops/docker-compose.yml ps

# View service logs
docker-compose -f ops/docker-compose.yml logs [service-name]

# Test network connectivity
docker-compose -f ops/docker-compose.yml exec stock4u-api curl -f http://localhost:8000/health
```

### Security Checklist

Before deploying to production:

- [ ] All passwords are strong and unique
- [ ] SSL certificates are CA-signed
- [ ] Environment variables are properly set
- [ ] Firewall rules are configured
- [ ] Monitoring is enabled
- [ ] Backup strategy is implemented
- [ ] Security scanning is completed
- [ ] Access controls are tested
- [ ] Rate limiting is configured
- [ ] Audit logging is enabled

## 📞 Security Contacts

For security issues or questions:

1. **Security Issues**: Create a private issue with security label
2. **Configuration Help**: Check the troubleshooting section
3. **Best Practices**: Review this security guide
4. **Updates**: Monitor for security updates and patches

## 📚 Additional Resources

- [Docker Security Best Practices](https://docs.docker.com/engine/security/)
- [Nginx Security Headers](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers)
- [OWASP Security Guidelines](https://owasp.org/www-project-top-ten/)
- [PostgreSQL Security](https://www.postgresql.org/docs/current/security.html)
- [Redis Security](https://redis.io/topics/security)
