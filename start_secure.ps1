#!/usr/bin/env pwsh
# Secure Startup Script for Stock4U
# This script ensures proper security configuration before starting the application

param(
    [switch]$Dev,
    [switch]$GenerateSSL,
    [switch]$CheckSecurity
)

Write-Host "🔐 Stock4U Secure Startup Script" -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Green

# Function to check if .env file exists
function Test-EnvFile {
    if (-not (Test-Path ".env")) {
        Write-Host "❌ .env file not found!" -ForegroundColor Red
        Write-Host "📝 Please create a .env file with the following variables:" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "POSTGRES_PASSWORD=your_secure_password" -ForegroundColor Cyan
        Write-Host "REDIS_PASSWORD=your_secure_password" -ForegroundColor Cyan
        Write-Host "JWT_SECRET_KEY=your_very_long_random_key" -ForegroundColor Cyan
        Write-Host "API_KEY=your_secure_api_key" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "💡 You can copy from .env.example if it exists" -ForegroundColor Yellow
        exit 1
    }
}

# Function to validate environment variables
function Test-EnvironmentVariables {
    Write-Host "🔍 Checking environment variables..." -ForegroundColor Yellow
    
    $requiredVars = @(
        "POSTGRES_PASSWORD",
        "REDIS_PASSWORD", 
        "JWT_SECRET_KEY",
        "API_KEY"
    )
    
    $missingVars = @()
    
    foreach ($var in $requiredVars) {
        if ([string]::IsNullOrEmpty([Environment]::GetEnvironmentVariable($var))) {
            $missingVars += $var
        }
    }
    
    if ($missingVars.Count -gt 0) {
        Write-Host "❌ Missing required environment variables:" -ForegroundColor Red
        foreach ($var in $missingVars) {
            Write-Host "   - $var" -ForegroundColor Red
        }
        Write-Host ""
        Write-Host "💡 Make sure your .env file is loaded properly" -ForegroundColor Yellow
        exit 1
    }
    
    Write-Host "✅ All required environment variables are set" -ForegroundColor Green
}

# Function to generate SSL certificates
function New-SSLCertificates {
    Write-Host "🔐 Generating SSL certificates..." -ForegroundColor Yellow
    
    if (-not (Test-Path "ops/security/generate_ssl_cert.ps1")) {
        Write-Host "❌ SSL certificate generation script not found!" -ForegroundColor Red
        exit 1
    }
    
    & "ops/security/generate_ssl_cert.ps1"
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to generate SSL certificates" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "✅ SSL certificates generated successfully" -ForegroundColor Green
}

# Function to perform security checks
function Test-SecurityConfiguration {
    Write-Host "🔒 Performing security checks..." -ForegroundColor Yellow
    
    # Check if running as administrator (for Windows)
    $isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")
    
    if ($isAdmin) {
        Write-Host "⚠️  Running as administrator - consider using a non-privileged user" -ForegroundColor Yellow
    }
    
    # Check Docker installation
    try {
        $dockerVersion = docker --version
        Write-Host "✅ Docker is installed: $dockerVersion" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Docker is not installed or not in PATH" -ForegroundColor Red
        exit 1
    }
    
    # Check if Docker is running
    try {
        docker info | Out-Null
        Write-Host "✅ Docker daemon is running" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Docker daemon is not running" -ForegroundColor Red
        Write-Host "💡 Please start Docker Desktop or Docker daemon" -ForegroundColor Yellow
        exit 1
    }
    
    Write-Host "✅ Security checks completed" -ForegroundColor Green
}

# Function to start services
function Start-Services {
    Write-Host "🚀 Starting Stock4U services..." -ForegroundColor Yellow
    
    $composeFiles = @("docker-compose.yml")
    
    if ($Dev) {
        $composeFiles += "docker-compose.dev.yml"
        Write-Host "🔧 Starting in development mode" -ForegroundColor Cyan
    }
    
    $composeArgs = $composeFiles | ForEach-Object { "-f", $_ }
    $composeArgs += "up", "-d"
    
    if ($Dev) {
        $composeArgs += "--profile", "dev"
    }
    
    Write-Host "📋 Running: docker-compose $($composeArgs -join ' ')" -ForegroundColor Gray
    
    & docker-compose @composeArgs
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to start services" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "✅ Services started successfully" -ForegroundColor Green
}

# Function to show service status
function Show-ServiceStatus {
    Write-Host ""
    Write-Host "📊 Service Status:" -ForegroundColor Cyan
    Write-Host "=================" -ForegroundColor Cyan
    
    docker-compose ps
    
    Write-Host ""
    Write-Host "🌐 Access URLs:" -ForegroundColor Cyan
    Write-Host "==============" -ForegroundColor Cyan
    
    if ($Dev) {
        Write-Host "📈 Grafana:     http://localhost:3000" -ForegroundColor White
        Write-Host "📊 Prometheus:  http://localhost:9090" -ForegroundColor White
        Write-Host "🗄️  PgAdmin:     http://localhost:8080" -ForegroundColor White
        Write-Host "🔍 Redis Cmd:   http://localhost:8081" -ForegroundColor White
    }
    
    Write-Host "🔐 API (HTTPS): https://localhost" -ForegroundColor White
    Write-Host "🔓 API (HTTP):  http://localhost:8000" -ForegroundColor White
}

# Main execution
try {
    # Load environment variables
    if (Test-Path ".env") {
        Get-Content ".env" | ForEach-Object {
            if ($_ -match "^([^#][^=]+)=(.*)$") {
                [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
            }
        }
    }
    
    # Check environment file
    Test-EnvFile
    
    # Perform security checks if requested
    if ($CheckSecurity) {
        Test-SecurityConfiguration
        exit 0
    }
    
    # Generate SSL certificates if requested
    if ($GenerateSSL) {
        New-SSLCertificates
        exit 0
    }
    
    # Validate environment variables
    Test-EnvironmentVariables
    
    # Perform security checks
    Test-SecurityConfiguration
    
    # Generate SSL certificates if they don't exist
    if (-not (Test-Path "ops/security/ssl/cert.pem") -or -not (Test-Path "ops/security/ssl/key.pem")) {
        Write-Host "🔐 SSL certificates not found, generating..." -ForegroundColor Yellow
        New-SSLCertificates
    }
    
    # Start services
    Start-Services
    
    # Show status
    Show-ServiceStatus
    
    Write-Host ""
    Write-Host "🎉 Stock4U is now running securely!" -ForegroundColor Green
    Write-Host "💡 Use 'docker-compose logs -f' to view logs" -ForegroundColor Yellow
    Write-Host "💡 Use 'docker-compose down' to stop services" -ForegroundColor Yellow
    
}
catch {
    Write-Host "❌ Error: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
