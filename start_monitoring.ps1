#!/usr/bin/env pwsh
# Stock4U Monitoring Stack Startup Script

Write-Host "🚀 Starting Stock4U Monitoring Stack..." -ForegroundColor Green

# Check if Docker is running
try {
    docker version | Out-Null
} catch {
    Write-Host "❌ Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
    exit 1
}

# Create network if it doesn't exist
Write-Host "📡 Creating Docker network..." -ForegroundColor Yellow
docker network create stock4u-network 2>$null

# Create token file if it doesn't exist
$tokenFile = "ops/prometheus/token.txt"
if (-not (Test-Path $tokenFile)) {
    Write-Host "🔑 Creating API token file..." -ForegroundColor Yellow
    $token = Read-Host "Enter your API token (or press Enter for 'demo-token')"
    if (-not $token) { $token = "demo-token" }
    Set-Content -Path $tokenFile -Value $token -NoNewline
}

# Start monitoring services
Write-Host "🔧 Starting monitoring services..." -ForegroundColor Yellow
docker-compose -f docker-compose.monitoring.yml up -d

# Wait for services to start
Write-Host "⏳ Waiting for services to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Check service status
Write-Host "📊 Service Status:" -ForegroundColor Green
docker-compose -f docker-compose.monitoring.yml ps

Write-Host ""
Write-Host "🎉 Monitoring stack is ready!" -ForegroundColor Green
Write-Host ""
Write-Host "📈 Access your monitoring dashboards:" -ForegroundColor Cyan
Write-Host "   Grafana:     http://localhost:3000 (admin/admin123)" -ForegroundColor White
Write-Host "   Prometheus:  http://localhost:9090" -ForegroundColor White
Write-Host "   AlertManager: http://localhost:9093" -ForegroundColor White
Write-Host ""
Write-Host "🔧 To stop monitoring: docker-compose -f docker-compose.monitoring.yml down" -ForegroundColor Yellow
