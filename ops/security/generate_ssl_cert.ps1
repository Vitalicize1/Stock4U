#!/usr/bin/env pwsh
# Generate SSL certificates for Stock4U development/testing

Write-Host "🔐 Generating SSL certificates for Stock4U..." -ForegroundColor Green

# Create SSL directory
$sslDir = "ops/security/ssl"
if (-not (Test-Path $sslDir)) {
    New-Item -ItemType Directory -Path $sslDir -Force | Out-Null
}

# Generate private key
Write-Host "📝 Generating private key..." -ForegroundColor Yellow
$keyPath = "$sslDir/key.pem"
openssl genrsa -out $keyPath 2048

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to generate private key" -ForegroundColor Red
    exit 1
}

# Generate certificate signing request
Write-Host "📝 Generating certificate signing request..." -ForegroundColor Yellow
$csrPath = "$sslDir/cert.csr"
openssl req -new -key $keyPath -out $csrPath -subj "/C=US/ST=State/L=City/O=Stock4U/OU=Development/CN=localhost"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to generate CSR" -ForegroundColor Red
    exit 1
}

# Generate self-signed certificate
Write-Host "📝 Generating self-signed certificate..." -ForegroundColor Yellow
$certPath = "$sslDir/cert.pem"
openssl x509 -req -in $csrPath -signkey $keyPath -out $certPath -days 365

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to generate certificate" -ForegroundColor Red
    exit 1
}

# Clean up CSR
Remove-Item $csrPath -Force

# Set proper permissions (Windows)
Write-Host "🔒 Setting file permissions..." -ForegroundColor Yellow
$acl = Get-Acl $keyPath
$acl.SetAccessRuleProtection($true, $false)
$rule = New-Object System.Security.AccessControl.FileSystemAccessRule("Administrators","FullControl","Allow")
$acl.AddAccessRule($rule)
Set-Acl $keyPath $acl

Write-Host "✅ SSL certificates generated successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "📁 Certificate files:" -ForegroundColor Cyan
Write-Host "   Private Key: $keyPath" -ForegroundColor White
Write-Host "   Certificate: $certPath" -ForegroundColor White
Write-Host ""
Write-Host "⚠️  Note: These are self-signed certificates for development only." -ForegroundColor Yellow
Write-Host "   For production, use certificates from a trusted CA." -ForegroundColor Yellow
