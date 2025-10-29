# BDH Web Application Startup Script
# This script sets up and runs the BDH web interface

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "🐉 Baby Dragon Hatchling Web Interface" -ForegroundColor Magenta
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
Write-Host "Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found! Please install Python 3.8 or higher." -ForegroundColor Red
    exit 1
}

Write-Host ""

# Check if requirements are installed
Write-Host "Checking dependencies..." -ForegroundColor Yellow
$requirementsFile = "requirements-web.txt"

if (Test-Path $requirementsFile) {
    Write-Host "Installing/updating dependencies..." -ForegroundColor Yellow
    pip install -r $requirementsFile
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Dependencies installed successfully" -ForegroundColor Green
    } else {
        Write-Host "✗ Failed to install dependencies" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "✗ $requirementsFile not found!" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Check if model exists
Write-Host "Checking for trained model..." -ForegroundColor Yellow
if (Test-Path "bdh_model.pt") {
    Write-Host "✓ Found trained model: bdh_model.pt" -ForegroundColor Green
} else {
    Write-Host "⚠ No trained model found." -ForegroundColor Yellow
    Write-Host "  The app will work with random initialization." -ForegroundColor Yellow
    Write-Host "  For better results, run 'python train.py' first." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting BDH Web Server..." -ForegroundColor Magenta
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Once started, open your browser to:" -ForegroundColor Green
Write-Host "  http://localhost:5000" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Run the application
python app.py
