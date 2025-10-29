#!/bin/bash

# BDH Web Application Startup Script for Linux/Mac
# This script sets up and runs the BDH web interface

echo "========================================"
echo "🐉 Baby Dragon Hatchling Web Interface"
echo "========================================"
echo ""

# Check if Python is installed
echo "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
    PYTHON_VERSION=$(python3 --version)
    echo "✓ Found: $PYTHON_VERSION"
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
    PYTHON_VERSION=$(python --version)
    echo "✓ Found: $PYTHON_VERSION"
else
    echo "✗ Python not found! Please install Python 3.8 or higher."
    exit 1
fi

echo ""

# Check if requirements are installed
echo "Checking dependencies..."
REQUIREMENTS_FILE="requirements-web.txt"

if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "Installing/updating dependencies..."
    $PYTHON_CMD -m pip install -r $REQUIREMENTS_FILE
    
    if [ $? -eq 0 ]; then
        echo "✓ Dependencies installed successfully"
    else
        echo "✗ Failed to install dependencies"
        exit 1
    fi
else
    echo "✗ $REQUIREMENTS_FILE not found!"
    exit 1
fi

echo ""

# Check if model exists
echo "Checking for trained model..."
if [ -f "bdh_model.pt" ]; then
    echo "✓ Found trained model: bdh_model.pt"
else
    echo "⚠ No trained model found."
    echo "  The app will work with random initialization."
    echo "  For better results, run '$PYTHON_CMD train.py' first."
fi

echo ""
echo "========================================"
echo "Starting BDH Web Server..."
echo "========================================"
echo ""
echo "Once started, open your browser to:"
echo "  http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the application
$PYTHON_CMD app.py
