#!/bin/bash

# SkyMatch Quick Start Script

echo "🌤️  SkyMatch Setup"
echo "=================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Check for CSV files
if [ ! -f "all_soundings_2024.csv" ]; then
    echo "⚠️  Warning: all_soundings_2024.csv not found!"
    echo "   Please copy your CSV files to this directory."
fi

if [ ! -f "xcontest_data.csv" ]; then
    echo "⚠️  Warning: xcontest_data.csv not found!"
    echo "   Please copy your CSV files to this directory."
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Starting SkyMatch..."
echo "Navigate to: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the app
python app.py
