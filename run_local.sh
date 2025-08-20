#!/bin/bash

# Data Science Assessment Tool - Local Runner
# This script sets up and runs the Streamlit application locally

echo "🚀 Starting Data Science Assessment Tool..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Run the Streamlit application
echo "🎯 Starting Streamlit application..."
echo "📱 The app will open in your browser at: http://localhost:8501"
echo "🛑 Press Ctrl+C to stop the application"
echo ""

streamlit run streamlit_app_production.py

