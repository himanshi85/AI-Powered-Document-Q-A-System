#!/bin/bash

# KUSUMA UI - Quick Start Script
# This script sets up and runs the Streamlit UI

set -e  # Exit on any error

echo "🚀 Starting KUSUMA UI Server..."
echo ""

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found."
    echo "Creating .env from .env.example..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✓ .env created. Please edit it with your GOOGLE_API_KEY"
    fi
fi

# Check if dependencies are installed
python3 -c "import streamlit" 2>/dev/null || {
    echo "📦 Installing dependencies..."
    pip install -e .
}

# Start Streamlit
echo ""
echo "✨ Launching KUSUMA UI..."
echo "🌐 Open your browser at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run app.py
