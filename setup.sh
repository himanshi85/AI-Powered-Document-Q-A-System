#!/bin/bash
# Quick setup script for KUSUMA RAG System

set -e

echo "=========================================="
echo "KUSUMA RAG System - Quick Setup"
echo "=========================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "✓ Virtual environment created and activated"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip setuptools wheel
pip install -e .

echo "✓ Dependencies installed"

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p data
mkdir -p tests

echo "✓ Directories created"

# Setup environment file
echo ""
echo "Setting up environment..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "⚠ .env file created - please update with your API keys"
else
    echo "✓ .env file already exists"
fi

# Run tests
echo ""
echo "Running tests..."
pytest tests/ -v --tb=short || echo "⚠ Some tests may have failed"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Update .env with your GOOGLE_API_KEY"
echo "2. Add your documents to data/ directory"
echo "3. Run: python main.py"
echo ""
echo "For more info, see:"
echo "- README.md (usage guide)"
echo "- SETUP.md (detailed setup)"
echo "- ARCHITECTURE.md (design details)"
echo ""
