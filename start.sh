#!/bin/bash

echo "🚀 Starting Neuralif AI Chatbot..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Install Playwright browsers for web scraping
echo "🌐 Installing Playwright browsers..."
playwright install chromium

# Check environment variables
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found. Please create one from .env.example"
    echo "📝 Copy .env.example to .env and fill in your API keys"
    exit 1
fi

# Start the application
echo "✅ Starting the application..."
python main.py
