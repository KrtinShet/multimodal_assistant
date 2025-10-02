#!/bin/bash
# Setup script for Vera Assistant

set -e

echo "Setting up Vera Assistant..."

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "Installing UV package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    brew install ollama
fi

# Install Python dependencies
echo "Installing Python dependencies..."
uv sync

# Install package in editable mode
echo "Installing package..."
uv pip install -e .

# Start Ollama in background if not running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama service..."
    ollama serve &
    sleep 2
fi

# Pull the LLM model
echo "Pulling gemma3:4b model..."
ollama pull gemma3:4b

echo "Setup complete!"
echo ""
echo "To run the assistant:"
echo "  ./run.sh"
echo "  or: uv run multimodal-assistant"
echo ""
echo "To run tests:"
echo "  uv run pytest tests/ -v"
