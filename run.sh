#!/bin/bash
# Run script for Vera Assistant

# Ensure Ollama is running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama service..."
    ollama serve &
    sleep 2
fi

# Run the assistant
echo "Starting Vera Assistant..."
uv run multimodal-assistant
