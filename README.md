# Multimodal AI Assistant (Vera)

A production-ready multimodal AI assistant optimized for Apple M4 Pro with 24GB unified memory.

## Features

- Ultra-low latency voice interaction
- Multimodal support (voice + vision)
- Streaming-first architecture
- MPS acceleration for Apple Silicon
- Event-driven architecture

## Installation

```bash
# Install dependencies
uv sync

# Install package in editable mode
uv pip install -e .
```

## Usage

```bash
# Run the assistant
uv run multimodal-assistant

# Or use Python directly
uv run python -m multimodal_assistant.main
```

## Requirements

- macOS 14+ (for MPS support)
- Python 3.12+
- Microphone access
- Ollama with gemma3:4b model

## Setup Ollama

```bash
# Start Ollama
ollama serve &

# Pull the model
ollama pull gemma3:4b
```

## Testing

```bash
uv run pytest tests/ -v
```
