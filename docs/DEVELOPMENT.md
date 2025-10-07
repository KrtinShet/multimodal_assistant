# Development Guide

## Quick Start

### Initial Setup

```bash
# Run setup script
./setup.sh
```

### Running the Assistant

```bash
# Use the run script
./run.sh

# Or manually
PYTHONPATH=src uv run python src/multimodal_assistant/main.py
```

### Running Tests

```bash
# All tests
PYTHONPATH=src uv run pytest tests/ -v

# Specific test file
PYTHONPATH=src uv run pytest tests/test_streams.py -v

# With coverage
PYTHONPATH=src uv run pytest tests/ --cov=multimodal_assistant --cov-report=html
```

## Project Structure

```
vera-assistant/
├── src/multimodal_assistant/     # Main application code
│   ├── main.py                   # Entry point
│   ├── config/                   # Configuration
│   ├── core/                     # Core infrastructure
│   ├── engines/                  # AI model engines
│   ├── pipeline/                 # Processing pipeline
│   ├── processors/               # Stream processors
│   └── utils/                    # Utilities
├── tests/                        # Test suite
├── setup.sh                      # Setup script
├── run.sh                        # Run script
└── pyproject.toml                # Project config
```

## Architecture Overview

### Core Components

1. **Event Bus** (`core/event_bus.py`)
   - Pub/sub event system
   - Decouples components
   - Enables extensibility

2. **Async Streams** (`core/streams.py`)
   - Generic streaming abstraction
   - Backpressure handling
   - Map/filter operations

3. **State Manager** (`core/state_manager.py`)
   - Pipeline state machine
   - Session management
   - Conversation history

### Engine Layer

All engines implement common interfaces:

- **STT Engine**: Faster-Whisper (speech → text)
- **Vision Engine**: CLIP (image → embedding)
- **LLM Engine**: Ollama (text → response)
- **TTS Engine**: Kokoro (text → speech)

### Pipeline Flow

```
Audio Input → VAD → STT ──┐
                           ├─→ LLM → TTS → Audio Output
Video Input → Vision ──────┘
```

## Development Workflow

### Adding a New Feature

1. **Create Interface** (if new component type)
   ```python
   # In engines/base.py or new file
   from abc import ABC, abstractmethod

   class INewEngine(ABC):
       @abstractmethod
       async def initialize(self): pass

       @abstractmethod
       async def process(self, input): pass
   ```

2. **Implement Engine**
   ```python
   # In engines/new_engine.py
   from .base import INewEngine

   class NewEngine(INewEngine):
       async def initialize(self):
           # Load models, etc.
           pass

       async def process(self, input):
           # Process input
           pass
   ```

3. **Write Tests**
   ```python
   # In tests/test_new_engine.py
   import pytest
   from multimodal_assistant.engines.new_engine import NewEngine

   @pytest.mark.asyncio
   async def test_new_engine():
       engine = NewEngine()
       await engine.initialize()
       result = await engine.process(test_input)
       assert result is not None
   ```

4. **Update Pipeline**
   - Add to `main.py` initialization
   - Wire into `PipelineCoordinator`
   - Publish/subscribe to relevant events

### Code Style

```bash
# Format code
uv run black src/ tests/

# Check linting
uv run ruff check src/ tests/

# Type checking
uv run mypy src/
```

### Debugging

Enable detailed logging:

```python
from multimodal_assistant.utils.logger import setup_logger
import logging

logger = setup_logger()
logger.setLevel(logging.DEBUG)
```

Monitor performance:

```python
from multimodal_assistant.utils.performance import PerformanceMonitor

monitor = PerformanceMonitor()
monitor.start_timer("component_name")
# ... do work ...
latency = monitor.end_timer("component_name")
print(f"Latency: {latency}ms")
```

## Common Development Tasks

### Adding a New Model

1. Implement the engine interface
2. Update `config/settings.py`
3. Add model-specific dependencies to `pyproject.toml`
4. Update initialization in `main.py`

### Adding Event Types

1. Define event in `core/event_bus.py`
2. Publish event in producer component
3. Subscribe in consumer component

### Optimizing Performance

1. Use `PerformanceMonitor` to measure
2. Identify bottlenecks
3. Optimize:
   - Model quantization
   - Parallel processing
   - Caching
   - Batching

## Testing Strategy

### Unit Tests
- Test individual components in isolation
- Mock dependencies
- Fast execution (<1s per test)

### Integration Tests
- Test component interactions
- Use real engines (or test doubles)
- May be slower

### Performance Tests
- Measure latency targets
- Track memory usage
- Monitor CPU/GPU utilization

## Troubleshooting

### Import Errors

Always use `PYTHONPATH=src` when running:

```bash
PYTHONPATH=src uv run python src/multimodal_assistant/main.py
```

### Ollama Not Running

```bash
# Check if running
pgrep ollama

# Start manually
ollama serve &

# Check models
ollama list
```

### MPS Issues

```python
# Check MPS availability
import torch
print(torch.backends.mps.is_available())
print(torch.backends.mps.is_built())
```

### Audio Device Issues

```python
import sounddevice as sd
print(sd.query_devices())
```

## Performance Optimization Tips

1. **Lazy Loading**: Load models on first use
2. **Caching**: Cache vision embeddings for repeated frames
3. **Quantization**: Use INT8/INT4 for models
4. **Batching**: Batch similar operations
5. **Parallel Processing**: STT + Vision can run simultaneously
6. **Streaming**: Process incrementally, don't wait for complete inputs

## Release Checklist

- [ ] All tests pass
- [ ] Code formatted (black)
- [ ] Linting passes (ruff)
- [ ] Type checking passes (mypy)
- [ ] Documentation updated
- [ ] Performance targets met
- [ ] Memory usage < 8GB
- [ ] README updated

## Resources

- [Framework Spec](multimodal-ai-assistant-spec.md)
- [Python Implementation Guide](multimodal-assistant-python-impl.md)
- [Faster-Whisper Docs](https://github.com/guillaumekln/faster-whisper)
- [Ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [CLIP Model](https://github.com/openai/CLIP)
- [UV Documentation](https://docs.astral.sh/uv/)
