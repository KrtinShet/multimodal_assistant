# Agent Tests

Comprehensive test suite for the modular LangGraph agent.

## Test Structure

```
tests/
├── conftest.py              # Pytest fixtures and configuration
├── test_memory.py           # Memory backend tests
├── test_checkpointers.py    # Checkpointer backend tests
├── test_llms.py             # LLM provider tests
├── test_graph.py            # Graph builder tests
└── test_agent.py            # End-to-end agent tests
```

## Running Tests

### Run All Tests

```bash
pytest tests/
```

### Run Specific Test File

```bash
pytest tests/test_agent.py
```

### Run Specific Test Class

```bash
pytest tests/test_agent.py::TestAgentInitialization
```

### Run Specific Test Function

```bash
pytest tests/test_agent.py::TestAgentInitialization::test_initialize_with_settings
```

### Run with Verbose Output

```bash
pytest tests/ -v
```

### Run with Coverage

```bash
pytest tests/ --cov=src/assistant --cov-report=html
```

## Test Categories

### Unit Tests - Memory Backends

Tests for swappable memory implementations:
- `TestInMemory` - In-memory storage tests
- `TestChromaMemory` - ChromaDB persistence tests

**Key test areas:**
- Initialization and configuration
- Storing and retrieving memories
- Semantic search capabilities
- User isolation
- Shutdown and cleanup

### Unit Tests - Checkpointers

Tests for conversation state persistence:
- `TestMemoryCheckpointer` - In-memory checkpointer tests
- `TestSqliteCheckpointer` - SQLite persistence tests

**Key test areas:**
- Initialization and database setup
- State persistence across sessions
- History retrieval
- Shutdown and cleanup

### Unit Tests - LLM Providers

Tests for language model integrations:
- `TestOllamaLLM` - Ollama provider tests
- `TestOpenAILLM` - OpenAI provider tests

**Key test areas:**
- Initialization with various configurations
- Mock-based invocation tests
- Streaming capabilities
- Error handling
- Shutdown and cleanup

### Unit Tests - Graph Builder

Tests for graph construction:
- `TestAgentGraphBuilder` - Graph building and customization tests

**Key test areas:**
- Building graphs with/without memory
- Adding custom nodes and edges
- Method chaining
- Component integration

### Integration Tests - Agent

End-to-end agent tests:
- `TestAgentInitialization` - Agent setup tests
- `TestAgentBackendSelection` - Backend selection tests
- `TestAgentConversation` - Conversation capability tests
- `TestAgentMemory` - Memory integration tests
- `TestAgentConversationHistory` - History retrieval tests
- `TestAgentShutdown` - Cleanup tests
- `TestAgentErrorHandling` - Error handling tests

**Key test areas:**
- Configuration-based initialization
- Dependency injection
- Backend swapping
- Conversation flow
- Memory integration
- Error handling

## Test Fixtures

### Core Fixtures

- `test_settings` - Settings with in-memory backends and temporary directories
- `chroma_settings` - Settings configured for ChromaDB testing
- `sqlite_settings` - Settings configured for SQLite testing

### Data Fixtures

- `mock_llm_responses` - Sample LLM responses
- `sample_memories` - Sample memory data
- `sample_conversation` - Sample conversation messages

## Mocking Strategy

Tests use mocking to avoid external dependencies:

1. **LLM Responses** - Mock `ainvoke` and `astream` methods
2. **Memory Operations** - Use in-memory backends for speed
3. **Checkpointers** - Use in-memory savers for most tests

## Best Practices

1. **Isolation** - Each test is independent and uses fresh fixtures
2. **Cleanup** - Temporary directories are automatically cleaned up
3. **Async** - All async tests use `@pytest.mark.asyncio`
4. **Mocking** - External services are mocked to ensure fast, reliable tests
5. **Coverage** - Tests cover happy paths, edge cases, and error conditions

## Test Markers

Mark tests with categories:

```python
@pytest.mark.unit
@pytest.mark.asyncio
async def test_something():
    pass
```

Available markers:
- `unit` - Unit tests for individual components
- `integration` - Integration tests
- `e2e` - End-to-end tests
- `slow` - Long-running tests
- `requires_ollama` - Requires Ollama running
- `requires_openai` - Requires OpenAI API key

Run tests by marker:

```bash
pytest tests/ -m unit
pytest tests/ -m "not slow"
```

## Adding New Tests

When adding new components, follow this pattern:

1. **Create test file** - `test_<component>.py`
2. **Import component** - Import the class/function to test
3. **Create test class** - `TestComponentName`
4. **Write tests** - Cover initialization, main functionality, edge cases, errors
5. **Add fixtures** - Add to `conftest.py` if needed by multiple tests
6. **Document** - Add to this README

Example:

```python
"""Tests for new component."""

import pytest
from assistant.agent.new_component import NewComponent

class TestNewComponent:
    """Tests for NewComponent."""

    @pytest.mark.asyncio
    async def test_initialize(self, test_settings):
        """Test initialization."""
        component = NewComponent(test_settings)
        await component.initialize()

        assert component.is_ready is True

    @pytest.mark.asyncio
    async def test_main_functionality(self, test_settings):
        """Test main functionality."""
        component = NewComponent(test_settings)
        await component.initialize()

        result = await component.do_something()
        assert result is not None
```

## Continuous Integration

Tests should be run in CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Run tests
  run: pytest tests/ --cov=src/assistant --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
```

## Troubleshooting

### Tests Fail with "Module not found"

Make sure you're running from the project root:
```bash
cd /path/to/vera-assistant
pytest tests/
```

### ChromaDB Tests Fail

ChromaDB tests require the `chromadb` package. Install dev dependencies:
```bash
uv sync --group dev
# or
pip install -e ".[dev]"
```

### Async Tests Not Running

Ensure `pytest-asyncio` is installed:
```bash
pip install pytest-asyncio
```

### Temporary Directories Not Cleaned

Fixtures handle cleanup automatically. If issues persist, manually clean:
```bash
rm -rf /tmp/test_*
```

## Future Enhancements

- [ ] Add performance benchmarks
- [ ] Add integration tests with real Ollama
- [ ] Add tests for custom tools
- [ ] Add tests for human-in-the-loop workflows
- [ ] Add mutation testing
- [ ] Add property-based tests with Hypothesis
