# Testing Setup Guide with uv

Complete guide to setting up and running tests for the vera-assistant agent using `uv`.

## Quick Start

```bash
# 1. Sync dependencies including dev group
uv sync --group dev

# 2. Run all tests
uv run pytest tests/

# 3. Run with coverage
uv run pytest tests/ --cov=src/assistant --cov-report=html
```

## Initial Setup

### 1. Install uv (if not already installed)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with Homebrew
brew install uv
```

### 2. Sync All Dependencies

```bash
# Sync main dependencies + dev dependencies
uv sync --group dev

# This will install:
# - All project dependencies (LangChain, ChromaDB, etc.)
# - Dev dependencies (pytest, black, ruff, mypy)
# - Create/update virtual environment
```

### 3. Verify Installation

```bash
# Check pytest is available
uv run pytest --version

# Should output: pytest 8.4.2 (or higher)
```

## Running Tests

### Basic Commands

```bash
# Run all tests with uv
uv run pytest tests/

# Run with verbose output
uv run pytest tests/ -v

# Run with extra verbose output (show test names)
uv run pytest tests/ -vv

# Run quietly (only show summary)
uv run pytest tests/ -q
```

### Run Specific Tests

```bash
# Run single test file
uv run pytest tests/test_agent.py

# Run specific test class
uv run pytest tests/test_agent.py::TestAgentInitialization

# Run specific test function
uv run pytest tests/test_agent.py::TestAgentInitialization::test_initialize_with_settings

# Run tests matching pattern
uv run pytest tests/ -k "initialize"
```

### Coverage Reports

```bash
# Generate HTML coverage report
uv run pytest tests/ --cov=src/assistant --cov-report=html

# Open coverage report (macOS)
open htmlcov/index.html

# Terminal coverage report
uv run pytest tests/ --cov=src/assistant --cov-report=term-missing

# Multiple report formats
uv run pytest tests/ \
    --cov=src/assistant \
    --cov-report=html \
    --cov-report=term \
    --cov-report=xml
```

### Watch Mode (Development)

```bash
# Install pytest-watch (optional)
uv add --dev pytest-watch

# Run tests on file changes
uv run ptw tests/ src/
```

## Test Output Options

### Show Print Statements

```bash
# Show print() and logging output
uv run pytest tests/ -s

# Show output only for failed tests
uv run pytest tests/ --tb=short
```

### Show Local Variables

```bash
# Show local variables in tracebacks
uv run pytest tests/ -l
```

### Stop on First Failure

```bash
# Stop after first failure
uv run pytest tests/ -x

# Stop after N failures
uv run pytest tests/ --maxfail=3
```

## Running by Category

### Run with Markers (after marking tests)

```bash
# Run only unit tests
uv run pytest tests/ -m unit

# Run only integration tests
uv run pytest tests/ -m integration

# Exclude slow tests
uv run pytest tests/ -m "not slow"

# Run tests that don't require external services
uv run pytest tests/ -m "not requires_ollama"
```

## Parallel Execution

```bash
# Install pytest-xdist
uv add --dev pytest-xdist

# Run tests in parallel (auto-detect CPU cores)
uv run pytest tests/ -n auto

# Run with specific number of workers
uv run pytest tests/ -n 4
```

## Common Workflows

### Pre-Commit Testing

```bash
# Quick test run before committing
uv run pytest tests/ -x --ff

# Options:
#   -x: stop on first failure
#   --ff: run failures first
```

### CI/CD Testing

```bash
# Full test suite with coverage
uv run pytest tests/ \
    --cov=src/assistant \
    --cov-report=xml \
    --cov-report=term \
    --junitxml=test-results.xml \
    -v
```

### Debug Mode

```bash
# Run with Python debugger on failures
uv run pytest tests/ --pdb

# Drop into debugger on first failure
uv run pytest tests/ -x --pdb
```

## Project-Specific Tests

### Memory Backend Tests

```bash
# Test in-memory backend (fast)
uv run pytest tests/test_memory.py::TestInMemory

# Test ChromaDB backend (requires chromadb)
uv run pytest tests/test_memory.py::TestChromaMemory -v
```

### Checkpointer Tests

```bash
# Test in-memory checkpointer
uv run pytest tests/test_checkpointers.py::TestMemoryCheckpointer

# Test SQLite checkpointer
uv run pytest tests/test_checkpointers.py::TestSqliteCheckpointer
```

### LLM Provider Tests

```bash
# Test Ollama LLM (mocked)
uv run pytest tests/test_llms.py::TestOllamaLLM

# Test OpenAI LLM (mocked)
uv run pytest tests/test_llms.py::TestOpenAILLM
```

### End-to-End Agent Tests

```bash
# Run all agent integration tests
uv run pytest tests/test_agent.py -v

# Test specific workflow
uv run pytest tests/test_agent.py::TestAgentConversation
```

## Updating Dependencies

```bash
# Update all dependencies
uv sync --upgrade

# Update only dev dependencies
uv sync --group dev --upgrade

# Update specific package
uv add --dev "pytest>=8.5.0"
```

## Troubleshooting

### Tests Not Found

```bash
# Check pytest can discover tests
uv run pytest --collect-only tests/

# Verify PYTHONPATH
uv run pytest tests/ -v
```

### Import Errors

```bash
# Verify installation
uv sync --group dev

# Check that src/assistant is importable
uv run python -c "from assistant.agent import Agent; print('OK')"
```

### ChromaDB Issues

```bash
# Reinstall ChromaDB
uv pip install --reinstall chromadb

# Clear ChromaDB cache
rm -rf tests/test_chroma_db
```

### Temp Directory Cleanup

```bash
# Clean up test artifacts
rm -rf htmlcov/ .pytest_cache/ .coverage

# Clean temp test directories
rm -rf /tmp/test_*
```

## Configuration Files

### pytest.ini

Located at project root, contains:
- Test discovery patterns
- Async mode configuration
- Markers definitions
- Log settings
- Coverage options

### conftest.py

Located in `tests/`, provides:
- Shared fixtures
- Test utilities
- Setup/teardown logic
- Mock data

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Install uv
      run: curl -LsSf https://astral.sh/uv/install.sh | sh

    - name: Sync dependencies
      run: uv sync --group dev

    - name: Run tests
      run: uv run pytest tests/ --cov=src/assistant --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## Best Practices

1. **Always use uv run** - Ensures correct virtual environment
2. **Run tests before committing** - Catch issues early
3. **Check coverage** - Aim for >80% coverage
4. **Use -x flag during development** - Stop on first failure
5. **Run full suite before PR** - Ensure all tests pass
6. **Keep tests fast** - Use mocks for external services
7. **Isolate tests** - Each test should be independent

## Performance Tips

```bash
# Run only fast tests during development
uv run pytest tests/ -m "not slow"

# Use parallel execution
uv run pytest tests/ -n auto

# Disable coverage for faster runs
uv run pytest tests/ --no-cov

# Run last failed tests first
uv run pytest tests/ --lf
```

## Aliases (Optional)

Add to your `~/.zshrc` or `~/.bashrc`:

```bash
# Test aliases
alias test="uv run pytest tests/"
alias testv="uv run pytest tests/ -v"
alias testcov="uv run pytest tests/ --cov=src/assistant --cov-report=html && open htmlcov/index.html"
alias testf="uv run pytest tests/ -x --ff"  # Fast fail mode
alias testlf="uv run pytest tests/ --lf"    # Last failed

# Usage:
# test              # Run all tests
# testv             # Verbose mode
# testcov           # With coverage report
# testf             # Stop on first failure
```

## Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [pytest-cov](https://pytest-cov.readthedocs.io/)
- [uv documentation](https://docs.astral.sh/uv/)

## Getting Help

```bash
# Pytest help
uv run pytest --help

# List all fixtures
uv run pytest --fixtures

# Show available markers
uv run pytest --markers
```
