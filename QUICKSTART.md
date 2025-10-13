# Quick Start - Testing with uv

Get started with testing in 30 seconds!

## 🚀 One-Command Setup

```bash
# Install dependencies and run tests
uv sync --group dev && uv run pytest tests/ -v
```

## 📋 Step-by-Step Setup

### 1. Install Dependencies

```bash
# Install all dependencies (main + dev)
uv sync --group dev
```

This installs:
- ✅ pytest (test framework)
- ✅ pytest-asyncio (async test support)
- ✅ pytest-cov (coverage reports)
- ✅ pytest-timeout (test timeouts)
- ✅ All agent dependencies (LangChain, ChromaDB, etc.)

### 2. Run Tests

```bash
# Run all tests
uv run pytest tests/

# Run with verbose output
uv run pytest tests/ -v
```

### 3. View Coverage

```bash
# Generate coverage report
uv run pytest tests/ --cov=src/assistant --cov-report=html

# Open in browser (macOS)
open htmlcov/index.html

# Or view in terminal
uv run pytest tests/ --cov=src/assistant --cov-report=term-missing
```

## 🎯 Quick Commands

### Using Make (Recommended)

```bash
make help          # Show all commands
make setup         # Install dependencies
make test          # Run all tests
make test-fast     # Fast-fail mode
make test-cov      # With coverage
make clean         # Clean artifacts
```

### Using Test Script

```bash
./scripts/test.sh           # Run all tests
./scripts/test.sh fast      # Fast-fail mode
./scripts/test.sh cov       # With coverage
./scripts/test.sh unit      # Unit tests only
./scripts/test.sh help      # Show help
```

### Using uv Directly

```bash
uv run pytest tests/                    # All tests
uv run pytest tests/ -v                 # Verbose
uv run pytest tests/test_agent.py       # Specific file
uv run pytest tests/ -k "memory"        # Match pattern
uv run pytest tests/ -x                 # Stop on first failure
```

## 📊 Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── test_agent.py            # Agent integration tests (22 tests)
├── test_checkpointers.py    # Checkpointer tests (9 tests)
├── test_graph.py            # Graph builder tests (5 tests)
├── test_llms.py             # LLM provider tests (10 tests)
└── test_memory.py           # Memory backend tests (10 tests)

Total: 56+ tests
```

## ✨ Common Workflows

### Development Mode

```bash
# Run tests on every change (requires pytest-watch)
uv add --dev pytest-watch
uv run ptw tests/ src/
```

### Pre-Commit

```bash
# Run fast tests before committing
make test-fast
# or
uv run pytest tests/ -x --ff
```

### Full Test Suite

```bash
# Run everything with coverage
make test-cov
# or
uv run pytest tests/ --cov=src/assistant --cov-report=html
```

## 🔧 Troubleshooting

### "Module not found" Errors

```bash
# Re-sync dependencies
uv sync --group dev

# Verify installation
uv run python -c "from assistant.agent import Agent; print('✓ OK')"
```

### Tests Not Found

```bash
# Check test discovery
uv run pytest --collect-only tests/

# Run from project root
cd /path/to/vera-assistant
uv run pytest tests/
```

### Clean Start

```bash
# Clean everything and reinstall
make clean
uv sync --group dev
uv run pytest tests/ -v
```

## 📚 More Information

- **Detailed Guide**: See [TESTING.md](TESTING.md)
- **Test Documentation**: See [tests/README.md](tests/README.md)
- **Configuration**: See [pytest.ini](pytest.ini)

## 🎓 Example Output

```bash
$ uv run pytest tests/ -v

tests/test_memory.py::TestInMemory::test_initialize PASSED              [ 2%]
tests/test_memory.py::TestInMemory::test_store_memory PASSED            [ 4%]
tests/test_checkpointers.py::TestMemoryCheckpointer::test_initialize PASSED [ 6%]
tests/test_llms.py::TestOllamaLLM::test_initialize PASSED              [ 8%]
tests/test_graph.py::TestAgentGraphBuilder::test_build_with_memory PASSED [10%]
tests/test_agent.py::TestAgentInitialization::test_initialize_with_settings PASSED [12%]
...

===================== 56 passed in 3.45s =====================
```

## 🚨 Next Steps

1. **Run tests**: `make test`
2. **Check coverage**: `make test-cov`
3. **Add your tests**: See [tests/README.md](tests/README.md)
4. **Set up CI/CD**: See [TESTING.md](TESTING.md#cicd-integration)

Happy testing! 🎉
