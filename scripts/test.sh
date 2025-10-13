#!/bin/bash
# Quick test runner script for vera-assistant

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Vera Assistant Test Runner ===${NC}\n"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}Warning: uv not found. Install it with:${NC}"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Default command
CMD="${1:-test}"

case $CMD in
    test)
        echo -e "${GREEN}Running all tests...${NC}"
        uv run pytest tests/ -v
        ;;

    fast)
        echo -e "${GREEN}Running tests (fast mode)...${NC}"
        uv run pytest tests/ -x --ff
        ;;

    cov)
        echo -e "${GREEN}Running tests with coverage...${NC}"
        uv run pytest tests/ --cov=src/assistant --cov-report=html --cov-report=term
        echo -e "\n${GREEN}Coverage report generated at: htmlcov/index.html${NC}"
        ;;

    watch)
        echo -e "${GREEN}Running tests in watch mode...${NC}"
        if ! uv pip list | grep -q pytest-watch; then
            echo "Installing pytest-watch..."
            uv add --dev pytest-watch
        fi
        uv run ptw tests/ src/
        ;;

    unit)
        echo -e "${GREEN}Running unit tests...${NC}"
        uv run pytest tests/test_memory.py tests/test_checkpointers.py tests/test_llms.py tests/test_graph.py -v
        ;;

    integration)
        echo -e "${GREEN}Running integration tests...${NC}"
        uv run pytest tests/test_agent.py -v
        ;;

    clean)
        echo -e "${GREEN}Cleaning test artifacts...${NC}"
        rm -rf htmlcov/ .pytest_cache/ .coverage test-results.xml
        rm -rf /tmp/test_*
        echo "Done!"
        ;;

    setup)
        echo -e "${GREEN}Setting up test environment...${NC}"
        uv sync --group dev
        echo -e "\n${GREEN}Setup complete! Run './scripts/test.sh' to run tests.${NC}"
        ;;

    help|--help|-h)
        echo "Usage: ./scripts/test.sh [command]"
        echo ""
        echo "Commands:"
        echo "  test         Run all tests (default)"
        echo "  fast         Run tests in fast-fail mode"
        echo "  cov          Run tests with coverage report"
        echo "  watch        Run tests in watch mode"
        echo "  unit         Run only unit tests"
        echo "  integration  Run only integration tests"
        echo "  clean        Clean test artifacts"
        echo "  setup        Install test dependencies"
        echo "  help         Show this help message"
        echo ""
        echo "Examples:"
        echo "  ./scripts/test.sh"
        echo "  ./scripts/test.sh fast"
        echo "  ./scripts/test.sh cov"
        ;;

    *)
        echo -e "${YELLOW}Unknown command: $CMD${NC}"
        echo "Run './scripts/test.sh help' for usage information"
        exit 1
        ;;
esac
