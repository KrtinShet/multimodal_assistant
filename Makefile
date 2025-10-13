# Makefile for vera-assistant development tasks

.PHONY: help setup test test-fast test-cov test-watch clean lint format

# Default target
.DEFAULT_GOAL := help

help:  ## Show this help message
	@echo "Vera Assistant - Development Commands"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup:  ## Install all dependencies including dev tools
	uv sync --group dev

test:  ## Run all tests
	uv run pytest tests/ -v

test-fast:  ## Run tests in fast-fail mode (stop on first failure)
	uv run pytest tests/ -x --ff

test-cov:  ## Run tests with coverage report
	uv run pytest tests/ --cov=src/assistant --cov-report=html --cov-report=term
	@echo "\nCoverage report: htmlcov/index.html"

test-unit:  ## Run only unit tests
	uv run pytest tests/test_memory.py tests/test_checkpointers.py tests/test_llms.py tests/test_graph.py -v

test-integration:  ## Run only integration tests
	uv run pytest tests/test_agent.py -v

test-watch:  ## Run tests in watch mode
	uv run ptw tests/ src/

clean:  ## Clean test artifacts and caches
	rm -rf htmlcov/ .pytest_cache/ .coverage test-results.xml
	rm -rf /tmp/test_*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

lint:  ## Run linters (ruff)
	uv run ruff check src/ tests/

format:  ## Format code with black
	uv run black src/ tests/

format-check:  ## Check code formatting
	uv run black --check src/ tests/

type-check:  ## Run type checker (mypy)
	uv run mypy src/

install:  ## Install package in development mode
	uv pip install -e .

update:  ## Update all dependencies
	uv sync --upgrade --group dev

# CI/CD targets
ci-test:  ## Run tests for CI/CD
	uv run pytest tests/ --cov=src/assistant --cov-report=xml --cov-report=term -v

ci-lint:  ## Run all checks for CI/CD
	uv run ruff check src/ tests/
	uv run black --check src/ tests/
	uv run mypy src/
