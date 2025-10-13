"""Pytest configuration and shared fixtures."""

import pytest
import tempfile
import shutil
from pathlib import Path

from assistant.config.settings import Settings


@pytest.fixture
def test_settings():
    """Create test settings with temporary directories."""
    temp_dir = tempfile.mkdtemp()

    settings = Settings(
        # Use in-memory backends for faster tests
        llm_backend="ollama",
        memory_backend="inmemory",
        checkpointer_backend="memory",

        # Test model configuration
        agent_model_name="gemma3:4b",

        # Temporary paths
        checkpoint_db_path=f"{temp_dir}/test_checkpoints.db",
        chroma_db_path=f"{temp_dir}/test_chroma_db",

        # Test settings
        memory_retrieval_limit=3,
        enable_semantic_memory=True,
        log_level="DEBUG",

        # LLM settings
        temperature=0.7,
        top_p=0.9,
    )

    yield settings

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def chroma_settings():
    """Settings configured for ChromaDB testing."""
    temp_dir = tempfile.mkdtemp()

    settings = Settings(
        memory_backend="chroma",
        checkpointer_backend="sqlite",
        chroma_db_path=f"{temp_dir}/test_chroma_db",
        checkpoint_db_path=f"{temp_dir}/test_checkpoints.db",
        enable_semantic_memory=True,
        log_level="DEBUG",
    )

    yield settings

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sqlite_settings():
    """Settings configured for SQLite checkpointer testing."""
    temp_dir = tempfile.mkdtemp()

    settings = Settings(
        checkpointer_backend="sqlite",
        checkpoint_db_path=f"{temp_dir}/test_checkpoints.db",
        log_level="DEBUG",
    )

    yield settings

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_llm_responses():
    """Mock LLM responses for testing."""
    return [
        "Hello! How can I assist you today?",
        "I'm doing great, thank you for asking!",
        "Your name is Bob, as you mentioned earlier.",
    ]


@pytest.fixture
def sample_memories():
    """Sample memories for testing."""
    return [
        {
            "text": "User: My name is Bob\nAssistant: Nice to meet you, Bob!",
            "user_id": "test_user",
            "metadata": {"type": "conversation"}
        },
        {
            "text": "User: I like pizza\nAssistant: Pizza is delicious!",
            "user_id": "test_user",
            "metadata": {"type": "conversation"}
        },
        {
            "text": "User: I prefer morning workouts\nAssistant: Morning workouts are a great way to start the day!",
            "user_id": "test_user",
            "metadata": {"type": "conversation"}
        },
    ]


@pytest.fixture
def sample_conversation():
    """Sample conversation messages."""
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

    return [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Hello!"),
        AIMessage(content="Hi! How can I help you today?"),
        HumanMessage(content="What's the weather?"),
        AIMessage(content="I don't have access to weather data."),
    ]
