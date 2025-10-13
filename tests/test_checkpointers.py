"""Tests for checkpointer backends."""

import pytest
from langchain_core.messages import HumanMessage, AIMessage

from assistant.agent.checkpointers import (
    BaseCheckpointer,
    SqliteCheckpointer,
    MemoryCheckpointer
)


class TestMemoryCheckpointer:
    """Tests for MemoryCheckpointer backend."""

    @pytest.mark.asyncio
    async def test_initialize(self, test_settings):
        """Test checkpointer initialization."""
        checkpointer = MemoryCheckpointer(test_settings)
        await checkpointer.initialize()

        assert checkpointer.checkpointer is not None

    @pytest.mark.asyncio
    async def test_get_checkpointer(self, test_settings):
        """Test getting the underlying checkpointer instance."""
        checkpointer = MemoryCheckpointer(test_settings)
        await checkpointer.initialize()

        instance = checkpointer.get_checkpointer()
        assert instance is not None

    @pytest.mark.asyncio
    async def test_get_history_empty(self, test_settings):
        """Test retrieving history from empty checkpointer."""
        checkpointer = MemoryCheckpointer(test_settings)
        await checkpointer.initialize()

        history = await checkpointer.get_history("thread_1", limit=10)
        assert history == []

    @pytest.mark.asyncio
    async def test_shutdown(self, test_settings):
        """Test checkpointer shutdown."""
        checkpointer = MemoryCheckpointer(test_settings)
        await checkpointer.initialize()

        await checkpointer.shutdown()
        assert checkpointer.checkpointer is None


class TestSqliteCheckpointer:
    """Tests for SqliteCheckpointer backend."""

    @pytest.mark.asyncio
    async def test_initialize(self, sqlite_settings):
        """Test SQLite checkpointer initialization."""
        checkpointer = SqliteCheckpointer(sqlite_settings)
        await checkpointer.initialize()

        assert checkpointer.checkpointer is not None

        # Check that database file was created
        from pathlib import Path
        db_path = Path(sqlite_settings.checkpoint_db_path)
        assert db_path.exists()

    @pytest.mark.asyncio
    async def test_get_checkpointer(self, sqlite_settings):
        """Test getting the underlying AsyncSqliteSaver instance."""
        checkpointer = SqliteCheckpointer(sqlite_settings)
        await checkpointer.initialize()

        instance = checkpointer.get_checkpointer()
        assert instance is not None

    @pytest.mark.asyncio
    async def test_persistence(self, sqlite_settings):
        """Test that SQLite checkpointer persists to disk."""
        checkpointer = SqliteCheckpointer(sqlite_settings)
        await checkpointer.initialize()

        # Shutdown and reinitialize
        await checkpointer.shutdown()

        checkpointer2 = SqliteCheckpointer(sqlite_settings)
        await checkpointer2.initialize()

        # Should successfully reconnect to same database
        assert checkpointer2.checkpointer is not None

    @pytest.mark.asyncio
    async def test_get_history_empty(self, sqlite_settings):
        """Test retrieving history from empty checkpointer."""
        checkpointer = SqliteCheckpointer(sqlite_settings)
        await checkpointer.initialize()

        history = await checkpointer.get_history("thread_1", limit=10)
        assert history == []

    @pytest.mark.asyncio
    async def test_shutdown(self, sqlite_settings):
        """Test SQLite checkpointer shutdown."""
        checkpointer = SqliteCheckpointer(sqlite_settings)
        await checkpointer.initialize()

        await checkpointer.shutdown()
        assert checkpointer.checkpointer is None
