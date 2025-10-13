"""Tests for memory backends."""

import pytest

from assistant.agent.memory import BaseMemory, ChromaMemory, InMemory


class TestInMemory:
    """Tests for InMemory backend."""

    @pytest.mark.asyncio
    async def test_initialize(self, test_settings):
        """Test memory initialization."""
        memory = InMemory(test_settings)
        await memory.initialize()

        assert memory.memories == {}
        assert len(memory.user_index) == 0

    @pytest.mark.asyncio
    async def test_store_memory(self, test_settings):
        """Test storing a memory."""
        memory = InMemory(test_settings)
        await memory.initialize()

        memory_id = await memory.store(
            memory_text="Test memory content",
            user_id="test_user",
            metadata={"type": "test"}
        )

        assert memory_id is not None
        assert memory_id in memory.memories
        assert memory.memories[memory_id]["text"] == "Test memory content"
        assert memory.memories[memory_id]["user_id"] == "test_user"

    @pytest.mark.asyncio
    async def test_retrieve_memories(self, test_settings, sample_memories):
        """Test retrieving memories by keyword."""
        memory = InMemory(test_settings)
        await memory.initialize()

        # Store sample memories
        for mem in sample_memories:
            await memory.store(mem["text"], mem["user_id"], mem["metadata"])

        # Retrieve memories
        results = await memory.retrieve("pizza", "test_user", limit=5)

        assert len(results) > 0
        assert any("pizza" in r.lower() for r in results)

    @pytest.mark.asyncio
    async def test_search_memories(self, test_settings, sample_memories):
        """Test searching memories with details."""
        memory = InMemory(test_settings)
        await memory.initialize()

        # Store sample memories
        for mem in sample_memories:
            await memory.store(mem["text"], mem["user_id"], mem["metadata"])

        # Search memories
        results = await memory.search("Bob", "test_user", limit=5)

        assert len(results) > 0
        assert all("content" in r for r in results)
        assert all("metadata" in r for r in results)
        assert all("distance" in r for r in results)

    @pytest.mark.asyncio
    async def test_user_isolation(self, test_settings):
        """Test that memories are isolated by user."""
        memory = InMemory(test_settings)
        await memory.initialize()

        # Store memories for different users
        await memory.store("User 1 memory", "user1", {"type": "test"})
        await memory.store("User 2 memory", "user2", {"type": "test"})

        # Retrieve for user1
        user1_results = await memory.retrieve("memory", "user1")
        assert len(user1_results) == 1
        assert "User 1" in user1_results[0]

        # Retrieve for user2
        user2_results = await memory.retrieve("memory", "user2")
        assert len(user2_results) == 1
        assert "User 2" in user2_results[0]

    @pytest.mark.asyncio
    async def test_shutdown(self, test_settings):
        """Test memory shutdown."""
        memory = InMemory(test_settings)
        await memory.initialize()

        await memory.store("Test", "user1")
        assert len(memory.memories) > 0

        await memory.shutdown()
        assert len(memory.memories) == 0


class TestChromaMemory:
    """Tests for ChromaDB backend."""

    @pytest.mark.asyncio
    async def test_initialize(self, chroma_settings):
        """Test ChromaDB initialization."""
        memory = ChromaMemory(chroma_settings)
        await memory.initialize()

        assert memory.client is not None
        assert memory.collection is not None

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, chroma_settings):
        """Test storing and retrieving from ChromaDB."""
        memory = ChromaMemory(chroma_settings)
        await memory.initialize()

        # Store memory
        memory_id = await memory.store(
            memory_text="I love hiking in the mountains",
            user_id="test_user",
            metadata={"type": "preference"}
        )

        assert memory_id is not None

        # Retrieve memory
        results = await memory.retrieve("mountains", "test_user", limit=1)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_semantic_search(self, chroma_settings):
        """Test semantic search capabilities."""
        memory = ChromaMemory(chroma_settings)
        await memory.initialize()

        # Store related memories
        await memory.store("I enjoy outdoor activities", "test_user")
        await memory.store("Coding is my passion", "test_user")

        # Search for outdoor-related content
        results = await memory.search("nature", "test_user", limit=2)

        # ChromaDB should find semantic similarity
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_shutdown(self, chroma_settings):
        """Test ChromaDB shutdown."""
        memory = ChromaMemory(chroma_settings)
        await memory.initialize()

        await memory.shutdown()
        assert memory.client is None
        assert memory.collection is None
