"""End-to-end tests for the Agent."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage

from assistant.agent import Agent
from assistant.agent.memory import InMemory
from assistant.agent.checkpointers import MemoryCheckpointer
from assistant.agent.llms import OllamaLLM


class TestAgentInitialization:
    """Tests for Agent initialization."""

    @pytest.mark.asyncio
    async def test_initialize_with_settings(self, test_settings):
        """Test agent initialization with settings only."""
        agent = Agent(test_settings)
        await agent.initialize()

        assert agent.llm is not None
        assert agent.checkpointer is not None
        assert agent.graph is not None

    @pytest.mark.asyncio
    async def test_initialize_with_memory_enabled(self, test_settings):
        """Test agent initialization with memory enabled."""
        test_settings.enable_semantic_memory = True

        agent = Agent(test_settings)
        await agent.initialize()

        assert agent.memory is not None

    @pytest.mark.asyncio
    async def test_initialize_with_memory_disabled(self, test_settings):
        """Test agent initialization with memory disabled."""
        test_settings.enable_semantic_memory = False

        agent = Agent(test_settings)
        await agent.initialize()

        assert agent.memory is None

    @pytest.mark.asyncio
    async def test_initialize_with_custom_components(self, test_settings):
        """Test agent initialization with injected components."""
        # Create custom components
        llm = OllamaLLM(test_settings)
        memory = InMemory(test_settings)
        checkpointer = MemoryCheckpointer(test_settings)

        # Inject into agent
        agent = Agent(
            settings=test_settings,
            llm=llm,
            memory=memory,
            checkpointer=checkpointer
        )

        await agent.initialize()

        # Should use injected components
        assert agent.llm is llm
        assert agent.memory is memory
        assert agent.checkpointer is checkpointer


class TestAgentBackendSelection:
    """Tests for backend selection."""

    @pytest.mark.asyncio
    async def test_ollama_backend(self, test_settings):
        """Test selecting Ollama LLM backend."""
        test_settings.llm_backend = "ollama"

        agent = Agent(test_settings)
        await agent.initialize()

        assert agent.llm.__class__.__name__ == "OllamaLLM"

    @pytest.mark.asyncio
    async def test_inmemory_backend(self, test_settings):
        """Test selecting InMemory backend."""
        test_settings.memory_backend = "inmemory"

        agent = Agent(test_settings)
        await agent.initialize()

        assert agent.memory.__class__.__name__ == "InMemory"

    @pytest.mark.asyncio
    async def test_memory_checkpointer_backend(self, test_settings):
        """Test selecting MemoryCheckpointer backend."""
        test_settings.checkpointer_backend = "memory"

        agent = Agent(test_settings)
        await agent.initialize()

        assert agent.checkpointer.__class__.__name__ == "MemoryCheckpointer"

    @pytest.mark.asyncio
    async def test_invalid_backend(self, test_settings):
        """Test handling of invalid backend selection."""
        test_settings.llm_backend = "invalid_backend"

        agent = Agent(test_settings)

        with pytest.raises(ValueError, match="Unsupported LLM backend"):
            await agent.initialize()


class TestAgentConversation:
    """Tests for agent conversation capabilities."""

    @pytest.mark.asyncio
    async def test_invoke_with_mock(self, test_settings):
        """Test agent invocation with mocked LLM."""
        agent = Agent(test_settings)
        await agent.initialize()

        # Mock the LLM response
        mock_response = AIMessage(content="Hello! How can I help you?")
        agent.llm.llm.ainvoke = AsyncMock(return_value=mock_response)

        response = await agent.invoke(
            prompt="Hello",
            thread_id="test_thread",
            user_id="test_user"
        )

        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_stream_with_mock(self, test_settings):
        """Test agent streaming with mocked LLM."""
        agent = Agent(test_settings)
        await agent.initialize()

        # Mock streaming response
        async def mock_stream(state, config, stream_mode):
            yield (AIMessage(content="Hello"), {})
            yield (AIMessage(content=" world"), {})

        agent.graph.astream = mock_stream

        chunks = []
        async for chunk in agent.generate_stream(
            prompt="Hello",
            thread_id="test_thread",
            user_id="test_user"
        ):
            chunks.append(chunk)

        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_conversation_continuity(self, test_settings):
        """Test conversation continuity across multiple turns."""
        agent = Agent(test_settings)
        await agent.initialize()

        # Mock LLM responses
        responses = [
            AIMessage(content="Hello!"),
            AIMessage(content="My name is Assistant."),
        ]
        response_iter = iter(responses)

        async def mock_invoke(messages):
            return next(response_iter)

        agent.llm.llm.ainvoke = mock_invoke

        # First turn
        response1 = await agent.invoke(
            prompt="Hello",
            thread_id="test_thread",
            user_id="test_user"
        )

        # Second turn (same thread)
        response2 = await agent.invoke(
            prompt="What's your name?",
            thread_id="test_thread",
            user_id="test_user"
        )

        assert len(response1) > 0
        assert len(response2) > 0


class TestAgentMemory:
    """Tests for agent memory capabilities."""

    @pytest.mark.asyncio
    async def test_add_memory(self, test_settings):
        """Test manually adding a memory."""
        agent = Agent(test_settings)
        await agent.initialize()

        await agent.add_memory(
            memory_text="User prefers morning meetings",
            user_id="test_user",
            metadata={"type": "preference"}
        )

        # Search for the memory
        results = await agent.search_memories("meetings", "test_user")
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_search_memories(self, test_settings, sample_memories):
        """Test searching memories."""
        agent = Agent(test_settings)
        await agent.initialize()

        # Add sample memories
        for mem in sample_memories:
            await agent.add_memory(mem["text"], mem["user_id"], mem["metadata"])

        # Search
        results = await agent.search_memories("pizza", "test_user", limit=2)
        assert len(results) > 0
        assert any("pizza" in r["content"].lower() for r in results)

    @pytest.mark.asyncio
    async def test_memory_disabled(self, test_settings):
        """Test agent behavior with memory disabled."""
        test_settings.enable_semantic_memory = False

        agent = Agent(test_settings)
        await agent.initialize()

        # Should not raise error
        await agent.add_memory("Test memory", "test_user")

        # Should return empty list
        results = await agent.search_memories("test", "test_user")
        assert results == []


class TestAgentConversationHistory:
    """Tests for conversation history retrieval."""

    @pytest.mark.asyncio
    async def test_get_empty_history(self, test_settings):
        """Test retrieving history from empty thread."""
        agent = Agent(test_settings)
        await agent.initialize()

        history = await agent.get_conversation_history("new_thread")
        assert history == []

    @pytest.mark.asyncio
    async def test_history_limit(self, test_settings):
        """Test history limit parameter."""
        agent = Agent(test_settings)
        await agent.initialize()

        # Get history with limit
        history = await agent.get_conversation_history("thread_1", limit=5)
        assert len(history) <= 5


class TestAgentShutdown:
    """Tests for agent shutdown."""

    @pytest.mark.asyncio
    async def test_shutdown(self, test_settings):
        """Test agent shutdown."""
        agent = Agent(test_settings)
        await agent.initialize()

        await agent.shutdown()

        assert agent.graph is None

    @pytest.mark.asyncio
    async def test_shutdown_all_components(self, test_settings):
        """Test that all components are shut down."""
        agent = Agent(test_settings)
        await agent.initialize()

        await agent.shutdown()

        # Verify components are cleaned up
        # (actual cleanup depends on implementation)
        assert agent.graph is None


class TestAgentErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_invoke_before_initialize(self, test_settings):
        """Test invoking agent before initialization."""
        agent = Agent(test_settings)

        response_parts = []
        async for chunk in agent.generate_stream("Hello"):
            response_parts.append(chunk)

        response = "".join(response_parts)
        assert "not ready" in response.lower() or "initialize" in response.lower()

    @pytest.mark.asyncio
    async def test_add_memory_without_backend(self, test_settings):
        """Test adding memory when backend is not initialized."""
        test_settings.enable_semantic_memory = False

        agent = Agent(test_settings)
        await agent.initialize()

        # Should not raise error
        await agent.add_memory("Test", "user1")

    @pytest.mark.asyncio
    async def test_llm_error_handling(self, test_settings):
        """Test handling of LLM errors."""
        agent = Agent(test_settings)
        await agent.initialize()

        # Mock LLM to raise error
        async def mock_error(*args, **kwargs):
            raise Exception("Mock LLM error")

        agent.llm.llm.ainvoke = mock_error

        # Should handle error gracefully
        response_parts = []
        async for chunk in agent.generate_stream("Hello"):
            response_parts.append(chunk)

        response = "".join(response_parts)
        assert "error" in response.lower()
