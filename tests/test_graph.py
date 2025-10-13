"""Tests for graph builder."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import HumanMessage, AIMessage

from assistant.agent.graph import AgentGraphBuilder
from assistant.agent.memory import InMemory
from assistant.agent.checkpointers import MemoryCheckpointer
from assistant.agent.llms import OllamaLLM


class TestAgentGraphBuilder:
    """Tests for AgentGraphBuilder."""

    @pytest.mark.asyncio
    async def test_build_with_memory(self, test_settings):
        """Test building graph with memory enabled."""
        # Initialize components
        llm = OllamaLLM(test_settings)
        memory = InMemory(test_settings)
        checkpointer = MemoryCheckpointer(test_settings)

        await llm.initialize()
        await memory.initialize()
        await checkpointer.initialize()

        # Build graph
        builder = AgentGraphBuilder(llm, memory, checkpointer, test_settings)
        graph = builder.build()

        assert graph is not None

    @pytest.mark.asyncio
    async def test_build_without_memory(self, test_settings):
        """Test building graph without memory."""
        test_settings.enable_semantic_memory = False

        # Initialize components
        llm = OllamaLLM(test_settings)
        checkpointer = MemoryCheckpointer(test_settings)

        await llm.initialize()
        await checkpointer.initialize()

        # Build graph without memory
        builder = AgentGraphBuilder(llm, None, checkpointer, test_settings)
        graph = builder.build()

        assert graph is not None

    @pytest.mark.asyncio
    async def test_add_custom_node(self, test_settings):
        """Test adding custom nodes to the graph."""
        # Initialize components
        llm = OllamaLLM(test_settings)
        checkpointer = MemoryCheckpointer(test_settings)

        await llm.initialize()
        await checkpointer.initialize()

        # Create custom node
        async def custom_node(state, config):
            return {"messages": [AIMessage(content="Custom response")]}

        # Build graph with custom node
        builder = AgentGraphBuilder(llm, None, checkpointer, test_settings)
        builder.add_node("custom_node", custom_node)

        graph = builder.build()
        assert graph is not None

    @pytest.mark.asyncio
    async def test_add_custom_edge(self, test_settings):
        """Test adding custom edges to the graph."""
        # Initialize components
        llm = OllamaLLM(test_settings)
        checkpointer = MemoryCheckpointer(test_settings)

        await llm.initialize()
        await checkpointer.initialize()

        # Build graph with custom edge
        builder = AgentGraphBuilder(llm, None, checkpointer, test_settings)

        # Note: Custom edges won't override core edges in our implementation
        # This test ensures the API works without errors
        builder.add_edge("custom_start", "custom_end")

        graph = builder.build()
        assert graph is not None

    @pytest.mark.asyncio
    async def test_method_chaining(self, test_settings):
        """Test method chaining for builder pattern."""
        # Initialize components
        llm = OllamaLLM(test_settings)
        checkpointer = MemoryCheckpointer(test_settings)

        await llm.initialize()
        await checkpointer.initialize()

        # Test method chaining
        async def node1(state, config):
            return {}

        async def node2(state, config):
            return {}

        builder = AgentGraphBuilder(llm, None, checkpointer, test_settings)
        result = builder.add_node("node1", node1).add_node("node2", node2)

        assert result is builder  # Should return self for chaining
