"""Tests for LLM providers."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from assistant.agent.llms import BaseLLM, OllamaLLM, OpenAILLM


class TestOllamaLLM:
    """Tests for Ollama LLM provider."""

    @pytest.mark.asyncio
    async def test_initialize(self, test_settings):
        """Test Ollama LLM initialization."""
        llm = OllamaLLM(test_settings)
        await llm.initialize()

        assert llm.llm is not None

    @pytest.mark.asyncio
    async def test_get_llm(self, test_settings):
        """Test getting the underlying LLM instance."""
        llm = OllamaLLM(test_settings)
        await llm.initialize()

        instance = llm.get_llm()
        assert instance is not None

    @pytest.mark.asyncio
    async def test_invoke_with_mock(self, test_settings):
        """Test LLM invocation with mocked response."""
        llm = OllamaLLM(test_settings)
        await llm.initialize()

        # Mock the LLM response
        mock_response = AIMessage(content="Hello! How can I help you?")
        llm.llm.ainvoke = AsyncMock(return_value=mock_response)

        messages = [HumanMessage(content="Hello")]
        response = await llm.invoke(messages)

        assert isinstance(response, AIMessage)
        assert response.content == "Hello! How can I help you?"

    @pytest.mark.asyncio
    async def test_stream_with_mock(self, test_settings):
        """Test LLM streaming with mocked response."""
        llm = OllamaLLM(test_settings)
        await llm.initialize()

        # Mock streaming response
        async def mock_stream(messages):
            chunks = ["Hello", " ", "world", "!"]
            for chunk in chunks:
                yield AIMessage(content=chunk)

        llm.llm.astream = mock_stream

        messages = [HumanMessage(content="Hello")]
        chunks = []
        async for chunk in llm.stream(messages):
            chunks.append(chunk)

        assert "".join(chunks) == "Hello world!"

    @pytest.mark.asyncio
    async def test_shutdown(self, test_settings):
        """Test LLM shutdown."""
        llm = OllamaLLM(test_settings)
        await llm.initialize()

        await llm.shutdown()
        assert llm.llm is None


class TestOpenAILLM:
    """Tests for OpenAI LLM provider."""

    @pytest.mark.asyncio
    async def test_initialize_without_api_key(self, test_settings):
        """Test OpenAI initialization without explicit API key."""
        test_settings.llm_backend = "openai"
        test_settings.agent_model_name = "gpt-3.5-turbo"

        llm = OpenAILLM(test_settings)

        # Should initialize even without key (will use env var or fail at invoke time)
        await llm.initialize()
        assert llm.llm is not None

    @pytest.mark.asyncio
    async def test_initialize_with_api_key(self, test_settings):
        """Test OpenAI initialization with API key."""
        test_settings.llm_backend = "openai"
        test_settings.agent_model_name = "gpt-3.5-turbo"
        test_settings.openai_api_key = "sk-test-key"

        llm = OpenAILLM(test_settings)
        await llm.initialize()

        assert llm.llm is not None

    @pytest.mark.asyncio
    async def test_initialize_with_custom_base_url(self, test_settings):
        """Test OpenAI initialization with custom base URL."""
        test_settings.llm_backend = "openai"
        test_settings.agent_model_name = "gpt-3.5-turbo"
        test_settings.openai_api_key = "sk-test-key"
        test_settings.openai_base_url = "http://localhost:8000/v1"

        llm = OpenAILLM(test_settings)
        await llm.initialize()

        assert llm.llm is not None

    @pytest.mark.asyncio
    async def test_invoke_with_mock(self, test_settings):
        """Test OpenAI invocation with mocked response."""
        test_settings.llm_backend = "openai"
        test_settings.openai_api_key = "sk-test-key"

        llm = OpenAILLM(test_settings)
        await llm.initialize()

        # Mock the LLM response
        mock_response = AIMessage(content="I'm an AI assistant.")
        llm.llm.ainvoke = AsyncMock(return_value=mock_response)

        messages = [HumanMessage(content="Who are you?")]
        response = await llm.invoke(messages)

        assert isinstance(response, AIMessage)
        assert "assistant" in response.content.lower()

    @pytest.mark.asyncio
    async def test_shutdown(self, test_settings):
        """Test OpenAI LLM shutdown."""
        test_settings.llm_backend = "openai"
        test_settings.openai_api_key = "sk-test-key"

        llm = OpenAILLM(test_settings)
        await llm.initialize()

        await llm.shutdown()
        assert llm.llm is None
