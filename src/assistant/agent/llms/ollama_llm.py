"""Ollama LLM provider implementation."""

from typing import Any, List, AsyncIterator
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama

from assistant.agent.llms.base import BaseLLM
from assistant.config.settings import Settings
from assistant.utils.logger import setup_logger


class OllamaLLM(BaseLLM):
    """
    Ollama LLM provider for local inference.

    Features:
    - Local model execution
    - No API key required
    - Privacy-focused
    - Supports various open-source models
    """

    def __init__(self, settings: Settings):
        """
        Initialize Ollama LLM provider.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.logger = setup_logger("assistant.agent.llm.ollama", settings.log_level)

        self.llm = None

    async def initialize(self) -> None:
        """Initialize Ollama LLM."""
        model_name = self.settings.agent_model_name
        self.logger.info(f"Initializing Ollama LLM: {model_name}")

        try:
            self.llm = ChatOllama(
                model=model_name,
                temperature=self.settings.temperature,
                top_p=self.settings.top_p,
            )

            self.logger.info(f"Ollama LLM initialized: {model_name}")

        except Exception as e:
            self.logger.error(f"Failed to initialize Ollama LLM: {e}", exc_info=True)
            raise

    def get_llm(self) -> Any:
        """
        Get the underlying ChatOllama instance.

        Returns:
            ChatOllama instance
        """
        if not self.llm:
            raise RuntimeError("LLM not initialized")

        return self.llm

    async def invoke(self, messages: List[BaseMessage]) -> BaseMessage:
        """
        Invoke the LLM with messages.

        Args:
            messages: List of messages for the conversation

        Returns:
            AI response message
        """
        if not self.llm:
            raise RuntimeError("LLM not initialized")

        try:
            response = await self.llm.ainvoke(messages)
            return response

        except Exception as e:
            self.logger.error(f"Failed to invoke LLM: {e}", exc_info=True)
            raise

    async def stream(self, messages: List[BaseMessage]) -> AsyncIterator[str]:
        """
        Stream responses from the LLM.

        Args:
            messages: List of messages for the conversation

        Yields:
            Chunks of the AI response
        """
        if not self.llm:
            raise RuntimeError("LLM not initialized")

        try:
            async for chunk in self.llm.astream(messages):
                if chunk.content:
                    yield chunk.content

        except Exception as e:
            self.logger.error(f"Failed to stream from LLM: {e}", exc_info=True)
            raise

    async def shutdown(self) -> None:
        """Clean up resources."""
        self.logger.info("Shutting down Ollama LLM")
        self.llm = None


__all__ = ["OllamaLLM"]
