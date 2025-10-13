"""OpenAI LLM provider implementation."""

from typing import Any, List, AsyncIterator
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from assistant.agent.llms.base import BaseLLM
from assistant.config.settings import Settings
from assistant.utils.logger import setup_logger


class OpenAILLM(BaseLLM):
    """
    OpenAI LLM provider for cloud-based inference.

    Features:
    - Access to GPT models
    - High-quality responses
    - Requires API key
    - Supports custom base URLs (for compatible APIs)
    """

    def __init__(self, settings: Settings):
        """
        Initialize OpenAI LLM provider.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.logger = setup_logger("assistant.agent.llm.openai", settings.log_level)

        self.llm = None

    async def initialize(self) -> None:
        """Initialize OpenAI LLM."""
        model_name = self.settings.agent_model_name
        self.logger.info(f"Initializing OpenAI LLM: {model_name}")

        try:
            # Build initialization kwargs
            init_kwargs = {
                "model": model_name,
                "temperature": self.settings.temperature,
                "top_p": self.settings.top_p,
            }

            # Add API key if provided
            if self.settings.openai_api_key:
                init_kwargs["api_key"] = self.settings.openai_api_key

            # Add custom base URL if provided
            if self.settings.openai_base_url:
                init_kwargs["base_url"] = self.settings.openai_base_url

            self.llm = ChatOpenAI(**init_kwargs)

            self.logger.info(f"OpenAI LLM initialized: {model_name}")

        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI LLM: {e}", exc_info=True)
            raise

    def get_llm(self) -> Any:
        """
        Get the underlying ChatOpenAI instance.

        Returns:
            ChatOpenAI instance
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
        self.logger.info("Shutting down OpenAI LLM")
        self.llm = None


__all__ = ["OpenAILLM"]
