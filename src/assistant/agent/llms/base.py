"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from typing import Any, List, AsyncIterator
from langchain_core.messages import BaseMessage


class BaseLLM(ABC):
    """
    Abstract interface for LLM providers.

    LLM providers wrap different language model APIs (Ollama, OpenAI, etc.)
    and provide a unified interface for the agent.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the LLM provider.

        Raises:
            Exception: If initialization fails
        """
        pass

    @abstractmethod
    def get_llm(self) -> Any:
        """
        Get the underlying LangChain LLM instance.

        Returns:
            LLM instance (e.g., ChatOllama, ChatOpenAI)
        """
        pass

    @abstractmethod
    async def invoke(self, messages: List[BaseMessage]) -> BaseMessage:
        """
        Invoke the LLM with a list of messages.

        Args:
            messages: List of messages for the conversation

        Returns:
            AI response message

        Raises:
            Exception: If invocation fails
        """
        pass

    @abstractmethod
    async def stream(self, messages: List[BaseMessage]) -> AsyncIterator[str]:
        """
        Stream responses from the LLM.

        Args:
            messages: List of messages for the conversation

        Yields:
            Chunks of the AI response

        Raises:
            Exception: If streaming fails
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Clean up resources."""
        pass


__all__ = ["BaseLLM"]
