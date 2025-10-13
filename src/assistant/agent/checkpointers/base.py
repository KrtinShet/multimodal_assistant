"""Abstract base class for checkpointer backends."""

from abc import ABC, abstractmethod
from typing import Any, List
from langchain_core.messages import BaseMessage


class BaseCheckpointer(ABC):
    """
    Abstract interface for checkpointer backends.

    Checkpointers provide conversation state persistence across sessions,
    enabling features like conversation history and state recovery.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the checkpointer backend.

        Raises:
            Exception: If initialization fails
        """
        pass

    @abstractmethod
    def get_checkpointer(self) -> Any:
        """
        Get the underlying LangGraph checkpointer instance.

        Returns:
            Checkpointer instance (e.g., AsyncSqliteSaver, InMemorySaver)
        """
        pass

    @abstractmethod
    async def get_history(
        self,
        thread_id: str,
        limit: int = 10
    ) -> List[BaseMessage]:
        """
        Retrieve conversation history for a thread.

        Args:
            thread_id: Thread identifier
            limit: Maximum number of messages to retrieve

        Returns:
            List of messages from the conversation

        Raises:
            Exception: If retrieval fails
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Clean up resources."""
        pass


__all__ = ["BaseCheckpointer"]
