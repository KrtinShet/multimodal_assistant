"""Abstract base class for memory backends."""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class BaseMemory(ABC):
    """
    Abstract interface for memory backends.

    Memory backends provide long-term storage and retrieval of information
    using semantic search capabilities.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the memory backend.

        Raises:
            Exception: If initialization fails
        """
        pass

    @abstractmethod
    async def store(
        self,
        memory_text: str,
        user_id: str = "default_user",
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Store a memory in the backend.

        Args:
            memory_text: Text content to store
            user_id: User identifier for namespacing
            metadata: Additional metadata to associate with the memory

        Returns:
            Memory ID

        Raises:
            Exception: If storage fails
        """
        pass

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        user_id: str = "default_user",
        limit: int = 5
    ) -> List[str]:
        """
        Retrieve memories by semantic similarity.

        Args:
            query: Search query
            user_id: User identifier for filtering
            limit: Maximum number of results

        Returns:
            List of memory texts

        Raises:
            Exception: If retrieval fails
        """
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        user_id: str = "default_user",
        limit: int = 5
    ) -> List[Dict]:
        """
        Search memories and return detailed results.

        Args:
            query: Search query
            user_id: User identifier for filtering
            limit: Maximum number of results

        Returns:
            List of dictionaries containing:
                - content: Memory text
                - metadata: Associated metadata
                - distance/score: Similarity score

        Raises:
            Exception: If search fails
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Clean up resources."""
        pass


__all__ = ["BaseMemory"]
