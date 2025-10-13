"""In-memory fallback implementation for testing."""

from typing import List, Dict, Optional
from uuid import uuid4
from collections import defaultdict

from assistant.agent.memory.base import BaseMemory
from assistant.config.settings import Settings
from assistant.utils.logger import setup_logger


class InMemory(BaseMemory):
    """
    Simple in-memory storage backend for testing and development.

    Features:
    - No external dependencies
    - Fast for small datasets
    - Simple keyword-based search (not semantic)
    - Data lost on restart
    """

    def __init__(self, settings: Settings):
        """
        Initialize in-memory storage.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.logger = setup_logger("assistant.agent.memory.inmemory", settings.log_level)

        # Storage: {memory_id: {text, user_id, metadata}}
        self.memories: Dict[str, Dict] = {}

        # Index by user for faster lookup
        self.user_index: Dict[str, List[str]] = defaultdict(list)

    async def initialize(self) -> None:
        """Initialize the in-memory storage."""
        self.logger.info("Initializing in-memory storage")
        self.memories = {}
        self.user_index = defaultdict(list)

    async def store(
        self,
        memory_text: str,
        user_id: str = "default_user",
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Store a memory in-memory.

        Args:
            memory_text: Text content to store
            user_id: User identifier for namespacing
            metadata: Additional metadata

        Returns:
            Memory ID
        """
        memory_id = str(uuid4())

        # Store memory
        self.memories[memory_id] = {
            "text": memory_text,
            "user_id": user_id,
            "metadata": metadata or {}
        }

        # Update user index
        self.user_index[user_id].append(memory_id)

        self.logger.debug(f"Memory stored: {memory_id}")
        return memory_id

    async def retrieve(
        self,
        query: str,
        user_id: str = "default_user",
        limit: int = 5
    ) -> List[str]:
        """
        Retrieve memories using simple keyword matching.

        Args:
            query: Search query
            user_id: User identifier for filtering
            limit: Maximum number of results

        Returns:
            List of memory texts
        """
        query_lower = query.lower()
        results = []

        # Get memories for this user
        memory_ids = self.user_index.get(user_id, [])

        # Simple keyword matching
        for memory_id in memory_ids:
            memory = self.memories.get(memory_id)
            if memory and query_lower in memory["text"].lower():
                results.append(memory["text"])

                if len(results) >= limit:
                    break

        self.logger.debug(f"Retrieved {len(results)} memories")
        return results

    async def search(
        self,
        query: str,
        user_id: str = "default_user",
        limit: int = 5
    ) -> List[Dict]:
        """
        Search memories with keyword matching.

        Args:
            query: Search query
            user_id: User identifier for filtering
            limit: Maximum number of results

        Returns:
            List of dictionaries with content, metadata, and score
        """
        query_lower = query.lower()
        results = []

        # Get memories for this user
        memory_ids = self.user_index.get(user_id, [])

        # Simple keyword matching with score
        for memory_id in memory_ids:
            memory = self.memories.get(memory_id)
            if memory:
                text_lower = memory["text"].lower()

                # Simple scoring: count occurrences
                if query_lower in text_lower:
                    score = text_lower.count(query_lower)

                    results.append({
                        "content": memory["text"],
                        "metadata": memory["metadata"],
                        "distance": 1.0 / (score + 1)  # Lower is better
                    })

        # Sort by score (lower distance is better)
        results.sort(key=lambda x: x["distance"])
        results = results[:limit]

        self.logger.debug(f"Found {len(results)} memories")
        return results

    async def shutdown(self) -> None:
        """Clean up resources."""
        self.logger.info("Shutting down in-memory storage")
        self.memories.clear()
        self.user_index.clear()


__all__ = ["InMemory"]
