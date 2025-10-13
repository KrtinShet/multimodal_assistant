"""ChromaDB-based memory implementation."""

from pathlib import Path
from typing import List, Dict, Optional
from uuid import uuid4
import chromadb
from chromadb.config import Settings as ChromaSettings

from assistant.agent.memory.base import BaseMemory
from assistant.config.settings import Settings
from assistant.utils.logger import setup_logger


class ChromaMemory(BaseMemory):
    """
    ChromaDB-based memory backend with persistent vector storage.

    Features:
    - Persistent storage on disk
    - Semantic search with cosine similarity
    - User-specific namespacing
    - Automatic embedding generation
    """

    def __init__(self, settings: Settings):
        """
        Initialize ChromaDB memory backend.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.logger = setup_logger("assistant.agent.memory.chroma", settings.log_level)

        self.client = None
        self.collection = None

        # Ensure storage directory exists
        self._ensure_storage_directory()

    def _ensure_storage_directory(self):
        """Create ChromaDB storage directory if it doesn't exist."""
        chroma_dir = Path(self.settings.chroma_db_path)
        chroma_dir.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"ChromaDB storage directory ensured: {chroma_dir}")

    async def initialize(self) -> None:
        """Initialize ChromaDB client and collection."""
        self.logger.info(f"Initializing ChromaDB memory: {self.settings.chroma_db_path}")

        try:
            # Initialize persistent ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.settings.chroma_db_path,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                )
            )

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.settings.memory_collection_name,
                metadata={"hnsw:space": "cosine"}
            )

            self.logger.info(f"ChromaDB collection ready: {self.collection.name}")

        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB memory: {e}", exc_info=True)
            raise

    async def store(
        self,
        memory_text: str,
        user_id: str = "default_user",
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Store a memory in ChromaDB.

        Args:
            memory_text: Text content to store
            user_id: User identifier for namespacing
            metadata: Additional metadata

        Returns:
            Memory ID
        """
        if not self.collection:
            raise RuntimeError("ChromaDB collection not initialized")

        try:
            memory_id = str(uuid4())

            # Build metadata
            mem_metadata = {"user_id": user_id, "type": "manual"}
            if metadata:
                mem_metadata.update(metadata)

            # Store in ChromaDB
            self.collection.add(
                documents=[memory_text],
                metadatas=[mem_metadata],
                ids=[memory_id]
            )

            self.logger.debug(f"Memory stored: {memory_id}")
            return memory_id

        except Exception as e:
            self.logger.error(f"Failed to store memory: {e}", exc_info=True)
            raise

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
        """
        if not self.collection:
            raise RuntimeError("ChromaDB collection not initialized")

        try:
            # Query ChromaDB
            results = self.collection.query(
                query_texts=[query],
                n_results=limit,
                where={"user_id": user_id} if user_id != "default_user" else None
            )

            # Extract documents
            if results["documents"] and results["documents"][0]:
                self.logger.debug(f"Retrieved {len(results['documents'][0])} memories")
                return results["documents"][0]

            return []

        except Exception as e:
            self.logger.error(f"Failed to retrieve memories: {e}", exc_info=True)
            return []

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
            List of dictionaries with content, metadata, and distance
        """
        if not self.collection:
            raise RuntimeError("ChromaDB collection not initialized")

        try:
            # Query ChromaDB
            results = self.collection.query(
                query_texts=[query],
                n_results=limit,
                where={"user_id": user_id} if user_id != "default_user" else None
            )

            # Format results
            memories = []
            if results["documents"] and results["documents"][0]:
                for doc, metadata, distance in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                ):
                    memories.append({
                        "content": doc,
                        "metadata": metadata,
                        "distance": distance
                    })

            self.logger.debug(f"Found {len(memories)} memories")
            return memories

        except Exception as e:
            self.logger.error(f"Failed to search memories: {e}", exc_info=True)
            return []

    async def shutdown(self) -> None:
        """Clean up resources."""
        self.logger.info("Shutting down ChromaDB memory")
        self.client = None
        self.collection = None


__all__ = ["ChromaMemory"]
