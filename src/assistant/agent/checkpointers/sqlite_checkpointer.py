"""SQLite-based checkpointer implementation."""

from pathlib import Path
from typing import Any, List
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from assistant.agent.checkpointers.base import BaseCheckpointer
from assistant.config.settings import Settings
from assistant.utils.logger import setup_logger


class SqliteCheckpointer(BaseCheckpointer):
    """
    SQLite-based checkpointer for persistent conversation state.

    Features:
    - Persistent storage on disk
    - Async operations
    - Efficient state management
    - Thread-based conversation tracking
    """

    def __init__(self, settings: Settings):
        """
        Initialize SQLite checkpointer.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.logger = setup_logger("assistant.agent.checkpointer.sqlite", settings.log_level)

        self.checkpointer = None

        # Ensure storage directory exists
        self._ensure_storage_directory()

    def _ensure_storage_directory(self):
        """Create checkpoint storage directory if it doesn't exist."""
        checkpoint_dir = Path(self.settings.checkpoint_db_path).parent
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"Checkpoint storage directory ensured: {checkpoint_dir}")

    async def initialize(self) -> None:
        """Initialize SQLite checkpointer."""
        self.logger.info(f"Initializing SQLite checkpointer: {self.settings.checkpoint_db_path}")

        try:
            # Create async SQLite checkpointer
            self.checkpointer = AsyncSqliteSaver.from_conn_string(
                self.settings.checkpoint_db_path
            )

            # Setup database schema
            await self.checkpointer.setup()

            self.logger.info("SQLite checkpointer initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize SQLite checkpointer: {e}", exc_info=True)
            raise

    def get_checkpointer(self) -> Any:
        """
        Get the underlying AsyncSqliteSaver instance.

        Returns:
            AsyncSqliteSaver instance
        """
        if not self.checkpointer:
            raise RuntimeError("Checkpointer not initialized")

        return self.checkpointer

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
        """
        if not self.checkpointer:
            raise RuntimeError("Checkpointer not initialized")

        try:
            config = {"configurable": {"thread_id": thread_id}}

            # Get the latest checkpoint
            checkpoint_tuple = await self.checkpointer.aget_tuple(config)

            if checkpoint_tuple and checkpoint_tuple.checkpoint:
                channel_values = checkpoint_tuple.checkpoint.get("channel_values", {})
                messages = channel_values.get("messages", [])
                result = messages[-limit:] if messages else []

                self.logger.debug(f"Retrieved {len(result)} messages for thread {thread_id}")
                return result

            return []

        except Exception as e:
            self.logger.error(f"Failed to retrieve history: {e}", exc_info=True)
            return []

    async def shutdown(self) -> None:
        """Clean up resources."""
        self.logger.info("Shutting down SQLite checkpointer")
        # AsyncSqliteSaver cleanup is handled automatically
        self.checkpointer = None


__all__ = ["SqliteCheckpointer"]
