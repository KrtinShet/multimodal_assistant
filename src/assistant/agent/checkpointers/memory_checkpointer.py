"""In-memory checkpointer implementation for testing."""

from typing import Any, List
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import MemorySaver

from assistant.agent.checkpointers.base import BaseCheckpointer
from assistant.config.settings import Settings
from assistant.utils.logger import setup_logger


class MemoryCheckpointer(BaseCheckpointer):
    """
    In-memory checkpointer for testing and development.

    Features:
    - No external dependencies
    - Fast operation
    - Data lost on restart
    - Good for testing
    """

    def __init__(self, settings: Settings):
        """
        Initialize memory checkpointer.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.logger = setup_logger("assistant.agent.checkpointer.memory", settings.log_level)

        self.checkpointer = None

    async def initialize(self) -> None:
        """Initialize memory checkpointer."""
        self.logger.info("Initializing memory checkpointer")

        try:
            self.checkpointer = MemorySaver()
            self.logger.info("Memory checkpointer initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize memory checkpointer: {e}", exc_info=True)
            raise

    def get_checkpointer(self) -> Any:
        """
        Get the underlying MemorySaver instance.

        Returns:
            MemorySaver instance
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
            checkpoint_tuple = self.checkpointer.get_tuple(config)

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
        self.logger.info("Shutting down memory checkpointer")
        self.checkpointer = None


__all__ = ["MemoryCheckpointer"]
