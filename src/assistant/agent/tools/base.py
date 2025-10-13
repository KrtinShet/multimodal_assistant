"""Abstract base class for tools with human-in-the-loop support."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseTool(ABC):
    """
    Abstract interface for tools that can be used by the agent.

    Tools can require human approval before execution, enabling
    human-in-the-loop workflows for sensitive operations.
    """

    def __init__(self, requires_approval: bool = False):
        """
        Initialize the tool.

        Args:
            requires_approval: Whether this tool requires human approval before execution
        """
        self.requires_approval = requires_approval

    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        """
        Execute the tool with the given arguments.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Tool execution result

        Raises:
            Exception: If execution fails
        """
        pass

    @abstractmethod
    def get_approval_prompt(self, *args, **kwargs) -> str:
        """
        Get a human-readable prompt for approval.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Approval prompt string
        """
        pass

    @abstractmethod
    def get_schema(self) -> Dict:
        """
        Get the tool schema for LangChain integration.

        Returns:
            Dictionary containing tool name, description, and parameters
        """
        pass

    async def request_approval(self, *args, **kwargs) -> bool:
        """
        Request human approval for tool execution.

        This can be overridden to implement custom approval logic.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            True if approved, False otherwise
        """
        # Default implementation - always approve
        # Override this in subclasses for actual approval logic
        return True


__all__ = ["BaseTool"]
