"""State definitions for the LangGraph agent."""

from typing import Annotated
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages


class AgentState(MessagesState):
    """
    Extended state for the LangGraph agent.

    Attributes:
        messages: Conversation history (inherited from MessagesState)
        memory_context: Retrieved memory context from vector store
        user_id: User identifier for memory namespacing
    """
    memory_context: str = ""
    user_id: str = "default_user"


__all__ = ["AgentState"]
