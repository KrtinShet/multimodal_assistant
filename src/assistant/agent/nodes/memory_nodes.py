"""Memory-related graph nodes."""

from typing import Callable, Dict
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from assistant.agent.state import AgentState
from assistant.agent.memory.base import BaseMemory
from assistant.config.settings import Settings
from assistant.utils.logger import setup_logger


def create_retrieve_memories_node(
    memory: BaseMemory,
    settings: Settings
) -> Callable:
    """
    Create a node that retrieves relevant memories from the memory backend.

    Args:
        memory: Memory backend instance
        settings: Application settings

    Returns:
        Async function that retrieves memories
    """
    logger = setup_logger("assistant.agent.nodes.retrieve_memories", settings.log_level)

    async def retrieve_memories(state: AgentState, config: RunnableConfig) -> Dict:
        """Retrieve relevant memories from the memory backend."""
        if not settings.enable_semantic_memory or not memory:
            return {"memory_context": ""}

        try:
            # Get the latest user message
            messages = state["messages"]
            if not messages:
                return {"memory_context": ""}

            last_message = messages[-1]
            if not isinstance(last_message, HumanMessage):
                return {"memory_context": ""}

            query_text = last_message.content
            user_id = state.get("user_id", "default_user")

            logger.debug(f"Retrieving memories for query: {query_text[:50]}...")

            # Retrieve memories
            memories = await memory.retrieve(
                query=query_text,
                user_id=user_id,
                limit=settings.memory_retrieval_limit
            )

            # Format memories into context
            memory_context = ""
            if memories:
                memory_context = "\n".join([f"- {mem}" for mem in memories])
                logger.debug(f"Retrieved {len(memories)} relevant memories")

            return {"memory_context": memory_context}

        except Exception as e:
            logger.error(f"Error retrieving memories: {e}", exc_info=True)
            return {"memory_context": ""}

    return retrieve_memories


def create_store_memories_node(
    memory: BaseMemory,
    settings: Settings
) -> Callable:
    """
    Create a node that stores important information as memories.

    Args:
        memory: Memory backend instance
        settings: Application settings

    Returns:
        Async function that stores memories
    """
    logger = setup_logger("assistant.agent.nodes.store_memories", settings.log_level)

    async def store_memories(state: AgentState, config: RunnableConfig) -> Dict:
        """Store important information as memories."""
        if not settings.enable_semantic_memory or not memory:
            return {}

        try:
            from langchain_core.messages import AIMessage, HumanMessage

            messages = state["messages"]
            if len(messages) < 2:
                return {}

            # Get the last exchange (user message + assistant response)
            last_user_msg = None
            last_ai_msg = None

            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and not last_ai_msg:
                    last_ai_msg = msg
                elif isinstance(msg, HumanMessage) and not last_user_msg:
                    last_user_msg = msg

                if last_user_msg and last_ai_msg:
                    break

            if not (last_user_msg and last_ai_msg):
                return {}

            user_id = state.get("user_id", "default_user")

            # Check if the conversation contains information worth remembering
            keywords = ["remember", "my name is", "i am", "i like", "i prefer", "important"]
            user_content_lower = last_user_msg.content.lower()

            if any(keyword in user_content_lower for keyword in keywords):
                # Store this exchange as a memory
                memory_text = f"User: {last_user_msg.content}\nAssistant: {last_ai_msg.content}"

                logger.debug("Storing new memory")

                await memory.store(
                    memory_text=memory_text,
                    user_id=user_id,
                    metadata={"type": "conversation"}
                )

                logger.info("Memory stored successfully")

            return {}

        except Exception as e:
            logger.error(f"Error storing memories: {e}", exc_info=True)
            return {}

    return store_memories


__all__ = ["create_retrieve_memories_node", "create_store_memories_node"]
