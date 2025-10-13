"""LLM-related graph nodes."""

from typing import Callable, Dict, List
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from assistant.agent.state import AgentState
from assistant.agent.llms.base import BaseLLM
from assistant.config.settings import Settings
from assistant.utils.logger import setup_logger


def create_call_model_node(
    llm: BaseLLM,
    settings: Settings
) -> Callable:
    """
    Create a node that calls the LLM with conversation history and memory context.

    Args:
        llm: LLM provider instance
        settings: Application settings

    Returns:
        Async function that calls the model
    """
    logger = setup_logger("assistant.agent.nodes.call_model", settings.log_level)

    async def call_model(state: AgentState, config: RunnableConfig) -> Dict:
        """Call the LLM with current state."""
        try:
            messages: List[BaseMessage] = list(state["messages"])
            memory_context = state.get("memory_context", "")

            # Build system message with memory context
            system_content = settings.system_prompt
            if memory_context:
                system_content += f"\n\nRelevant memories:\n{memory_context}"

            # Inject or update system message
            if not any(isinstance(m, SystemMessage) for m in messages):
                messages = [SystemMessage(content=system_content)] + messages
            else:
                # Update existing system message with memory context
                for i, msg in enumerate(messages):
                    if isinstance(msg, SystemMessage):
                        messages[i] = SystemMessage(content=system_content)
                        break

            logger.debug("Calling LLM with context...")

            # Get LLM instance and invoke
            llm_instance = llm.get_llm()
            response = await llm_instance.ainvoke(messages)

            return {"messages": [response]}

        except Exception as e:
            logger.error(f"Error calling LLM: {e}", exc_info=True)
            # Return error message
            from langchain_core.messages import AIMessage
            error_response = AIMessage(content="I encountered an error processing your request.")
            return {"messages": [error_response]}

    return call_model


__all__ = ["create_call_model_node"]
