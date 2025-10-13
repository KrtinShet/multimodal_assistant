"""Modular agent module with swappable components."""

# Main agent
from assistant.agent.Agent import Agent

# State
from assistant.agent.state import AgentState

# Graph builder
from assistant.agent.graph import AgentGraphBuilder

# Memory backends
from assistant.agent.memory import BaseMemory, ChromaMemory, InMemory

# Checkpointer backends
from assistant.agent.checkpointers import BaseCheckpointer, SqliteCheckpointer, MemoryCheckpointer

# LLM providers
from assistant.agent.llms import BaseLLM, OllamaLLM, OpenAILLM

# Tools
from assistant.agent.tools import BaseTool

# Nodes (for custom graph building)
from assistant.agent.nodes import (
    create_retrieve_memories_node,
    create_store_memories_node,
    create_call_model_node
)

__all__ = [
    # Main agent
    "Agent",
    "AgentState",
    "AgentGraphBuilder",
    # Memory
    "BaseMemory",
    "ChromaMemory",
    "InMemory",
    # Checkpointers
    "BaseCheckpointer",
    "SqliteCheckpointer",
    "MemoryCheckpointer",
    # LLMs
    "BaseLLM",
    "OllamaLLM",
    "OpenAILLM",
    # Tools
    "BaseTool",
    # Nodes
    "create_retrieve_memories_node",
    "create_store_memories_node",
    "create_call_model_node",
]
