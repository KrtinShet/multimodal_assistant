"""Graph builder for the agent."""

from typing import Optional, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from assistant.agent.state import AgentState
from assistant.agent.memory.base import BaseMemory
from assistant.agent.checkpointers.base import BaseCheckpointer
from assistant.agent.llms.base import BaseLLM
from assistant.agent.nodes import (
    create_retrieve_memories_node,
    create_store_memories_node,
    create_call_model_node
)
from assistant.config.settings import Settings
from assistant.utils.logger import setup_logger


class AgentGraphBuilder:
    """
    Builder for creating LangGraph agent graphs with modular components.

    Features:
    - Composable graph structure
    - Pluggable memory, checkpointer, and LLM backends
    - Extensible with custom nodes and edges
    - Human-in-the-loop support
    """

    def __init__(
        self,
        llm: BaseLLM,
        memory: Optional[BaseMemory],
        checkpointer: BaseCheckpointer,
        settings: Settings
    ):
        """
        Initialize the graph builder.

        Args:
            llm: LLM provider instance
            memory: Memory backend instance (optional)
            checkpointer: Checkpointer backend instance
            settings: Application settings
        """
        self.llm = llm
        self.memory = memory
        self.checkpointer = checkpointer
        self.settings = settings

        self.logger = setup_logger("assistant.agent.graph", settings.log_level)

        self.builder = None
        self.custom_nodes = []
        self.custom_edges = []

    def build(self) -> CompiledStateGraph:
        """
        Build and compile the agent graph.

        Returns:
            Compiled LangGraph instance

        Raises:
            RuntimeError: If build fails
        """
        try:
            self.logger.info("Building agent graph...")

            # Initialize graph builder
            self.builder = StateGraph(AgentState)

            # Add core nodes
            self._add_core_nodes()

            # Add custom nodes
            self._add_custom_nodes()

            # Define core edges
            self._define_core_edges()

            # Add custom edges
            self._add_custom_edges()

            # Compile with checkpointer
            graph = self.builder.compile(
                checkpointer=self.checkpointer.get_checkpointer()
            )

            self.logger.info("Agent graph compiled successfully")
            return graph

        except Exception as e:
            self.logger.error(f"Failed to build graph: {e}", exc_info=True)
            raise RuntimeError(f"Graph build failed: {e}")

    def _add_core_nodes(self):
        """Add core nodes to the graph."""
        # Memory retrieval node
        if self.settings.enable_semantic_memory and self.memory:
            retrieve_memories = create_retrieve_memories_node(self.memory, self.settings)
            self.builder.add_node("retrieve_memories", retrieve_memories)
            self.logger.debug("Added retrieve_memories node")

        # LLM call node
        call_model = create_call_model_node(self.llm, self.settings)
        self.builder.add_node("call_model", call_model)
        self.logger.debug("Added call_model node")

        # Memory storage node
        if self.settings.enable_semantic_memory and self.memory:
            store_memories = create_store_memories_node(self.memory, self.settings)
            self.builder.add_node("store_memories", store_memories)
            self.logger.debug("Added store_memories node")

    def _define_core_edges(self):
        """Define core edges in the graph."""
        if self.settings.enable_semantic_memory and self.memory:
            # Flow: START -> retrieve_memories -> call_model -> store_memories -> END
            self.builder.add_edge(START, "retrieve_memories")
            self.builder.add_edge("retrieve_memories", "call_model")
            self.builder.add_edge("call_model", "store_memories")
            self.builder.add_edge("store_memories", END)
        else:
            # Simplified flow without memory: START -> call_model -> END
            self.builder.add_edge(START, "call_model")
            self.builder.add_edge("call_model", END)

        self.logger.debug("Core edges defined")

    def add_node(self, name: str, func):
        """
        Add a custom node to the graph.

        Args:
            name: Node name
            func: Node function

        Returns:
            Self for method chaining
        """
        self.custom_nodes.append((name, func))
        self.logger.debug(f"Custom node registered: {name}")
        return self

    def add_edge(self, from_node: str, to_node: str):
        """
        Add a custom edge to the graph.

        Args:
            from_node: Source node name
            to_node: Destination node name

        Returns:
            Self for method chaining
        """
        self.custom_edges.append((from_node, to_node))
        self.logger.debug(f"Custom edge registered: {from_node} -> {to_node}")
        return self

    def _add_custom_nodes(self):
        """Add all custom nodes to the graph."""
        for name, func in self.custom_nodes:
            self.builder.add_node(name, func)
            self.logger.debug(f"Added custom node: {name}")

    def _add_custom_edges(self):
        """Add all custom edges to the graph."""
        for from_node, to_node in self.custom_edges:
            self.builder.add_edge(from_node, to_node)
            self.logger.debug(f"Added custom edge: {from_node} -> {to_node}")


__all__ = ["AgentGraphBuilder"]
