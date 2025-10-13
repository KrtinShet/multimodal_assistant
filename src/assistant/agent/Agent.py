"""Modular LangGraph agent with swappable components."""

from typing import AsyncIterator, Optional, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from assistant.agent.state import AgentState
from assistant.agent.memory.base import BaseMemory
from assistant.agent.memory.chroma_memory import ChromaMemory
from assistant.agent.memory.in_memory import InMemory
from assistant.agent.checkpointers.base import BaseCheckpointer
from assistant.agent.checkpointers.sqlite_checkpointer import SqliteCheckpointer
from assistant.agent.checkpointers.memory_checkpointer import MemoryCheckpointer
from assistant.agent.llms.base import BaseLLM
from assistant.agent.llms.ollama_llm import OllamaLLM
from assistant.agent.llms.openai_llm import OpenAILLM
from assistant.agent.graph import AgentGraphBuilder
from assistant.config.settings import Settings
from assistant.utils.logger import setup_logger


class Agent:
    """
    Modular LangGraph agent with swappable components.

    Features:
    - Pluggable LLM providers (Ollama, OpenAI)
    - Swappable memory backends (ChromaDB, InMemory)
    - Configurable checkpointers (SQLite, InMemory)
    - Human-in-the-loop capabilities
    - Persistent conversation state
    - Long-term semantic memory
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        llm: Optional[BaseLLM] = None,
        memory: Optional[BaseMemory] = None,
        checkpointer: Optional[BaseCheckpointer] = None
    ):
        """
        Initialize the agent with swappable components.

        Args:
            settings: Application settings
            llm: Custom LLM provider (optional, will be created from settings if not provided)
            memory: Custom memory backend (optional, will be created from settings if not provided)
            checkpointer: Custom checkpointer (optional, will be created from settings if not provided)
        """
        self.settings = settings or Settings()
        self.logger = setup_logger("assistant.agent", self.settings.log_level)

        # Components (injected or created from settings)
        self.llm = llm
        self.memory = memory
        self.checkpointer = checkpointer

        # Graph
        self.graph = None

    async def initialize(self):
        """Initialize all agent components and build the graph."""
        self.logger.info("Initializing Agent with modular components...")

        try:
            # Initialize LLM if not injected
            if not self.llm:
                self.llm = self._create_llm()
            await self.llm.initialize()

            # Initialize checkpointer if not injected
            if not self.checkpointer:
                self.checkpointer = self._create_checkpointer()
            await self.checkpointer.initialize()

            # Initialize memory if not injected and enabled
            if self.settings.enable_semantic_memory:
                if not self.memory:
                    self.memory = self._create_memory()
                await self.memory.initialize()

            # Build the graph
            await self._build_graph()

            self.logger.info("Agent initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Agent: {e}", exc_info=True)
            raise

    def _create_llm(self) -> BaseLLM:
        """Create LLM provider based on settings."""
        backend = self.settings.llm_backend or self.settings.agent_model_provider
        backend = backend.lower()

        self.logger.info(f"Creating LLM backend: {backend}")

        if backend == "openai":
            return OpenAILLM(self.settings)
        elif backend == "ollama":
            return OllamaLLM(self.settings)
        else:
            raise ValueError(f"Unsupported LLM backend: {backend}")

    def _create_memory(self) -> BaseMemory:
        """Create memory backend based on settings."""
        backend = self.settings.memory_backend.lower()

        self.logger.info(f"Creating memory backend: {backend}")

        if backend == "chroma":
            return ChromaMemory(self.settings)
        elif backend == "inmemory":
            return InMemory(self.settings)
        else:
            raise ValueError(f"Unsupported memory backend: {backend}")

    def _create_checkpointer(self) -> BaseCheckpointer:
        """Create checkpointer based on settings."""
        backend = self.settings.checkpointer_backend.lower()

        self.logger.info(f"Creating checkpointer backend: {backend}")

        if backend == "sqlite":
            return SqliteCheckpointer(self.settings)
        elif backend == "memory":
            return MemoryCheckpointer(self.settings)
        else:
            raise ValueError(f"Unsupported checkpointer backend: {backend}")

    async def _build_graph(self):
        """Build the agent graph using the graph builder."""
        self.logger.info("Building agent graph...")

        builder = AgentGraphBuilder(
            llm=self.llm,
            memory=self.memory if self.settings.enable_semantic_memory else None,
            checkpointer=self.checkpointer,
            settings=self.settings
        )

        self.graph = builder.build()
        self.logger.info("Agent graph built successfully")

    async def generate_stream(
        self,
        prompt: str,
        thread_id: str = "main_conversation",
        user_id: str = "default_user",
    ) -> AsyncIterator[str]:
        """
        Stream generation from the agent.

        Args:
            prompt: User's input prompt
            thread_id: Thread ID for conversation continuity
            user_id: User ID for memory namespacing

        Yields:
            Chunks of the assistant's response
        """
        if not self.graph:
            self.logger.error("Graph not initialized!")
            yield "Agent not ready. Please initialize first."
            return

        try:
            # Configuration with thread and user IDs
            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "user_id": user_id,
                }
            }

            # Initial state
            initial_state = {
                "messages": [HumanMessage(content=prompt)],
                "user_id": user_id,
            }

            self.logger.debug(f"Streaming response for thread: {thread_id}")

            # Stream the response
            async for event in self.graph.astream(
                initial_state,
                config=config,
                stream_mode="messages"
            ):
                # Extract content from AIMessage chunks
                if isinstance(event, tuple) and len(event) == 2:
                    message, _ = event
                    if isinstance(message, AIMessage) and message.content:
                        content = message.content
                        if content and content not in ['\\', '\n\n\n']:
                            yield content
                elif hasattr(event, 'content') and event.content:
                    content = event.content
                    if content and content not in ['\\', '\n\n\n']:
                        yield content

        except Exception as e:
            self.logger.error(f"Error in agent streaming: {e}", exc_info=True)
            yield "I encountered an error processing your request."

    async def invoke(
        self,
        prompt: str,
        thread_id: str = "main_conversation",
        user_id: str = "default_user",
    ) -> str:
        """
        Invoke the agent and get a complete response.

        Args:
            prompt: User's input prompt
            thread_id: Thread ID for conversation continuity
            user_id: User ID for memory namespacing

        Returns:
            Complete assistant response
        """
        response_parts = []
        async for chunk in self.generate_stream(prompt, thread_id, user_id):
            response_parts.append(chunk)

        return "".join(response_parts)

    async def get_conversation_history(
        self,
        thread_id: str = "main_conversation",
        limit: int = 10
    ) -> List[BaseMessage]:
        """
        Retrieve conversation history for a thread.

        Args:
            thread_id: Thread ID to retrieve history for
            limit: Maximum number of messages to retrieve

        Returns:
            List of messages from the conversation
        """
        if not self.checkpointer:
            self.logger.warning("Checkpointer not initialized")
            return []

        try:
            return await self.checkpointer.get_history(thread_id, limit)
        except Exception as e:
            self.logger.error(f"Error retrieving conversation history: {e}", exc_info=True)
            return []

    async def add_memory(
        self,
        memory_text: str,
        user_id: str = "default_user",
        metadata: Optional[dict] = None
    ):
        """
        Manually add a memory to the vector store.

        Args:
            memory_text: Text content of the memory
            user_id: User ID for namespacing
            metadata: Additional metadata for the memory
        """
        if not self.memory:
            self.logger.warning("Memory backend not initialized")
            return

        try:
            await self.memory.store(memory_text, user_id, metadata)
            self.logger.info("Memory added successfully")
        except Exception as e:
            self.logger.error(f"Error adding memory: {e}", exc_info=True)

    async def search_memories(
        self,
        query: str,
        user_id: str = "default_user",
        limit: int = 5
    ) -> List[dict]:
        """
        Search memories by semantic similarity.

        Args:
            query: Search query
            user_id: User ID for filtering
            limit: Maximum number of results

        Returns:
            List of memory dictionaries with content and metadata
        """
        if not self.memory:
            self.logger.warning("Memory backend not initialized")
            return []

        try:
            return await self.memory.search(query, user_id, limit)
        except Exception as e:
            self.logger.error(f"Error searching memories: {e}", exc_info=True)
            return []

    async def shutdown(self):
        """Cleanup resources."""
        self.logger.info("Shutting down Agent...")

        try:
            # Shutdown components
            if self.llm:
                await self.llm.shutdown()

            if self.memory:
                await self.memory.shutdown()

            if self.checkpointer:
                await self.checkpointer.shutdown()

            self.graph = None

            self.logger.info("Agent shutdown complete")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}", exc_info=True)


__all__ = ["Agent"]
