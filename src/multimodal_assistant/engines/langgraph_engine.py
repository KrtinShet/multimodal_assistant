"""LangGraph-based agent engine for multimodal assistant."""
import asyncio
from typing import AsyncIterator, Optional, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
from .base import ILLMEngine, VisionEmbedding
from multimodal_assistant.utils.logger import setup_logger


class AgentState(TypedDict):
    """State for the LangGraph agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]


class LangGraphAgentEngine(ILLMEngine):
    """LangGraph-based agent with conversational state management."""

    def __init__(
        self,
        model_name: str = "gemma2:2b",
        system_prompt: str = None,
        temperature: float = 0.7,
        top_p: float = 0.9
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt or (
            "You are Vera, a helpful voice assistant. "
            "Respond naturally and conversationally. "
            "Keep responses concise and clear. "
            "Do not use markdown formatting, emojis, or special characters. "
            "Speak as if having a natural conversation."
        )
        self.temperature = temperature
        self.top_p = top_p
        self.llm = None
        self.graph = None
        self.checkpointer = None
        self.thread_id = "main_conversation"
        self.logger = setup_logger("multimodal_assistant.engines.langgraph")

    async def initialize(self):
        """Initialize LangGraph agent with Ollama LLM."""
        self.logger.info(f"Initializing LangGraph Agent (model={self.model_name})")

        # Initialize Ollama LLM
        self.llm = ChatOllama(
            model=self.model_name,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        # Create memory saver for conversation history
        self.checkpointer = MemorySaver()

        # Build the agent graph
        await self._build_graph()

        self.logger.info("LangGraph Agent initialized successfully")

    async def _build_graph(self):
        """Build the LangGraph state graph."""

        async def call_model(state: AgentState) -> AgentState:
            """Node that calls the LLM with current conversation state."""
            # Add system message if this is the start of conversation
            messages = state["messages"]

            # Inject system prompt if not already present
            if not any(isinstance(m, SystemMessage) for m in messages):
                messages = [SystemMessage(content=self.system_prompt)] + list(messages)

            # Call the LLM
            response = await self.llm.ainvoke(messages)

            return {"messages": [response]}

        # Build the graph
        builder = StateGraph(AgentState)
        builder.add_node("call_model", call_model)
        builder.add_edge(START, "call_model")
        builder.add_edge("call_model", END)

        # Compile with checkpointer for memory
        self.graph = builder.compile(checkpointer=self.checkpointer)

        self.logger.debug("LangGraph state graph compiled successfully")

    async def generate_stream(
        self,
        prompt: str,
        vision_embedding: Optional[VisionEmbedding] = None
    ) -> AsyncIterator[str]:
        """Stream generation using LangGraph agent."""

        # Prepare the user message
        if vision_embedding is not None:
            self.logger.debug(f"Generating response with vision context for: {prompt}")
            user_message = f"[User is showing you something via camera]\n{prompt}"
        else:
            self.logger.debug(f"Generating response for: {prompt}")
            user_message = prompt

        # Create configuration with thread ID for conversation continuity
        config = {
            "configurable": {
                "thread_id": self.thread_id
            }
        }

        # Stream the response from the graph
        try:
            async for event in self.graph.astream(
                {"messages": [HumanMessage(content=user_message)]},
                config=config,
                stream_mode="messages"
            ):
                # Extract content from AIMessage chunks
                if isinstance(event, tuple) and len(event) == 2:
                    message, metadata = event
                    if isinstance(message, AIMessage) and message.content:
                        # Stream the content token by token
                        content = message.content
                        if content and content not in ['\\', '\n\n\n']:
                            yield content
                elif hasattr(event, 'content') and event.content:
                    # Direct message streaming
                    content = event.content
                    if content and content not in ['\\', '\n\n\n']:
                        yield content

        except Exception as e:
            self.logger.error(f"Error in LangGraph streaming: {e}", exc_info=True)
            yield "I apologize, but I encountered an error processing your request."

    async def shutdown(self):
        """Cleanup resources."""
        self.logger.info("Shutting down LangGraph Agent")
        self.llm = None
        self.graph = None
        self.checkpointer = None
