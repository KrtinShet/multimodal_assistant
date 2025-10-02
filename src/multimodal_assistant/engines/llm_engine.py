import ollama
from typing import AsyncIterator, Optional
from .base import ILLMEngine, VisionEmbedding
import asyncio
from multimodal_assistant.utils.logger import setup_logger

class OllamaLLMEngine(ILLMEngine):
    """Ollama LLM with streaming support"""

    def __init__(
        self,
        model_name: str = "gemma3:4b",
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
        self.client = None
        self.logger = setup_logger("multimodal_assistant.engines.llm")

    async def initialize(self):
        """Initialize Ollama client"""
        self.logger.info(f"Initializing Ollama LLM (model={self.model_name})")
        self.client = ollama.AsyncClient()

        # Ensure model is pulled
        try:
            await self.client.show(self.model_name)
            self.logger.info("Ollama LLM initialized successfully")
        except:
            self.logger.info(f"Pulling {self.model_name}...")
            await self.client.pull(self.model_name)
            self.logger.info("Ollama LLM initialized successfully")

    async def generate_stream(
        self,
        prompt: str,
        vision_embedding: Optional[VisionEmbedding] = None
    ) -> AsyncIterator[str]:
        """Stream generation with optional vision context"""

        # Add vision context if available
        if vision_embedding is not None:
            self.logger.debug(f"Generating response with vision context for: {prompt}")
            prompt = f"[User is showing you something via camera]\nUser: {prompt}"
        else:
            self.logger.debug(f"Generating response for: {prompt}")
            prompt = f"User: {prompt}"

        # Combine system prompt with user prompt
        full_prompt = f"{self.system_prompt}\n\n{prompt}\n\nVera:"

        # Stream tokens
        response = await self.client.generate(
            model=self.model_name,
            prompt=full_prompt,
            stream=True,
            options={
                "temperature": self.temperature,
                "top_p": self.top_p,
                "stop": ["\n\nUser:", "User:"]  # Stop at next user turn
            }
        )

        async for chunk in response:
            if 'response' in chunk:
                token = chunk['response']
                # Filter out any remaining control characters or backslashes
                if token and token not in ['\\', '\n\n\n']:  # Skip excessive formatting
                    yield token

    async def shutdown(self):
        """Cleanup"""
        self.client = None
