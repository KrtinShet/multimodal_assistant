import ollama
from typing import AsyncIterator, Optional
from .base import ILLMEngine, VisionEmbedding
import asyncio

class OllamaLLMEngine(ILLMEngine):
    """Ollama LLM with streaming support"""

    def __init__(self, model_name: str = "gemma3:4b"):
        self.model_name = model_name
        self.client = None

    async def initialize(self):
        """Initialize Ollama client"""
        self.client = ollama.AsyncClient()

        # Ensure model is pulled
        try:
            await self.client.show(self.model_name)
        except:
            print(f"Pulling {self.model_name}...")
            await self.client.pull(self.model_name)

    async def generate_stream(
        self,
        prompt: str,
        vision_embedding: Optional[VisionEmbedding] = None
    ) -> AsyncIterator[str]:
        """Stream generation with optional vision context"""

        # Add vision context if available
        if vision_embedding is not None:
            # Convert embedding to text description (simplified)
            prompt = f"[Image context provided]\n{prompt}"

        # Stream tokens
        response = await self.client.generate(
            model=self.model_name,
            prompt=prompt,
            stream=True
        )

        async for chunk in response:
            if 'response' in chunk:
                yield chunk['response']

    async def shutdown(self):
        """Cleanup"""
        self.client = None
