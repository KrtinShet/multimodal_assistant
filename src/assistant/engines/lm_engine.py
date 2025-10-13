from typing import AsyncIterator, Optional
from .base import ILLMEngine, VisionEmbedding
from assistant.config.settings import Settings
from assistant.utils.logger import setup_logger
import asyncio


class SimpleLLMEngine(ILLMEngine):
    """Simple Language Model engine implementation for testing"""

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize the LLM engine.

        Args:
            settings: Application settings
        """
        self.settings = settings or Settings()
        self.logger = setup_logger("assistant.engines.lm", self.settings.log_level)
        self._initialized = False
        self._model = None

    async def initialize(self):
        """Initialize the LLM engine."""
        self.logger.info("Initializing Simple LLM Engine")

        try:
            # For now, we'll use a simple mock implementation
            # In a real implementation, this would initialize an actual LLM
            self._model = "mock_model"
            self._initialized = True
            self.logger.info("Simple LLM Engine initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM Engine: {e}")
            raise

    async def generate_stream(
        self,
        prompt: str,
        vision_embedding: Optional[VisionEmbedding] = None
    ) -> AsyncIterator[str]:
        """Generate response stream from the LLM.

        Args:
            prompt: Input prompt for the LLM
            vision_embedding: Optional vision embedding for multimodal input

        Yields:
            Text tokens from the LLM response
        """
        if not self._initialized:
            self.logger.error("LLM Engine not initialized")
            yield "Error: LLM Engine not initialized"
            return

        self.logger.debug(f"Generating response for prompt: {prompt[:100]}...")

        try:
            # Simple mock response generation
            response = f"This is a mock response to: {prompt[:50]}..."

            # Simulate streaming by yielding character by character
            for char in response:
                yield char
                await asyncio.sleep(0.01)  # Simulate processing delay

        except Exception as e:
            self.logger.error(f"Error in generate_stream: {e}")
            yield "Error generating response"

    async def shutdown(self):
        """Shutdown the LLM engine."""
        self.logger.info("Shutting down Simple LLM Engine")

        try:
            self._model = None
            self._initialized = False
            self.logger.info("Simple LLM Engine shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


# Maintain backward compatibility
class LMEngine(SimpleLLMEngine):
    """Legacy LM Engine class - use SimpleLLMEngine instead"""
    pass