import asyncio
from typing import Optional

from assistant.core.streams import AsyncStream
from assistant.config.settings import Settings
from assistant.utils.logger import setup_logger


class TextInputHandler:
    """Handles text input from console/stdin.

    This handler provides a simple interface for capturing text input
    and streaming it to downstream consumers.
    """

    def __init__(self, settings: Settings):
        """Initialize text input handler.

        Args:
            settings: Application settings
        """
        self.logger = setup_logger("assistant.input.text", settings.log_level)
        self._running = False
        self._stream: Optional[AsyncStream[str]] = None

    async def start_capture(self) -> AsyncStream[str]:
        """Start capturing text input from console.

        Returns:
            AsyncStream yielding text strings
        """
        self.logger.info("Starting text input capture")
        self._stream = AsyncStream[str]()
        self._running = True

        # Start background task to capture input
        asyncio.create_task(self._capture_loop())

        return self._stream

    async def _capture_loop(self):
        """Background loop to capture text input."""
        loop = asyncio.get_running_loop()

        while self._running:
            try:
                # Run input() in executor to avoid blocking
                text = await loop.run_in_executor(
                    None,
                    lambda: input("You: ")
                )

                if text and self._stream:
                    await self._stream.put(text)
                    self.logger.debug(f"Captured text input: {text[:50]}...")

            except EOFError:
                self.logger.info("EOF received, stopping text input")
                break
            except Exception as e:
                self.logger.error(f"Error capturing text input: {e}")
                break

        await self.stop_capture()

    async def stop_capture(self):
        """Stop text input capture."""
        if self._running:
            self.logger.info("Stopping text input capture")
            self._running = False

            if self._stream:
                await self._stream.close()
                self._stream = None
