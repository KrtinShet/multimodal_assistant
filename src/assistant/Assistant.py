import contextlib
from assistant.config.settings import Settings
from assistant.utils.logger import setup_logger
from assistant.utils.performance import PerformanceMonitor
import asyncio

class Assistant:
    """
    The Assistant class is the main class for the assistant.
    It is responsible for running the assistant and shutting it down.
    """
    def __init__(self, settings: Settings = None):
        """
        The init method is responsible for initializing the assistant.
        """
        self.settings = settings or Settings()
        self.logger = setup_logger("assistant", self.settings.log_level)
        self.perf_monitor = PerformanceMonitor() if self.settings.log_level.upper() == "DEBUG" else None
   
    async def initialize(self):
        """
        The initialize method is responsible for initializing the assistant.
        """
        self.logger.info("Starting engine initialization...")
        await asyncio.gather(
            self.stt.initialize(),
            self.vision.initialize(),
            self.llm.initialize(),
            self.tts.initialize()
        )
        self.logger.info("All engines initialized successfully")
   
    async def run(self):
        """
        The run method is the main method for the assistant.
        It is responsible for running the assistant.
        """
        self.logger.info("Starting assistant...")

        # Start keyboard listener in background
        keyboard_task = asyncio.create_task(self._keyboard_listener())

        coordinator_task = asyncio.create_task(
            self.coordinator.run(self.audio_stream, self.video_stream)
        )

        try:
            pass
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            keyboard_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await keyboard_task
            await self.shutdown()

    async def shutdown(self):
        """
        The shutdown method is responsible for shutting down the assistant.
        """
        self.logger.info("Shutting down assistant...")
        await asyncio.gather(
            self.stt.shutdown(),
            self.vision.shutdown(),
            self.llm.shutdown(),
            self.tts.shutdown()
        )
        self.logger.info("All engines shutdown successfully")

    async def handle_event(self, event):
        """
        The handle_event method is responsible for handling events.
        """
        pass