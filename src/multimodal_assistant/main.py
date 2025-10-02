import asyncio
import sys
from multimodal_assistant.core.event_bus import EventBus
from multimodal_assistant.engines.stt_engine import FasterWhisperEngine
from multimodal_assistant.engines.vision_engine import CLIPVisionEngine
from multimodal_assistant.engines.llm_engine import OllamaLLMEngine
from multimodal_assistant.engines.tts_engine import KokoroTTSEngine
from multimodal_assistant.pipeline.input_handler import AudioInputHandler, VideoInputHandler
from multimodal_assistant.pipeline.coordinator import PipelineCoordinator
from multimodal_assistant.pipeline.output_handler import AudioOutputHandler, TextOutputHandler
from multimodal_assistant.config.settings import Settings
from multimodal_assistant.utils.logger import setup_logger
from multimodal_assistant.utils.performance import PerformanceMonitor

class Vera:
    """Main application class"""

    def __init__(self, settings: Settings = None):
        # Initialize settings
        self.settings = settings or Settings()

        # Setup logging based on settings
        # This creates the root logger for the entire application
        self.logger = setup_logger("multimodal_assistant", self.settings.log_level)

        # All child loggers will inherit this level automatically
        # Child loggers are created with names like:
        # - multimodal_assistant.engines.stt
        # - multimodal_assistant.engines.vision
        # - multimodal_assistant.coordinator
        # etc.

        self.logger.info(f"Initializing Vera Assistant (log_level={self.settings.log_level})")

        # Initialize performance monitor (enabled in DEBUG mode)
        self.perf_monitor = PerformanceMonitor() if self.settings.log_level.upper() == "DEBUG" else None

        # Initialize components
        self.event_bus = EventBus()
        self.stt = FasterWhisperEngine(model_size=self.settings.stt_model_size)
        self.vision = CLIPVisionEngine()
        self.llm = OllamaLLMEngine(
            model_name=self.settings.llm_model_name,
            system_prompt=self.settings.system_prompt,
            temperature=self.settings.temperature,
            top_p=self.settings.top_p
        )
        self.tts = KokoroTTSEngine()

        self.audio_input = AudioInputHandler(
            sample_rate=self.settings.audio_sample_rate
        )
        self.video_input = None
        self.video_enabled = False
        self.video_stream = None
        self.audio_output = AudioOutputHandler()
        self.text_output = TextOutputHandler()

        self.coordinator = PipelineCoordinator(
            stt_engine=self.stt,
            vision_engine=self.vision,
            llm_engine=self.llm,
            tts_engine=self.tts,
            event_bus=self.event_bus,
            audio_output=self.audio_output,
            text_output=self.text_output,
            perf_monitor=self.perf_monitor
        )

    async def initialize(self):
        """Initialize all engines"""
        self.logger.info("Starting engine initialization...")
        await self.event_bus.start()
        self.logger.debug("Event bus started")

        self.logger.debug("Initializing STT, Vision, LLM, and TTS engines...")
        await asyncio.gather(
            self.stt.initialize(),
            self.vision.initialize(),
            self.llm.initialize(),
            self.tts.initialize()
        )

        self.logger.info("✓ All engines ready!")

    async def enable_video(self):
        """Enable video input"""
        if self.video_enabled:
            self.logger.info("Video already enabled")
            return

        self.logger.debug(f"Initializing video input (fps={self.settings.video_fps})")
        if not self.video_input:
            self.video_input = VideoInputHandler(fps=self.settings.video_fps)

        self.video_stream = await self.video_input.start_capture()
        self.video_enabled = True
        self.logger.info("✓ Video enabled")

    async def disable_video(self):
        """Disable video input"""
        if not self.video_enabled:
            self.logger.info("Video already disabled")
            return

        self.logger.debug("Stopping video capture")
        if self.video_input:
            await self.video_input.stop_capture()
            self.video_stream = None
            self.video_enabled = False
            self.logger.info("✓ Video disabled")

    async def toggle_video(self):
        """Toggle video on/off"""
        self.logger.debug("Toggling video")
        if self.video_enabled:
            await self.disable_video()
        else:
            await self.enable_video()

    async def _keyboard_listener(self):
        """Listen for keyboard commands in the background"""
        loop = asyncio.get_event_loop()

        def read_input():
            return sys.stdin.readline().strip()

        while True:
            try:
                # Run the blocking readline in a thread pool
                command = await loop.run_in_executor(None, read_input)

                self.logger.debug(f"Keyboard command received: {command}")

                if command.lower() == 'v':
                    await self.toggle_video()
                elif command.lower() == 'q':
                    self.logger.info("Quit command received")
                    raise KeyboardInterrupt
            except asyncio.CancelledError:
                self.logger.debug("Keyboard listener cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in keyboard listener: {e}", exc_info=True)

    async def run(self):
        """Run the assistant"""
        self.logger.info("Starting assistant...")

        # Start input capture
        self.logger.debug("Starting audio input capture")
        audio_stream = await self.audio_input.start_capture()

        # Conditionally start video based on settings
        if self.settings.enable_vision:
            self.logger.debug("Auto-enabling video based on settings")
            await self.enable_video()

        self.logger.info("Listening... (Press Ctrl+C to stop)")
        self.logger.info("Commands: 'v' + Enter = toggle video | 'q' + Enter = quit")

        # Start keyboard listener in background
        keyboard_task = asyncio.create_task(self._keyboard_listener())

        try:
            while True:
                await self.coordinator.process_multimodal(
                    audio_stream,
                    self.video_stream
                )
                await asyncio.sleep(0.05) # 50ms
        except (KeyboardInterrupt, asyncio.CancelledError):
            self.logger.info("\nStopping assistant...")
        finally:
            keyboard_task.cancel()
            try:
                await keyboard_task
            except asyncio.CancelledError:
                pass
            await self.shutdown()

    async def shutdown(self):
        """Cleanup"""
        self.logger.info("Shutting down...")

        self.logger.debug("Stopping input handlers")
        await self.audio_input.stop_capture()
        if self.video_input:
            await self.video_input.stop_capture()
        await self.audio_output.stop_playback()

        self.logger.debug("Shutting down engines")
        await asyncio.gather(
            self.stt.shutdown(),
            self.vision.shutdown(),
            self.llm.shutdown(),
            self.tts.shutdown()
        )

        self.logger.debug("Stopping event bus")
        await self.event_bus.stop()

        # Print performance summary if debug mode
        if self.perf_monitor:
            self.logger.info("\n" + "="*50)
            self.perf_monitor.print_summary()
            self.logger.info("="*50)

        self.logger.info("✓ Shutdown complete")

async def main():
    assistant = Vera()
    await assistant.initialize()
    try:
        await assistant.run()
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nKeyboard interrupt received. Shutting down assistant...")
        await assistant.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
