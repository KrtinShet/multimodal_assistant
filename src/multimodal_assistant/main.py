import asyncio
import contextlib
import sys
from multimodal_assistant.core.event_bus import EventBus
from multimodal_assistant.engines.stt_engine import FasterWhisperEngine
from multimodal_assistant.engines.vision_engine import CLIPVisionEngine
from multimodal_assistant.engines.langgraph_engine import LangGraphAgentEngine
from multimodal_assistant.engines.tts_engine import KokoroTTSEngine
from multimodal_assistant.pipeline.input_handler import AudioInputHandler, VideoInputHandler
from multimodal_assistant.pipeline.coordinator import PipelineCoordinator
from multimodal_assistant.pipeline.output_handler import AudioOutputHandler, TextOutputHandler
from multimodal_assistant.config.settings import Settings
from multimodal_assistant.utils.logger import setup_logger
from multimodal_assistant.utils.performance import PerformanceMonitor
from multimodal_assistant.processors.webrtc_apm import WebRtcApmAEC as _MaybeApmAEC
from multimodal_assistant.processors.aec_core import IAEC

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
        # Use LangGraph agent instead of simple Ollama
        self.llm = LangGraphAgentEngine(
            model_name=self.settings.llm_model_name,
            system_prompt=self.settings.system_prompt,
            temperature=self.settings.temperature,
            top_p=self.settings.top_p
        )
        self.tts = KokoroTTSEngine()

        self.audio_input = AudioInputHandler(
            sample_rate=self.settings.audio_sample_rate,
            frame_duration_ms=self.settings.audio_frame_duration_ms,
            vad_aggressiveness=self.settings.vad_aggressiveness,
            enable_aec=self.settings.enable_aec,
            echo_suppression_strength=self.settings.echo_suppression_strength,
            echo_gate_correlation=self.settings.echo_gate_correlation,
            echo_gate_near_far_ratio=self.settings.echo_gate_near_far_ratio,
            input_device=self.settings.audio_input_device,
        )
        self.video_input = None
        self.video_enabled = False
        self.video_stream = None
        self.audio_enabled = True  # Audio starts enabled by default
        self.audio_output = AudioOutputHandler(
            sample_rate=self.settings.tts_sample_rate,
            frame_duration_ms=self.settings.playback_frame_duration_ms,
            reference_window_secs=self.settings.aec_reference_window_s,
            output_device=self.settings.audio_output_device,
        )
        self.text_output = TextOutputHandler()

        self.coordinator = PipelineCoordinator(
            stt_engine=self.stt,
            vision_engine=self.vision,
            llm_engine=self.llm,
            tts_engine=self.tts,
            event_bus=self.event_bus,
            audio_output=self.audio_output,
            text_output=self.text_output,
            perf_monitor=self.perf_monitor,
            audio_input=self.audio_input,
            settings=self.settings,
            audio_enabled_callback=lambda: self.audio_enabled,
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

        # Setup AEC backend wiring
        if self.settings.enable_aec and self.settings.aec_backend == "webrtc_apm":
            try:
                apm: IAEC = _MaybeApmAEC(
                    sample_rate_hz=self.settings.apm_sample_rate_hz,
                    frames_per_buffer=self.settings.apm_sample_rate_hz // 100,
                    enable_aec3=True,
                    enable_hpf=self.settings.apm_enable_hpf,
                    enable_ns=self.settings.apm_enable_ns,
                    enable_agc=self.settings.apm_enable_agc,
                    stream_delay_ms=self.settings.apm_stream_delay_ms,
                    lib_path=self.settings.apm_lib_path,
                )
                # Inject into input and have output push reverse frames
                self.audio_input.set_webrtc_apm(apm)
                self.audio_output.register_reverse_consumer(apm.push_reverse)
                self.logger.info("Using WebRTC APM for AEC")
            except Exception as e:
                self.logger.warning(
                    f"WebRTC APM not available ({e}); falling back to correlation AEC"
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

    async def enable_audio(self):
        """Enable audio output"""
        if self.audio_enabled:
            self.logger.info("Audio output already enabled")
            return

        self.logger.debug("Enabling audio output")
        self.audio_enabled = True
        self.logger.info("✓ Audio output enabled")

    async def disable_audio(self):
        """Disable audio output"""
        if not self.audio_enabled:
            self.logger.info("Audio output already disabled")
            return

        self.logger.debug("Disabling audio output")
        # Stop any active audio playback
        if self.audio_output:
            await self.audio_output.stop_playback()
        self.audio_enabled = False
        self.logger.info("✓ Audio output disabled")

    async def toggle_audio(self):
        """Toggle audio on/off"""
        self.logger.debug("Toggling audio")
        if self.audio_enabled:
            await self.disable_audio()
        else:
            await self.enable_audio()

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
                elif command.lower() == 'a':
                    await self.toggle_audio()
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
        self.logger.info("Commands: 'v' + Enter = toggle video | 'a' + Enter = toggle audio | 'q' + Enter = quit")

        # Start keyboard listener in background
        keyboard_task = asyncio.create_task(self._keyboard_listener())

        coordinator_task = asyncio.create_task(
            self.coordinator.run(audio_stream, self.video_stream)
        )

        try:
            await coordinator_task
        except (KeyboardInterrupt, asyncio.CancelledError):
            self.logger.info("\nStopping assistant...")
            coordinator_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await coordinator_task
        finally:
            keyboard_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await keyboard_task
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

async def _async_main():
    """Async entry point for the application."""
    assistant = Vera()
    await assistant.initialize()
    try:
        await assistant.run()
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nKeyboard interrupt received. Shutting down assistant...")
        await assistant.shutdown()

def main():
    """Synchronous CLI entrypoint.

    This wraps the async application entry in asyncio.run so console_scripts
    can invoke it without returning an un-awaited coroutine.
    """
    asyncio.run(_async_main())

if __name__ == "__main__":
    main()
