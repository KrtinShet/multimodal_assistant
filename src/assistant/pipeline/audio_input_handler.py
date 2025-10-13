import asyncio
from collections.abc import Callable
from typing import Optional

import numpy as np
import sounddevice as sd

from assistant.core.streams import AsyncStream
from assistant.engines.base import AudioChunk
from assistant.config.settings import Settings
from assistant.utils.logger import setup_logger


class AudioInputHandler:
    """Handles microphone input with basic VAD and echo cancellation support.

    This is a simplified implementation providing core audio capture functionality.
    For production use, integrate VAD processors and AEC modules.
    """

    def __init__(
        self,
        settings: Settings,
    ):
        """Initialize audio input handler.

        Args:
            settings: Application settings containing audio configuration
        """
        self.sample_rate = settings.audio_sample_rate
        self.frame_duration_ms = settings.audio_frame_duration_ms
        self.chunk_size = int(self.sample_rate * (self.frame_duration_ms / 1000.0))
        self.input_device = settings.audio_input_device

        self.stream: Optional[sd.InputStream] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.logger = setup_logger("assistant.input.audio", settings.log_level)
        self.reference_provider: Optional[Callable[[int], np.ndarray]] = None

        self._running = False

    def set_reference_provider(
        self,
        provider: Callable[[int], np.ndarray],
    ):
        """Register a callable that supplies far-end reference audio for AEC.

        Args:
            provider: Function that returns reference audio samples
        """
        self.reference_provider = provider

    async def start_capture(self) -> AsyncStream[AudioChunk]:
        """Start audio capture stream.

        Returns:
            AsyncStream yielding AudioChunk objects
        """
        self.logger.info(
            f"Starting audio capture (sample_rate={self.sample_rate}, "
            f"chunk_size={self.chunk_size})"
        )
        output_stream = AsyncStream[AudioChunk]()

        self.loop = asyncio.get_running_loop()
        self._running = True

        def audio_callback(indata, frames, _time, status):
            if status:
                self.logger.error(f"Audio input status: {status}")

            # Convert to mono float32
            audio_data = indata[:, 0].copy()

            # Process audio in async context
            asyncio.run_coroutine_threadsafe(
                self._process_audio(audio_data, output_stream),
                self.loop,
            )

        stream_kwargs = {
            "samplerate": self.sample_rate,
            "channels": 1,
            "blocksize": self.chunk_size,
            "dtype": "float32",
            "callback": audio_callback,
        }

        if self.input_device is not None:
            stream_kwargs["device"] = self.input_device

        self.stream = sd.InputStream(**stream_kwargs)
        self.stream.start()
        self.logger.info("Audio capture started")

        return output_stream

    async def _process_audio(self, audio_data: np.ndarray, stream: AsyncStream):
        """Process audio frame and push to downstream consumers.

        Args:
            audio_data: Raw audio samples
            stream: Output stream to push processed audio
        """
        if not self._running:
            return

        # Basic processing - in production, add VAD and AEC here
        processed = audio_data.copy()

        # Simple energy-based speech detection
        energy = np.sqrt(np.mean(processed ** 2))
        is_speech = energy > 0.01  # Basic threshold

        chunk = AudioChunk(
            data=processed,
            sample_rate=self.sample_rate,
            timestamp=asyncio.get_event_loop().time(),
            is_speech=is_speech,
        )

        await stream.put(chunk)

    async def stop_capture(self):
        """Stop audio capture and release resources."""
        if self.stream:
            self.logger.info("Stopping audio capture")
            self._running = False
            self.stream.stop()
            self.stream.close()
            self.stream = None
