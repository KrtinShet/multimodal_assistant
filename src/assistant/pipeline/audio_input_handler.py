import asyncio
from collections.abc import Callable
from typing import Optional

import numpy as np
import sounddevice as sd

from assistant.core.streams import AsyncStream
from assistant.engines.base import AudioChunk
from assistant.config.settings import Settings
from assistant.utils.logger import setup_logger
from assistant.processors.vad import ImprovedVADProcessor


class AudioInputHandler:
    """Handles microphone input with advanced VAD and echo cancellation support.

    Features improved Voice Activity Detection with:
    - WebRTC VAD integration
    - Adaptive noise floor estimation
    - Multi-stage speech detection
    - Smoothing for stable detection
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

        # Initialize improved VAD processor
        self.vad_processor = ImprovedVADProcessor(
            aggressiveness=settings.vad_aggressiveness,
            sample_rate=self.sample_rate,
            energy_threshold_ratio=settings.vad_energy_threshold_ratio,
            noise_floor_adaptation_rate=settings.vad_noise_floor_adaptation_rate,
            smoothing_window=settings.vad_smoothing_window,
            min_energy_threshold=settings.noise_rms_threshold,
            log_level=settings.log_level,
        )

        self._running = False
        self._frame_count = 0
        self._last_metrics_log = 0.0

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
        """Process audio frame with advanced VAD and push to downstream consumers.

        Args:
            audio_data: Raw audio samples
            stream: Output stream to push processed audio
        """
        if not self._running:
            return

        # Apply improved VAD processing
        processed = audio_data.copy()
        is_speech = self.vad_processor.is_speech(processed)

        chunk = AudioChunk(
            data=processed,
            sample_rate=self.sample_rate,
            timestamp=asyncio.get_event_loop().time(),
            is_speech=is_speech,
        )

        await stream.put(chunk)

        # Periodic metrics logging (every 5 seconds)
        self._frame_count += 1
        current_time = asyncio.get_event_loop().time()
        if current_time - self._last_metrics_log >= 5.0:
            metrics = self.vad_processor.get_metrics()
            self.logger.debug(
                f"VAD Metrics: speech_ratio={metrics.speech_ratio:.2%}, "
                f"noise_floor={metrics.current_noise_floor:.4f}, "
                f"threshold={metrics.energy_threshold:.4f}, "
                f"frames={metrics.total_frames}"
            )
            self._last_metrics_log = current_time

    async def stop_capture(self):
        """Stop audio capture and release resources."""
        if self.stream:
            # Log final VAD metrics
            metrics = self.vad_processor.get_metrics()
            self.logger.info(
                f"Stopping audio capture - Final VAD stats: "
                f"speech_ratio={metrics.speech_ratio:.2%}, "
                f"total_frames={metrics.total_frames}"
            )

            self._running = False
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def get_vad_metrics(self):
        """Get current VAD metrics for monitoring.

        Returns:
            VADMetrics object with current statistics
        """
        return self.vad_processor.get_metrics()

    def reset_vad(self):
        """Reset VAD state and metrics."""
        self.vad_processor.reset()
        self._frame_count = 0
        self._last_metrics_log = 0.0
        self.logger.info("VAD state reset")
