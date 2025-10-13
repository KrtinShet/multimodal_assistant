import asyncio
from collections import deque
from typing import AsyncIterator, Optional
import threading

import numpy as np
import sounddevice as sd

from assistant.engines.base import AudioChunk
from assistant.config.settings import Settings
from assistant.utils.logger import setup_logger


class AudioOutputHandler:
    """Handles low-latency audio playback with pause/resume support.

    This implementation provides buffered audio output with reference audio
    tracking for echo cancellation and playback control features.
    """

    def __init__(
        self,
        settings: Settings,
    ):
        """Initialize audio output handler.

        Args:
            settings: Application settings containing audio configuration
        """
        self.sample_rate = settings.tts_sample_rate
        self.frame_duration_ms = settings.playback_frame_duration_ms
        self.frame_size = int(self.sample_rate * (self.frame_duration_ms / 1000.0))
        self.reference_window_secs = settings.aec_reference_window_s
        self.output_device = settings.audio_output_device

        self._stream: Optional[sd.OutputStream] = None
        self._buffer: deque[np.ndarray] = deque()
        self._reference_buffer: deque[np.ndarray] = deque(
            maxlen=int(
                max(1, self.reference_window_secs * 1000 / self.frame_duration_ms)
            )
        )

        self._buffer_lock = threading.Lock()
        self._paused = threading.Event()
        self._paused.clear()
        self._running = threading.Event()
        self._running.clear()

        self.logger = setup_logger("assistant.output.audio", settings.log_level)
        self._reverse_consumer: Optional[callable] = None

    async def start_playback(self, audio_stream: AsyncIterator[AudioChunk]):
        """Consume an audio chunk stream and enqueue frames for playback.

        Args:
            audio_stream: Async iterator yielding AudioChunk objects
        """
        self.logger.debug("start_playback() called, ensuring output stream...")
        await self._ensure_stream()
        self.logger.debug("Output stream ready, consuming audio chunks...")

        chunk_count = 0
        try:
            async for chunk in audio_stream:
                chunk_count += 1
                self.logger.debug(
                    f"Received audio chunk #{chunk_count}: "
                    f"{chunk.data.shape if chunk and chunk.data is not None else None}"
                )
                await self.enqueue_chunk(chunk)
        except asyncio.CancelledError:
            self.logger.debug(
                f"Audio playback task cancelled after {chunk_count} chunks; "
                f"stopping stream"
            )
            await self.stop_playback()
            raise
        finally:
            self.logger.debug(f"Audio playback finished, total chunks: {chunk_count}")

    async def enqueue_chunk(self, chunk: AudioChunk):
        """Slice an AudioChunk into frames and enqueue for playback.

        Args:
            chunk: Audio chunk to enqueue
        """
        if chunk is None or chunk.data is None:
            return

        if chunk.sample_rate != self.sample_rate:
            raise ValueError(
                f"Audio chunk sample rate {chunk.sample_rate} "
                f"does not match output rate {self.sample_rate}"
            )

        data = chunk.data.astype(np.float32, copy=False)
        if data.ndim > 1:
            data = data[:, 0]

        frames = self._split_frames(data)

        with self._buffer_lock:
            for frame in frames:
                self._buffer.append(frame)

    def register_reverse_consumer(self, fn: callable):
        """Register a callable to receive each playback frame for AEC.

        The consumer receives mono float32 numpy arrays for echo cancellation.

        Args:
            fn: Function to receive playback frames
        """
        self._reverse_consumer = fn

    async def pause(self):
        """Pause playback (audio stream keeps buffering)."""
        self.logger.debug("Pausing audio playback")
        self._paused.set()

    async def resume(self):
        """Resume playback after pause."""
        self.logger.debug("Resuming audio playback")
        self._paused.clear()

    async def flush(self):
        """Clear buffered frames (used when cancelling responses)."""
        self.logger.debug("Flushing audio buffer")
        with self._buffer_lock:
            self._buffer.clear()

    async def stop_playback(self):
        """Completely stop playback and release the output stream."""
        if not self._running.is_set():
            return

        self.logger.debug("Stopping audio output stream")
        self._running.clear()
        try:
            if self._stream:
                self._stream.stop()
                self._stream.close()
        finally:
            self._stream = None
            with self._buffer_lock:
                self._buffer.clear()

    def is_playing(self) -> bool:
        """Return True if audio is currently playing.

        Returns:
            True if buffer not empty and not paused
        """
        with self._buffer_lock:
            return bool(self._buffer) and not self._paused.is_set()

    def get_reference_audio(self, num_samples: int) -> np.ndarray:
        """Return the latest far-end audio samples for echo cancellation.

        Args:
            num_samples: Number of samples to retrieve

        Returns:
            Array of reference audio samples
        """
        with self._buffer_lock:
            if not self._reference_buffer:
                return np.zeros(num_samples, dtype=np.float32)

            reference = np.concatenate(list(self._reference_buffer))

        if reference.size >= num_samples:
            return reference[-num_samples:].copy()

        padded = np.zeros(num_samples, dtype=np.float32)
        padded[-reference.size:] = reference
        return padded

    async def _ensure_stream(self):
        """Lazily create the underlying OutputStream."""
        if self._running.is_set():
            return

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._start_stream)

    def _start_stream(self):
        """Initialize and start the output stream (runs in executor)."""
        if self._running.is_set():
            return

        self.logger.debug(
            f"Creating OutputStream (sample_rate={self.sample_rate}, "
            f"frame_size={self.frame_size})"
        )

        stream_kwargs = {
            "samplerate": self.sample_rate,
            "blocksize": self.frame_size,
            "channels": 1,
            "dtype": "float32",
            "callback": self._audio_callback,
            "finished_callback": self._finished_callback,
        }

        if self.output_device is not None:
            stream_kwargs["device"] = self.output_device

        self._stream = sd.OutputStream(**stream_kwargs)

        self._running.set()
        self._paused.clear()
        self._stream.start()

    def _audio_callback(self, outdata, frames, _time, status):
        """Audio callback for sounddevice (runs in audio thread)."""
        if status:
            self.logger.warning(f"Output stream status: {status}")

        frame = None
        with self._buffer_lock:
            if not self._paused.is_set() and self._buffer:
                frame = self._buffer.popleft()
                self._reference_buffer.append(frame.copy())

        if frame is None:
            outdata.fill(0.0)
            return

        if frame.size != frames:
            if frame.size > frames:
                outdata[:, 0] = frame[:frames]
                remainder = frame[frames:]
                with self._buffer_lock:
                    self._buffer.appendleft(remainder)
            else:
                outdata[:, 0] = 0.0
                outdata[:frame.size, 0] = frame
            return

        outdata[:, 0] = frame

        # Push frames to reverse consumer if registered
        if self._reverse_consumer is not None:
            try:
                self._reverse_consumer(np.ascontiguousarray(frame, dtype=np.float32))
            except Exception as e:
                self.logger.debug(f"Error pushing reverse audio to AEC: {e}")

    def _finished_callback(self):
        """Called when output stream finishes."""
        self.logger.debug("Output stream finished")

    def _split_frames(self, data: np.ndarray) -> list[np.ndarray]:
        """Split raw audio into fixed-size frames.

        Args:
            data: Raw audio data

        Returns:
            List of fixed-size audio frames
        """
        if data.size == 0:
            return []

        total_frames = len(data) // self.frame_size
        remainder = len(data) % self.frame_size

        frames = [
            np.copy(data[i * self.frame_size : (i + 1) * self.frame_size])
            for i in range(total_frames)
        ]

        if remainder:
            padded = np.zeros(self.frame_size, dtype=np.float32)
            padded[:remainder] = data[-remainder:]
            frames.append(padded)

        return frames
