import asyncio
from collections import deque
from typing import AsyncIterator, Optional
import threading

import numpy as np
import sounddevice as sd

from multimodal_assistant.engines.base import AudioChunk
from multimodal_assistant.utils.logger import setup_logger


class AudioOutputHandler:
    """Handles low-latency audio playback with pause/resume support."""

    def __init__(
        self,
        sample_rate: int = 24000,
        frame_duration_ms: int = 20,
        reference_window_secs: float = 2.0,
        output_device: int | str | None = None,
    ):
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(self.sample_rate * (self.frame_duration_ms / 1000.0))
        self.reference_window_secs = reference_window_secs

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

        self.logger = setup_logger("multimodal_assistant.output.audio")
        self._reverse_consumer = None  # Optional callback for AEC reverse path
        self.output_device = output_device

    async def start_playback(self, audio_stream: AsyncIterator[AudioChunk]):
        """Consume an audio chunk stream and enqueue frames for playback."""
        self.logger.debug("start_playback() called, ensuring output stream...")
        await self._ensure_stream()
        self.logger.debug("Output stream ready, consuming audio chunks...")

        chunk_count = 0
        try:
            async for chunk in audio_stream:
                chunk_count += 1
                self.logger.debug(f"Received audio chunk #{chunk_count}: {chunk.data.shape if chunk and chunk.data is not None else None}")
                await self.enqueue_chunk(chunk)
        except asyncio.CancelledError:
            self.logger.debug(f"Audio playback task cancelled after {chunk_count} chunks; stopping stream")
            await self.stop_playback()
            raise
        finally:
            self.logger.debug(f"Audio playback finished, total chunks: {chunk_count}")

    async def enqueue_chunk(self, chunk: AudioChunk):
        """Slice an AudioChunk into frames and enqueue for playback."""
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

    def register_reverse_consumer(self, fn):
        """Register a callable to receive each 10 ms playback frame for AEC.

        The consumer should accept a mono float32 numpy array at 16 kHz.
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
        """Return True if audio is currently playing (buffer not empty and not paused)."""
        with self._buffer_lock:
            return bool(self._buffer) and not self._paused.is_set()

    def get_reference_audio(self, num_samples: int) -> np.ndarray:
        """Return the latest far-end audio samples for echo cancellation."""
        with self._buffer_lock:
            if not self._reference_buffer:
                return np.zeros(num_samples, dtype=np.float32)

            reference = np.concatenate(list(self._reference_buffer))

        if reference.size >= num_samples:
            return reference[-num_samples:].copy()

        padded = np.zeros(num_samples, dtype=np.float32)
        padded[-reference.size :] = reference
        return padded

    async def _ensure_stream(self):
        """Lazily create the underlying OutputStream."""
        if self._running.is_set():
            return

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._start_stream)

    def _start_stream(self):
        if self._running.is_set():
            return

        self.logger.debug(
            f"Creating OutputStream (sample_rate={self.sample_rate}, "
            f"frame_size={self.frame_size})"
        )

        stream_kwargs = dict(
            samplerate=self.sample_rate,
            blocksize=self.frame_size,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
            finished_callback=self._finished_callback,
        )
        if self.output_device is not None:
            stream_kwargs["device"] = self.output_device

        self._stream = sd.OutputStream(**stream_kwargs)

        self._running.set()
        self._paused.clear()
        self._stream.start()

    def _audio_callback(self, outdata, frames, _time, status):  # pragma: no cover
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
                outdata[: frame.size, 0] = frame
            return

        outdata[:, 0] = frame

        # Push reverse frames to AEC consumer if present, resampled to 16 kHz
        if self._reverse_consumer is not None:
            try:
                from multimodal_assistant.audio.resample import resample_linear

                reverse_16k = resample_linear(frame, self.sample_rate, 16000)
                fp = 16000 // 100  # 10 ms @ 16 kHz

                # Emit one or more contiguous 10 ms frames
                if reverse_16k.size >= fp:
                    num = reverse_16k.size // fp
                    for i in range(num):
                        seg = reverse_16k[i * fp : (i + 1) * fp]
                        # Ensure contiguous copy for ctypes
                        self._reverse_consumer(np.ascontiguousarray(seg, dtype=np.float32))
                else:
                    # Pad up to 10 ms if shorter
                    tmp = np.zeros(fp, dtype=np.float32)
                    n = min(fp, reverse_16k.size)
                    if n:
                        tmp[:n] = reverse_16k[:n]
                    self._reverse_consumer(tmp)
            except Exception as e:
                # Never break audio on reverse errors
                self.logger.debug(f"Error pushing reverse audio to AEC: {e}")

    def _finished_callback(self):  # pragma: no cover
        self.logger.debug("Output stream finished")

    def _split_frames(self, data: np.ndarray) -> list[np.ndarray]:
        """Split raw audio into fixed-size frames."""
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


class TextOutputHandler:
    """Handles text output display"""

    def __init__(self):
        self._buffer = []
        self.logger = setup_logger("multimodal_assistant.output.text")

    async def display_stream(self, text_stream: AsyncIterator[str]):
        """Display streaming text"""
        self._buffer = []
        token_count = 0

        async for token in text_stream:
            # Clean up token: remove standalone backslashes and excessive newlines
            cleaned_token = token

            # Skip standalone backslashes
            if token == '\\':
                continue

            # Replace multiple consecutive newlines with max 2
            if '\n' in token:
                parts = token.split('\n')
                # Keep at most 2 consecutive newlines
                cleaned_parts = []
                for i, part in enumerate(parts):
                    if i > 0:
                        # Add newline, but limit consecutive ones
                        if not (i > 1 and parts[i-1] == '' and part == ''):
                            cleaned_parts.append('')
                    cleaned_parts.append(part)
                cleaned_token = '\n'.join(cleaned_parts)

            if cleaned_token:
                print(cleaned_token, end='', flush=True)
                self._buffer.append(cleaned_token)
                token_count += 1

        print()  # New line at end
        self.logger.debug(f"Displayed {token_count} tokens")

    def display_message(self, message: str):
        """Display complete message"""
        self.logger.debug(f"Displaying message: {message}")
        print(message)
