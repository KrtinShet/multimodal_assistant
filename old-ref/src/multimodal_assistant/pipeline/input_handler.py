import asyncio
from collections.abc import Callable

import cv2
import numpy as np
import sounddevice as sd

from multimodal_assistant.core.streams import AsyncStream
from multimodal_assistant.engines.base import AudioChunk, ImageFrame
from multimodal_assistant.processors.aec import AECProcessor
from multimodal_assistant.processors.aec_core import IAEC
from multimodal_assistant.processors.vad import VADProcessor
from multimodal_assistant.utils.logger import setup_logger


class AudioInputHandler:
    """Handles microphone input with VAD and optional echo cancellation."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 10,
        vad_aggressiveness: int = 3,
        enable_aec: bool = True,
        echo_suppression_strength: float = 0.6,
        echo_gate_correlation: float = 0.35,
        echo_gate_near_far_ratio: float = 0.55,
        input_device: int | str | None = None,
    ):
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.chunk_size = int(sample_rate * (frame_duration_ms / 1000.0))

        self.vad = VADProcessor(aggressiveness=vad_aggressiveness)
        # Built-in fallback AEC; can be overridden by a WebRTC APM backend
        self.aec = AECProcessor(
            sample_rate=sample_rate,
            frame_size=self.chunk_size,
            enabled=enable_aec,
            suppression_strength=echo_suppression_strength,
        )
        self._apm: IAEC | None = None
        self.echo_gate_correlation = echo_gate_correlation
        self.echo_gate_near_far_ratio = echo_gate_near_far_ratio
        self.input_device = input_device

        self.stream = None
        self.loop = None
        self.logger = setup_logger("multimodal_assistant.input.audio")
        self.reference_provider: Callable[[int], np.ndarray] | None = None

    def set_reference_provider(
        self,
        provider: Callable[[int], np.ndarray],
    ):
        """Register a callable that supplies far-end reference audio."""

        self.reference_provider = provider

    def set_webrtc_apm(self, apm: IAEC):
        """Inject a WebRTC APM backend; supersedes the internal fallback AEC."""
        self._apm = apm

    async def start_capture(self) -> AsyncStream[AudioChunk]:
        """Start audio capture stream."""

        self.logger.info(
            "Starting audio capture (sample_rate=%s, chunk_size=%s)",
            self.sample_rate,
            self.chunk_size,
        )
        output_stream = AsyncStream[AudioChunk]()

        self.loop = asyncio.get_running_loop()

        def audio_callback(indata, frames, _time, status):
            if status:
                self.logger.error("Audio input status: %s", status)

            audio_data = indata[:, 0].copy()  # Mono

            asyncio.run_coroutine_threadsafe(
                self._process_audio(audio_data, output_stream),
                self.loop,
            )

        stream_kwargs = dict(
            samplerate=self.sample_rate,
            channels=1,
            blocksize=self.chunk_size,
            dtype="float32",
            callback=audio_callback,
        )
        if self.input_device is not None:
            stream_kwargs["device"] = self.input_device

        self.stream = sd.InputStream(**stream_kwargs)
        self.stream.start()
        self.logger.info("Audio capture started")

        return output_stream

    async def _process_audio(self, audio_data: np.ndarray, stream: AsyncStream):
        """Apply echo suppression, VAD, and push to downstream consumers."""

        if self._apm is not None:
            # WebRTC APM path: expect 10 ms frames at 16 kHz
            # If sample rate != 16k, resample
            from multimodal_assistant.audio.resample import resample_linear

            if self.sample_rate != 16000:
                near_16k = resample_linear(audio_data, self.sample_rate, 16000)
            else:
                near_16k = audio_data

            # Slice to ensure 10 ms and process each frame
            fp = 16000 // 100
            if near_16k.size != fp:
                # pad/trim to 10 ms
                tmp = np.zeros(fp, dtype=np.float32)
                n = min(fp, near_16k.size)
                tmp[:n] = near_16k[:n]
                near_16k = tmp

            processed_16k = self._apm.process_near(near_16k)
            # Resample back to the pipeline rate if necessary
            if self.sample_rate != 16000:
                processed = resample_linear(processed_16k, 16000, self.sample_rate)
            else:
                processed = processed_16k
            is_speech = self.vad.is_speech(processed, self.sample_rate)

            # Echo gate: if current mic frame strongly correlates with far-end,
            # treat as echo and suppress speech flag to avoid self-triggering.
            try:
                if self.reference_provider is not None:
                    window_len = min(self.sample_rate // 2, len(audio_data) * 20)  # ~up to 500ms
                    reference_window = self.reference_provider(window_len)
                    if reference_window is not None and reference_window.size > 0:
                        # Reuse fallback suppressor solely to compute correlation metrics
                        _ = self.aec.process(audio_data, reference_window)
                        if self.aec.is_echo(self.echo_gate_correlation, self.echo_gate_near_far_ratio):
                            processed = np.zeros_like(processed)
                            is_speech = False
            except Exception:
                # Never fail the audio path due to gating errors
                pass
        else:
            # Fallback correlation AEC path
            reference_window = None
            if self.reference_provider:
                window_len = min(self.sample_rate // 2, len(audio_data) * 20)  # up to ~500ms
                reference_window = self.reference_provider(window_len)

            processed = self.aec.process(audio_data, reference_window)

            if self.aec.is_echo(self.echo_gate_correlation, self.echo_gate_near_far_ratio):
                processed = np.zeros_like(processed)
                is_speech = False
            else:
                is_speech = self.vad.is_speech(processed, self.sample_rate)

        chunk = AudioChunk(
            data=processed,
            sample_rate=self.sample_rate,
            timestamp=asyncio.get_event_loop().time(),
            is_speech=is_speech,
        )

        await stream.put(chunk)

    async def stop_capture(self):
        """Stop audio capture."""

        if self.stream:
            self.logger.info("Stopping audio capture")
            self.stream.stop()
            self.stream.close()

class VideoInputHandler:
    """Handles camera input with frame sampling"""

    def __init__(self, fps: int = 1):
        self.fps = fps
        self.cap = None
        self.logger = setup_logger("multimodal_assistant.input.video")

    async def start_capture(self) -> AsyncStream[ImageFrame]:
        """Start video capture stream"""
        self.logger.info(f"Starting video capture (fps={self.fps})")
        output_stream = AsyncStream[ImageFrame]()

        self.cap = cv2.VideoCapture(0)

        async def _capture_loop():
            frame_id = 0
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    image_frame = ImageFrame(
                        data=frame_rgb,
                        timestamp=asyncio.get_event_loop().time(),
                        frame_id=f"frame_{frame_id}"
                    )

                    await output_stream.put(image_frame)
                    frame_id += 1
                    if frame_id % 10 == 0:
                        self.logger.debug(f"Captured {frame_id} frames")

                # Control FPS
                await asyncio.sleep(1.0 / self.fps)

        asyncio.create_task(_capture_loop())
        self.logger.info("Video capture started")
        return output_stream

    async def stop_capture(self):
        """Stop video capture"""
        if self.cap:
            self.logger.info("Stopping video capture")
            self.cap.release()
