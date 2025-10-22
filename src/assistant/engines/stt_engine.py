from faster_whisper import WhisperModel
import numpy as np
from typing import AsyncIterator
from collections import deque
from .base import ISTTEngine, AudioChunk, Transcription
import asyncio
from assistant.utils.logger import setup_logger

class FasterWhisperEngine(ISTTEngine):
    """Faster-Whisper STT implementation with improved real-time performance.

    Features:
    - Pre-roll buffer to capture speech start
    - Adaptive silence detection
    - Better utterance boundary detection
    - Optimized for low-latency transcription
    """

    def __init__(
        self,
        model_size: str = "small",
        min_speech_ms: int = 300,
        silence_ms: int = 600,
        max_utterance_ms: int = 15000,
        pre_roll_ms: int = 300
    ):
        self.model_size = model_size
        self.model = None
        self.min_speech_duration = min_speech_ms / 1000.0
        self.silence_duration = silence_ms / 1000.0
        self.max_utterance_duration = max_utterance_ms / 1000.0
        self.pre_roll_duration = pre_roll_ms / 1000.0
        self.logger = setup_logger("assistant.engines.stt")

    async def initialize(self):
        """Load model with CoreML optimization"""
        self.logger.info(f"Initializing Faster-Whisper (model_size={self.model_size})")
        # Run in thread pool to avoid blocking
        loop = asyncio.get_running_loop()
        self.model = await loop.run_in_executor(
            None,
            lambda: WhisperModel(
                self.model_size,
                device="cpu",  # faster-whisper uses CoreML internally
                compute_type="int8",  # Quantized for speed
                cpu_threads=8  # Optimize for M4 Pro cores
            )
        )
        self.logger.info("Faster-Whisper initialized successfully")

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[Transcription]:
        """Stream transcription with improved boundary detection and pre-roll buffer.

        The pre-roll buffer captures audio before speech is detected, ensuring
        we don't miss the beginning of utterances.
        """
        # Pre-roll buffer to capture audio before speech detection
        pre_roll_buffer = deque(maxlen=50)  # Adjust based on chunk duration
        speech_buffer = []
        speech_detected = False
        speech_duration = 0.0
        silence_duration = 0.0
        pre_roll_duration = 0.0
        last_chunk_timestamp = 0.0
        speech_start_time = 0.0

        while True:
            chunk = await audio_stream.next()

            if chunk is None:
                # No new audio available; check if we already captured an utterance
                if speech_detected and silence_duration >= self.silence_duration:
                    break
                await asyncio.sleep(0.01)
                continue

            chunk_duration = len(chunk.data) / float(chunk.sample_rate)
            last_chunk_timestamp = chunk.timestamp

            if chunk.is_speech:
                if not speech_detected:
                    # Speech just started - include pre-roll buffer
                    speech_detected = True
                    speech_start_time = chunk.timestamp

                    # Add pre-roll frames to speech buffer
                    pre_roll_frames = []
                    for pre_chunk in pre_roll_buffer:
                        pre_roll_frames.append(pre_chunk.data)
                        pre_roll_duration += len(pre_chunk.data) / float(pre_chunk.sample_rate)

                    if pre_roll_frames:
                        speech_buffer.extend(pre_roll_frames)
                        self.logger.debug(
                            f"Added {len(pre_roll_frames)} pre-roll frames "
                            f"({pre_roll_duration:.2f}s)"
                        )

                    pre_roll_buffer.clear()

                speech_duration += chunk_duration
                silence_duration = 0.0
                speech_buffer.append(chunk.data)
            else:
                if not speech_detected:
                    # Keep filling pre-roll buffer before speech starts
                    pre_roll_buffer.append(chunk)
                    continue

                # We're in a silence period after speech
                silence_duration += chunk_duration

                # Include silence frames up to threshold (for natural pauses)
                if silence_duration < self.silence_duration:
                    speech_buffer.append(chunk.data)
                else:
                    # Silence threshold reached - end of utterance
                    break

            # Check for max utterance duration
            if speech_detected and speech_duration >= self.max_utterance_duration:
                self.logger.debug(
                    f"Max utterance duration reached ({self.max_utterance_duration}s)"
                )
                break

        if not speech_buffer:
            return

        # Calculate total audio duration (including pre-roll)
        total_audio_duration = pre_roll_duration + speech_duration

        # Require minimum speech duration to avoid false positives
        if speech_duration < self.min_speech_duration:
            self.logger.debug(
                f"Speech too short ({speech_duration:.2f}s), ignoring "
                f"(min={self.min_speech_duration:.2f}s)"
            )
            return

        audio_data = np.concatenate(speech_buffer)
        self.logger.info(
            f"Transcribing utterance: speech={speech_duration:.2f}s, "
            f"total={total_audio_duration:.2f}s, "
            f"silence={silence_duration:.2f}s"
        )

        # Transcribe in thread pool
        loop = asyncio.get_running_loop()
        segments, info = await loop.run_in_executor(
            None,
            lambda: self.model.transcribe(
                audio_data,
                language="en",
                beam_size=1,  # Faster, less accurate
                vad_filter=False,  # VAD already applied
            )
        )

        for segment in segments:
            if segment.text.strip():  # Only yield non-empty transcriptions
                self.logger.info(f"Transcription: {segment.text}")
                yield Transcription(
                    text=segment.text,
                    confidence=segment.avg_logprob,
                    is_final=True,
                    timestamp=last_chunk_timestamp
                )

    async def shutdown(self):
        """Cleanup"""
        self.model = None

    async def transcribe_audio(self, audio_data: np.ndarray) -> str:
        """Transcribe a single audio buffer."""

        if audio_data.size == 0:
            return ""

        loop = asyncio.get_running_loop()
        segments, _ = await loop.run_in_executor(
            None,
            lambda: self.model.transcribe(
                audio_data,
                language="en",
                beam_size=1,
                vad_filter=False,
            ),
        )

        return " ".join(segment.text.strip() for segment in segments if segment.text).strip()
