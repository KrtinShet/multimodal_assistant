from faster_whisper import WhisperModel
import numpy as np
from typing import AsyncIterator
from .base import ISTTEngine, AudioChunk, Transcription
import asyncio
from multimodal_assistant.utils.logger import setup_logger

class FasterWhisperEngine(ISTTEngine):
    """Faster-Whisper STT implementation optimized for M4"""

    def __init__(
        self,
        model_size: str = "small",
        min_speech_ms: int = 300,
        silence_ms: int = 600,
        max_utterance_ms: int = 15000
    ):
        self.model_size = model_size
        self.model = None
        self.min_speech_duration = min_speech_ms / 1000.0
        self.silence_duration = silence_ms / 1000.0
        self.max_utterance_duration = max_utterance_ms / 1000.0
        self.logger = setup_logger("multimodal_assistant.engines.stt")

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
        """Stream transcription with chunking"""

        buffer = []
        speech_detected = False
        speech_duration = 0.0
        silence_duration = 0.0
        last_chunk_timestamp = 0.0

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
                speech_detected = True
                speech_duration += chunk_duration
                silence_duration = 0.0
                buffer.append(chunk.data)
            else:
                if not speech_detected:
                    continue

                silence_duration += chunk_duration
                if silence_duration >= self.silence_duration:
                    break

            if speech_detected and speech_duration >= self.max_utterance_duration:
                break

        if not buffer:
            return

        # Require minimum speech duration to avoid false positives
        if speech_duration < self.min_speech_duration:
            self.logger.debug(f"Speech too short ({speech_duration:.2f}s), ignoring")
            return

        audio_data = np.concatenate(buffer)
        self.logger.debug(f"Transcribing {speech_duration:.2f}s of speech")

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
