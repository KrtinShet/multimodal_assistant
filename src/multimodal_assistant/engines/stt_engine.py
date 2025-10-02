from faster_whisper import WhisperModel
import numpy as np
from typing import AsyncIterator
from .base import ISTTEngine, AudioChunk, Transcription
import asyncio

class FasterWhisperEngine(ISTTEngine):
    """Faster-Whisper STT implementation optimized for M4"""

    def __init__(self, model_size: str = "small"):
        self.model_size = model_size
        self.model = None

    async def initialize(self):
        """Load model with CoreML optimization"""
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        self.model = await loop.run_in_executor(
            None,
            lambda: WhisperModel(
                self.model_size,
                device="cpu",  # faster-whisper uses CoreML internally
                compute_type="int8",  # Quantized for speed
                cpu_threads=8  # Optimize for M4 Pro cores
            )
        )

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[Transcription]:
        """Stream transcription with chunking"""

        buffer = []
        async for chunk in audio_stream:
            if not chunk.is_speech:
                continue

            buffer.append(chunk.data)

            # Process when we have 3 seconds of audio
            if len(buffer) >= 30:  # 30 chunks × 100ms = 3s
                audio_data = np.concatenate(buffer)
                buffer = []

                # Transcribe in thread pool
                loop = asyncio.get_event_loop()
                segments, info = await loop.run_in_executor(
                    None,
                    lambda: self.model.transcribe(
                        audio_data,
                        language="en",
                        beam_size=1,  # Faster, less accurate
                        vad_filter=False,  # We already did VAD
                    )
                )

                # Stream results
                for segment in segments:
                    yield Transcription(
                        text=segment.text,
                        confidence=segment.avg_logprob,
                        is_final=True,
                        timestamp=chunk.timestamp
                    )

    async def shutdown(self):
        """Cleanup"""
        self.model = None
