import numpy as np
from typing import AsyncIterator
from .base import ITTSEngine, AudioChunk
import asyncio
from pathlib import Path
from assistant.utils.logger import setup_logger

class KokoroTTSEngine(ITTSEngine):
    """Kokoro TTS implementation using Python API"""

    def __init__(self):
        self.sample_rate = 24000
        self.kokoro = None
        self.model_path = Path("assets/kokoro/kokoro-v1.0.onnx")
        self.voices_path = Path("assets/kokoro/voices-v1.0.bin")
        self.logger = setup_logger("assistant.engines.tts")
        self.max_chars_per_chunk = 160

    async def initialize(self):
        """Initialize TTS"""
        try:
            from kokoro_onnx import Kokoro

            # Check if model files exist
            if not self.model_path.exists() or not self.voices_path.exists():
                self.logger.warning(f"Kokoro model files not found at {self.model_path.parent}")
                self.logger.warning("TTS will be disabled, but the assistant will still work")
                self.kokoro = None
                return

            self.logger.info("Initializing Kokoro TTS")
            self.kokoro = Kokoro(str(self.model_path), str(self.voices_path))
            self.logger.info("Kokoro TTS initialized successfully")
        except ImportError:
            self.logger.warning("Kokoro not found. Install: uv add kokoro-onnx")
            self.logger.warning("TTS will be disabled, but the assistant will still work")
            self.kokoro = None
        except Exception as e:
            self.logger.error(f"Kokoro TTS initialization failed: {e}")
            self.logger.warning("TTS will be disabled, but the assistant will still work")
            self.kokoro = None

    async def synthesize_stream(
        self,
        text_stream: AsyncIterator[str]
    ) -> AsyncIterator[AudioChunk]:
        """Stream TTS synthesis"""

        if not self.kokoro:
            # TTS disabled, just consume the stream
            self.logger.warning("TTS is disabled - kokoro is None, consuming text stream without synthesis")
            async for _ in text_stream:
                pass
            return

        self.logger.debug("Starting TTS synthesis stream")
        sentence_buffer: list[str] = []

        try:
            async for text in text_stream:
                self.logger.debug(f"TTS received text token: {repr(text)}")
                sentence_buffer.append(text)
                joined = ''.join(sentence_buffer)

                has_terminal = any(p in text for p in ['.', '!', '?', '\n'])
                over_length = len(joined) >= self.max_chars_per_chunk

                if has_terminal or over_length:
                    sentence = joined.strip()
                    sentence_buffer = []

                    if sentence:
                        async for chunk in self._synthesize_async(sentence):
                            yield chunk
        except asyncio.CancelledError:
            # Propagate cancellation after cleanup
            raise
        finally:
            if sentence_buffer:
                sentence = ''.join(sentence_buffer).strip()
                if sentence:
                    async for chunk in self._synthesize_async(sentence):
                        yield chunk

    async def _synthesize_async(self, sentence: str):
        self.logger.debug(f"Synthesizing: {sentence}")
        loop = asyncio.get_event_loop()
        audio_data = await loop.run_in_executor(
            None,
            self._synthesize_sentence,
            sentence
        )

        if audio_data is not None:
            self.logger.debug(f"Generated {len(audio_data)} audio samples")
            yield AudioChunk(
                data=audio_data,
                sample_rate=self.sample_rate,
                timestamp=asyncio.get_event_loop().time()
            )

    def _synthesize_sentence(self, text: str) -> np.ndarray:
        """Synthesize single sentence (blocking)"""
        if not self.kokoro:
            return None

        try:
            # Generate speech using Kokoro API (matching your working implementation)
            samples, sample_rate = self.kokoro.create(
                text=text,
                voice='af_sky',  # Using a default voice
                speed=1.0,
                lang='en-us'
            )

            self.sample_rate = sample_rate

            # Ensure float32 format
            if samples.dtype != np.float32:
                samples = samples.astype(np.float32)

            return samples
        except Exception as e:
            self.logger.error(f"TTS synthesis error: {e}")
            return None

    async def shutdown(self):
        """Cleanup"""
        self.kokoro = None
