import numpy as np
from typing import AsyncIterator
from .base import ITTSEngine, AudioChunk
import asyncio
from pathlib import Path
from multimodal_assistant.utils.logger import setup_logger

class KokoroTTSEngine(ITTSEngine):
    """Kokoro TTS implementation using Python API"""

    def __init__(self):
        self.sample_rate = 24000
        self.kokoro = None
        self.model_path = Path("assets/kokoro/kokoro-v1.0.onnx")
        self.voices_path = Path("assets/kokoro/voices-v1.0.bin")
        self.logger = setup_logger("multimodal_assistant.engines.tts")

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
            async for _ in text_stream:
                pass
            return

        sentence_buffer = []

        try:
            async for text in text_stream:
                sentence_buffer.append(text)

                # Wait for sentence end
                if any(p in text for p in ['.', '!', '?', '\n']):
                    sentence = ''.join(sentence_buffer).strip()
                    sentence_buffer = []

                    if sentence:
                        self.logger.debug(f"Synthesizing: {sentence}")
                        # Synthesize in thread pool
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
        except asyncio.CancelledError:
            # Propagate cancellation after cleanup
            raise

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
