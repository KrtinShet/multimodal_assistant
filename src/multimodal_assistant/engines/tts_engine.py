import numpy as np
from typing import AsyncIterator
from .base import ITTSEngine, AudioChunk
import asyncio

class KokoroTTSEngine(ITTSEngine):
    """Kokoro TTS implementation using Python API"""

    def __init__(self):
        self.sample_rate = 24000
        self.kokoro = None

    async def initialize(self):
        """Initialize TTS"""
        try:
            from kokoro_onnx import Kokoro
            self.kokoro = Kokoro("kokoro-v0_19.onnx", "voices.json")
            print("✅ Kokoro TTS initialized")
        except ImportError:
            raise RuntimeError(
                "Kokoro not found. Install: uv add kokoro-onnx"
            )
        except Exception as e:
            print(f"⚠️  Kokoro TTS initialization failed: {e}")
            print("TTS will be disabled, but the assistant will still work")
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

        async for text in text_stream:
            sentence_buffer.append(text)

            # Wait for sentence end
            if any(p in text for p in ['.', '!', '?', '\n']):
                sentence = ''.join(sentence_buffer).strip()
                sentence_buffer = []

                if sentence:
                    # Synthesize in thread pool
                    loop = asyncio.get_event_loop()
                    audio_data = await loop.run_in_executor(
                        None,
                        self._synthesize_sentence,
                        sentence
                    )

                    if audio_data is not None:
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
            # Generate speech using Kokoro API
            samples, sample_rate = self.kokoro.create(text, voice='af_sky', speed=1.0)

            # Ensure float32 format
            if samples.dtype != np.float32:
                samples = samples.astype(np.float32)

            return samples
        except Exception as e:
            print(f"TTS synthesis error: {e}")
            return None

    async def shutdown(self):
        """Cleanup"""
        self.kokoro = None
