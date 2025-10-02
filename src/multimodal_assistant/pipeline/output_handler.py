import sounddevice as sd
import numpy as np
from multimodal_assistant.engines.base import AudioChunk
from typing import AsyncIterator
import asyncio

class AudioOutputHandler:
    """Handles audio output playback"""

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.stream = None

    async def start_playback(self, audio_stream: AsyncIterator[AudioChunk]):
        """Play audio stream"""
        async for chunk in audio_stream:
            await self._play_chunk(chunk)

    async def _play_chunk(self, chunk: AudioChunk):
        """Play single audio chunk"""
        loop = asyncio.get_event_loop()

        def _play():
            sd.play(chunk.data, chunk.sample_rate)
            sd.wait()

        await loop.run_in_executor(None, _play)

    async def stop_playback(self):
        """Stop audio playback"""
        sd.stop()

class TextOutputHandler:
    """Handles text output display"""

    def __init__(self):
        pass

    async def display_stream(self, text_stream: AsyncIterator[str]):
        """Display streaming text"""
        async for token in text_stream:
            print(token, end='', flush=True)

        print()  # New line at end

    def display_message(self, message: str):
        """Display complete message"""
        print(message)
