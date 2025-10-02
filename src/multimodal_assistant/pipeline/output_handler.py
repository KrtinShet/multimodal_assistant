import sounddevice as sd
import numpy as np
from multimodal_assistant.engines.base import AudioChunk
from typing import AsyncIterator
import asyncio
from multimodal_assistant.utils.logger import setup_logger

class AudioOutputHandler:
    """Handles audio output playback"""

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.stream = None
        self.logger = setup_logger("multimodal_assistant.output.audio")

    async def start_playback(self, audio_stream: AsyncIterator[AudioChunk]):
        """Play audio stream"""
        self.logger.debug("Starting audio playback")
        try:
            async for chunk in audio_stream:
                await self._play_chunk(chunk)
        except asyncio.CancelledError:
            sd.stop()
            raise

    async def _play_chunk(self, chunk: AudioChunk):
        """Play single audio chunk"""
        self.logger.debug(f"Playing audio chunk ({len(chunk.data)} samples)")
        loop = asyncio.get_event_loop()

        def _play():
            sd.play(chunk.data, chunk.sample_rate)
            sd.wait()

        await loop.run_in_executor(None, _play)

    async def stop_playback(self):
        """Stop audio playback"""
        self.logger.debug("Stopping audio playback")
        sd.stop()

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
