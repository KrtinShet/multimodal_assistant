import sounddevice as sd
import cv2
import numpy as np
from multimodal_assistant.core.streams import AsyncStream
from multimodal_assistant.engines.base import AudioChunk, ImageFrame
import asyncio
from multimodal_assistant.processors.vad import VADProcessor

class AudioInputHandler:
    """Handles microphone input with VAD"""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * 0.1)  # 100ms chunks
        self.vad = VADProcessor()
        self.stream = None
        self.loop = None

    async def start_capture(self) -> AsyncStream[AudioChunk]:
        """Start audio capture stream"""
        output_stream = AsyncStream[AudioChunk]()

        # Store the main event loop for thread-safe async calls
        self.loop = asyncio.get_running_loop()

        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Audio error: {status}")

            # Create task to process audio using thread-safe method
            audio_data = indata[:, 0].copy()  # Mono

            # Schedule the coroutine in the main event loop from this thread
            asyncio.run_coroutine_threadsafe(
                self._process_audio(audio_data, output_stream),
                self.loop
            )

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            blocksize=self.chunk_size,
            callback=audio_callback
        )
        self.stream.start()

        return output_stream

    async def _process_audio(self, audio_data: np.ndarray, stream: AsyncStream):
        """Process audio with VAD"""
        is_speech = self.vad.is_speech(audio_data, self.sample_rate)

        chunk = AudioChunk(
            data=audio_data,
            sample_rate=self.sample_rate,
            timestamp=asyncio.get_event_loop().time(),
            is_speech=is_speech
        )

        await stream.put(chunk)

    async def stop_capture(self):
        """Stop audio capture"""
        if self.stream:
            self.stream.stop()
            self.stream.close()

class VideoInputHandler:
    """Handles camera input with frame sampling"""

    def __init__(self, fps: int = 1):
        self.fps = fps
        self.cap = None

    async def start_capture(self) -> AsyncStream[ImageFrame]:
        """Start video capture stream"""
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

                # Control FPS
                await asyncio.sleep(1.0 / self.fps)

        asyncio.create_task(_capture_loop())
        return output_stream

    async def stop_capture(self):
        """Stop video capture"""
        if self.cap:
            self.cap.release()
