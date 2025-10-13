import asyncio
from typing import AsyncIterator
from multimodal_assistant.engines.base import ImageFrame

class FrameSampler:
    """Samples video frames at specified rate"""

    def __init__(self, target_fps: int = 1):
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps

    async def sample_stream(
        self,
        frame_stream: AsyncIterator[ImageFrame]
    ) -> AsyncIterator[ImageFrame]:
        """Sample frames from stream"""
        last_sample_time = 0.0

        async for frame in frame_stream:
            current_time = asyncio.get_event_loop().time()

            if current_time - last_sample_time >= self.frame_interval:
                yield frame
                last_sample_time = current_time
