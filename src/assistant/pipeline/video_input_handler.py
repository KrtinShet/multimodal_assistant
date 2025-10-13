import asyncio
from typing import Optional

import cv2

from assistant.core.streams import AsyncStream
from assistant.engines.base import ImageFrame
from assistant.config.settings import Settings
from assistant.utils.logger import setup_logger


class VideoInputHandler:
    """Handles camera input with frame sampling.

    This handler captures video frames from a camera at a configurable
    frame rate and streams them to downstream consumers.
    """

    def __init__(self, settings: Settings):
        """Initialize video input handler.

        Args:
            settings: Application settings containing video configuration
        """
        self.fps = settings.video_fps
        self.cap: Optional[cv2.VideoCapture] = None
        self.logger = setup_logger("assistant.input.video", settings.log_level)
        self._running = False
        self._capture_task: Optional[asyncio.Task] = None

    async def start_capture(self) -> AsyncStream[ImageFrame]:
        """Start video capture stream from camera.

        Returns:
            AsyncStream yielding ImageFrame objects

        Raises:
            RuntimeError: If camera cannot be opened
        """
        self.logger.info(f"Starting video capture (fps={self.fps})")
        output_stream = AsyncStream[ImageFrame]()

        # Open camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")

        self._running = True

        # Start capture loop as background task
        self._capture_task = asyncio.create_task(
            self._capture_loop(output_stream)
        )

        self.logger.info("Video capture started")
        return output_stream

    async def _capture_loop(self, output_stream: AsyncStream[ImageFrame]):
        """Background loop to capture video frames.

        Args:
            output_stream: Stream to push captured frames
        """
        frame_id = 0

        while self._running and self.cap and self.cap.isOpened():
            try:
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

                    if frame_id % 10 == 0:
                        self.logger.debug(f"Captured {frame_id} frames")

                # Control FPS
                await asyncio.sleep(1.0 / self.fps)

            except Exception as e:
                self.logger.error(f"Error capturing video frame: {e}")
                break

        # Close stream when done
        await output_stream.close()
        self.logger.debug(f"Video capture loop ended, total frames: {frame_id}")

    async def stop_capture(self):
        """Stop video capture and release camera resources."""
        if self._running:
            self.logger.info("Stopping video capture")
            self._running = False

            # Wait for capture task to complete
            if self._capture_task:
                try:
                    await asyncio.wait_for(self._capture_task, timeout=2.0)
                except asyncio.TimeoutError:
                    self.logger.warning("Video capture task did not stop in time")
                    self._capture_task.cancel()

            # Release camera
            if self.cap:
                self.cap.release()
                self.cap = None

    def is_capturing(self) -> bool:
        """Check if video capture is currently active.

        Returns:
            True if capturing, False otherwise
        """
        return self._running and self.cap is not None and self.cap.isOpened()
