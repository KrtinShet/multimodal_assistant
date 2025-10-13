from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class IAEC(ABC):
    """AEC interface for pluggable backends (WebRTC APM, fallback)."""

    @abstractmethod
    def process_near(self, frame: np.ndarray) -> np.ndarray:
        """Process a 10 ms near-end mic frame and return echo-reduced frame."""
        raise NotImplementedError

    @abstractmethod
    def push_reverse(self, frame: np.ndarray) -> None:
        """Provide a 10 ms far-end playback frame for echo estimation."""
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
        """Release resources."""
        raise NotImplementedError

