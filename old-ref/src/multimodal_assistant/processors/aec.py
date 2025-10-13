"""Lightweight echo suppression utilities.

This module provides a placeholder acoustic echo cancellation (AEC) processor.
It exposes a consistent interface so that a full-featured backend (for example
WebRTC AudioProcessing) can be swapped in later without touching the rest of
the pipeline.  When the high-quality backend is unavailable, we fall back to a
correlation-based suppressor that attenuates strong echoes using a reference
signal supplied by the audio output handler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from multimodal_assistant.utils.logger import setup_logger


@dataclass
class _CorrelationConfig:
    """Configuration for the fallback correlation-based suppressor."""

    suppression_strength: float = 0.6
    correlation_threshold: float = 0.3
    min_reference_rms: float = 1e-4


class AECProcessor:
    """Apply acoustic echo suppression to microphone frames.

    Parameters
    ----------
    sample_rate:
        Audio sample rate expected by the processor.
    frame_size:
        Number of samples per frame.
    enabled:
        If ``False`` the processor is bypassed entirely.
    suppression_strength:
        Scaling factor applied to the estimated echo when using the fallback
        correlation-based suppressor. Values in ``[0, 1]`` work best.
    """

    def __init__(
        self,
        sample_rate: int,
        frame_size: int,
        enabled: bool = True,
        suppression_strength: float = 0.6,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.enabled = enabled
        self.logger = setup_logger("multimodal_assistant.processors.aec")

        self._correlation_cfg = _CorrelationConfig(
            suppression_strength=max(0.0, min(1.0, suppression_strength))
        )
        self._last_correlation: float = 0.0
        self._last_residual_ratio: float = 1.0

    def process(
        self,
        capture_frame: np.ndarray,
        reference_window: Optional[np.ndarray],
    ) -> np.ndarray:
        """Return capture_frame with far-end reference attenuated.

        The processor always returns a ``float32`` array of the original frame
        size. If echo suppression is disabled or no reference is available, the
        capture frame is returned unchanged.
        """

        if not self.enabled:
            self._last_correlation = 0.0
            self._last_residual_ratio = 1.0
            return capture_frame

        if reference_window is None or reference_window.size == 0:
            self._last_correlation = 0.0
            self._last_residual_ratio = 1.0
            return capture_frame

        if capture_frame.size != self.frame_size:
            capture = np.zeros(self.frame_size, dtype=np.float32)
            capture[: capture_frame.size] = capture_frame
        else:
            capture = capture_frame.astype(np.float32, copy=False)

        reference = reference_window.astype(np.float32, copy=False)

        # Compute root-mean-square energy for gating of capture.
        rms_capture = float(np.sqrt(np.mean(capture**2) + 1e-9))
        if rms_capture == 0.0:
            self._last_correlation = 0.0
            self._last_residual_ratio = 1.0
            return capture

        # Search best alignment in reference window (lag-aware correlation).
        if reference.size < self.frame_size:
            # Not enough reference; fall back to passthrough
            self._last_correlation = 0.0
            self._last_residual_ratio = 1.0
            return capture

        best_corr = 0.0
        best_seg = None
        best_gain = 0.0

        # Slide over the last N samples of reference and find the most correlated segment.
        end = reference.size
        start_min = max(0, end - 20 * self.frame_size)  # limit search to ~200ms
        for start in range(start_min, end - self.frame_size + 1):
            seg = reference[start : start + self.frame_size]
            rms_seg = float(np.sqrt(np.mean(seg**2) + 1e-9))
            if rms_seg < self._correlation_cfg.min_reference_rms:
                continue
            corr = float(np.dot(capture, seg)) / (
                self.frame_size * rms_capture * rms_seg + 1e-9
            )
            acorr = abs(corr)
            if acorr > best_corr:
                # Least-squares gain estimate for subtractive suppression
                gain = float(np.dot(capture, seg) / (np.dot(seg, seg) + 1e-9))
                best_corr = acorr
                best_seg = seg
                best_gain = gain

        self._last_correlation = best_corr

        if best_seg is None or best_corr < self._correlation_cfg.correlation_threshold:
            self._last_residual_ratio = 1.0
            return capture

        # Subtractive suppression using the best-aligned reference segment.
        cleaned = capture - self._correlation_cfg.suppression_strength * best_gain * best_seg

        # Residual energy relative to the echo segment energy.
        residual_rms = float(np.sqrt(np.mean(cleaned**2) + 1e-9))
        rms_best = float(np.sqrt(np.mean(best_seg**2) + 1e-9))
        self._last_residual_ratio = residual_rms / (rms_best + 1e-9)

        return cleaned.astype(np.float32, copy=False)

    def is_echo(self, correlation_threshold: float, residual_ratio: float) -> bool:
        if not self.enabled:
            return False

        return (
            self._last_correlation >= correlation_threshold
            and self._last_residual_ratio <= residual_ratio
        )
