"""Improved Voice Activity Detection with adaptive features."""

import numpy as np
import webrtcvad
from collections import deque
from dataclasses import dataclass
from typing import Optional
from assistant.utils.logger import setup_logger


@dataclass
class VADMetrics:
    """Metrics for VAD performance monitoring."""
    total_frames: int = 0
    speech_frames: int = 0
    noise_frames: int = 0
    current_noise_floor: float = 0.0
    energy_threshold: float = 0.0
    speech_ratio: float = 0.0


class ImprovedVADProcessor:
    """Advanced Voice Activity Detection with adaptive noise floor estimation.

    Features:
    - WebRTC VAD for robust speech detection
    - Adaptive noise floor estimation
    - Energy-based pre-filtering for efficiency
    - Smoothing with majority voting to prevent jitter
    - Configurable sensitivity and adaptation
    """

    def __init__(
        self,
        aggressiveness: int = 3,
        sample_rate: int = 16000,
        energy_threshold_ratio: float = 2.5,
        noise_floor_adaptation_rate: float = 0.01,
        smoothing_window: int = 5,
        min_energy_threshold: float = 0.005,
        log_level: str = "INFO",
    ):
        """Initialize improved VAD processor.

        Args:
            aggressiveness: WebRTC VAD aggressiveness (0-3, higher = more aggressive)
            sample_rate: Audio sample rate in Hz
            energy_threshold_ratio: Multiplier above noise floor for speech detection
            noise_floor_adaptation_rate: How quickly noise floor adapts (0-1)
            smoothing_window: Number of frames for majority voting
            min_energy_threshold: Minimum energy threshold to prevent false positives
            log_level: Logging level
        """
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.energy_threshold_ratio = energy_threshold_ratio
        self.noise_floor_adaptation_rate = noise_floor_adaptation_rate
        self.smoothing_window = smoothing_window
        self.min_energy_threshold = min_energy_threshold

        # Adaptive noise floor tracking
        self.noise_floor = 0.01  # Initial estimate
        self.noise_floor_buffer = deque(maxlen=50)  # Buffer for noise estimation

        # Smoothing buffer for majority voting
        self.recent_decisions = deque(maxlen=smoothing_window)

        # Metrics tracking
        self.metrics = VADMetrics()

        # Logger
        self.logger = setup_logger("assistant.processors.vad", log_level)
        self.logger.info(
            f"VAD initialized: aggressiveness={aggressiveness}, "
            f"energy_ratio={energy_threshold_ratio}, "
            f"smoothing_window={smoothing_window}"
        )

    def calculate_energy(self, audio_data: np.ndarray) -> float:
        """Calculate RMS energy of audio frame.

        Args:
            audio_data: Audio samples (float32, -1 to 1)

        Returns:
            RMS energy value
        """
        return float(np.sqrt(np.mean(audio_data ** 2)))

    def update_noise_floor(self, energy: float, is_speech: bool):
        """Update adaptive noise floor estimation.

        Args:
            energy: Current frame energy
            is_speech: Whether current frame is classified as speech
        """
        # Only update noise floor during non-speech periods
        if not is_speech:
            self.noise_floor_buffer.append(energy)

            # Update noise floor as moving average of recent non-speech frames
            if len(self.noise_floor_buffer) >= 10:
                recent_noise = np.percentile(list(self.noise_floor_buffer), 50)
                # Smooth adaptation
                self.noise_floor = (
                    (1 - self.noise_floor_adaptation_rate) * self.noise_floor +
                    self.noise_floor_adaptation_rate * recent_noise
                )

    def webrtc_vad_check(self, audio_data: np.ndarray) -> bool:
        """Run WebRTC VAD on audio frame.

        Args:
            audio_data: Audio samples (float32, -1 to 1)

        Returns:
            True if speech detected, False otherwise
        """
        # Convert float32 to int16 for WebRTC VAD
        audio_int16 = (audio_data * 32768).astype(np.int16, copy=False)

        # WebRTC VAD requires exactly 10/20/30 ms frames
        frame_ms = 10
        frame_len = int(self.sample_rate * (frame_ms / 1000.0))

        # Normalize to correct frame size
        if audio_int16.size != frame_len:
            if audio_int16.size > frame_len:
                # Take last frame_len samples
                audio_int16 = audio_int16[-frame_len:]
            else:
                # Pad with zeros
                tmp = np.zeros(frame_len, dtype=np.int16)
                n = min(frame_len, audio_int16.size)
                if n:
                    tmp[:n] = audio_int16[:n]
                audio_int16 = tmp

        try:
            return self.vad.is_speech(audio_int16.tobytes(), self.sample_rate)
        except Exception as e:
            self.logger.warning(f"WebRTC VAD error: {e}")
            return False

    def is_speech(self, audio_data: np.ndarray) -> bool:
        """Detect if audio chunk contains speech using multi-stage detection.

        Args:
            audio_data: Audio samples (float32, -1 to 1)

        Returns:
            True if speech detected, False otherwise
        """
        # Stage 1: Energy-based pre-filtering (fast rejection)
        energy = self.calculate_energy(audio_data)
        dynamic_threshold = max(
            self.min_energy_threshold,
            self.noise_floor * self.energy_threshold_ratio
        )

        # Quick rejection if energy too low
        if energy < dynamic_threshold:
            raw_decision = False
        else:
            # Stage 2: WebRTC VAD (more sophisticated)
            raw_decision = self.webrtc_vad_check(audio_data)

        # Stage 3: Smoothing with majority voting
        self.recent_decisions.append(raw_decision)

        # Require majority vote for final decision
        if len(self.recent_decisions) >= 3:
            speech_count = sum(self.recent_decisions)
            smoothed_decision = speech_count > (len(self.recent_decisions) // 2)
        else:
            smoothed_decision = raw_decision

        # Update noise floor
        self.update_noise_floor(energy, smoothed_decision)

        # Update metrics
        self.metrics.total_frames += 1
        if smoothed_decision:
            self.metrics.speech_frames += 1
        else:
            self.metrics.noise_frames += 1

        self.metrics.current_noise_floor = self.noise_floor
        self.metrics.energy_threshold = dynamic_threshold
        if self.metrics.total_frames > 0:
            self.metrics.speech_ratio = (
                self.metrics.speech_frames / self.metrics.total_frames
            )

        return smoothed_decision

    def reset(self):
        """Reset VAD state and metrics."""
        self.noise_floor = 0.01
        self.noise_floor_buffer.clear()
        self.recent_decisions.clear()
        self.metrics = VADMetrics()
        self.logger.debug("VAD state reset")

    def get_metrics(self) -> VADMetrics:
        """Get current VAD metrics.

        Returns:
            Current VAD metrics
        """
        return self.metrics

    def set_aggressiveness(self, level: int):
        """Change VAD aggressiveness level.

        Args:
            level: Aggressiveness level (0-3)
        """
        if 0 <= level <= 3:
            self.vad = webrtcvad.Vad(level)
            self.logger.info(f"VAD aggressiveness set to {level}")
        else:
            self.logger.warning(f"Invalid aggressiveness level: {level}")
