import webrtcvad
import numpy as np

class VADProcessor:
    """Voice Activity Detection using WebRTC VAD"""

    def __init__(self, aggressiveness: int = 3):
        self.vad = webrtcvad.Vad(aggressiveness)

    def is_speech(self, audio_data: np.ndarray, sample_rate: int) -> bool:
        """Detect if audio chunk contains speech"""
        # Convert float32 to int16
        audio_int16 = (audio_data * 32768).astype(np.int16)

        # VAD requires 10, 20, or 30ms frames at 8, 16, 32kHz
        frame_duration = 30  # ms

        try:
            return self.vad.is_speech(
                audio_int16.tobytes(),
                sample_rate
            )
        except:
            return True  # Default to speech if detection fails
