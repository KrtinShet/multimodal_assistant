import webrtcvad
import numpy as np

class VADProcessor:
    """Voice Activity Detection using WebRTC VAD"""

    def __init__(self, aggressiveness: int = 3):
        self.vad = webrtcvad.Vad(aggressiveness)

    def is_speech(self, audio_data: np.ndarray, sample_rate: int) -> bool:
        """Detect if audio chunk contains speech.

        WebRTC VAD expects exactly 10/20/30 ms 16‑bit PCM frames at
        8, 16, or 32 kHz. We normalize the buffer to 10 ms and never
        default to speech on errors to avoid self‑triggering.
        """
        # Convert float32 to int16
        audio_int16 = (audio_data * 32768).astype(np.int16, copy=False)

        # Use 10 ms frames across the pipeline
        frame_ms = 10
        frame_len = int(sample_rate * (frame_ms / 1000.0))

        if audio_int16.size != frame_len:
            if audio_int16.size > frame_len:
                audio_int16 = audio_int16[-frame_len:]
            else:
                tmp = np.zeros(frame_len, dtype=np.int16)
                n = min(frame_len, audio_int16.size)
                if n:
                    tmp[:n] = audio_int16[:n]
                audio_int16 = tmp

        try:
            return self.vad.is_speech(audio_int16.tobytes(), sample_rate)
        except Exception:
            # Be conservative on errors
            return False
