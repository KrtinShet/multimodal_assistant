from dataclasses import dataclass

@dataclass
class Settings:
    # Audio settings
    audio_sample_rate: int = 16000
    audio_chunk_duration_ms: int = 100

    # Model settings
    stt_model_size: str = "small"
    llm_model_name: str = "gemma3:4b"
    vision_model_name: str = "openai/clip-vit-base-patch32"

    # Performance settings
    enable_mps: bool = True
    max_concurrent_streams: int = 10

    # Video settings
    video_fps: int = 1
    enable_vision: bool = True

    # VAD settings
    vad_aggressiveness: int = 3  # 0-3, higher = more aggressive

    # TTS settings
    tts_sample_rate: int = 24000
