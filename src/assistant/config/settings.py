from dataclasses import dataclass

@dataclass
class Settings:
    # Audio settings
    audio_sample_rate: int = 16000
    audio_chunk_duration_ms: int = 100
    audio_frame_duration_ms: int = 10

    # Model settings
    stt_model_size: str = "small"
    llm_model_name: str = "gemma3:4b"
    vision_model_name: str = "openai/clip-vit-base-patch32"

    # Performance settings
    enable_mps: bool = True
    max_concurrent_streams: int = 10

    # Video settings
    video_fps: int = 1
    enable_vision: bool = False

    # VAD / audio frontend settings
    vad_aggressiveness: int = 3  # 0-3, higher = more aggressive filtering
    vad_energy_threshold_ratio: float = 2.5  # Multiplier above noise floor for speech
    vad_noise_floor_adaptation_rate: float = 0.01  # How quickly noise floor adapts (0-1)
    vad_smoothing_window: int = 5  # Frames for majority voting to prevent jitter

    # AEC backend selection
    aec_backend: str = "webrtc_apm"  # "webrtc_apm" | "fallback"
    enable_aec: bool = True
    echo_suppression_strength: float = 0.6  # for fallback AEC only
    aec_reference_window_s: float = 2.0     # for fallback AEC only
    playback_frame_duration_ms: int = 20
    min_user_speech_ms: int = 250
    user_silence_ms: int = 320
    noise_pause_ms: int = 120
    noise_resume_ms: int = 450
    noise_rms_threshold: float = 0.015  # Minimum energy threshold for VAD
    echo_gate_correlation: float = 0.35
    echo_gate_near_far_ratio: float = 0.4

    # STT real-time settings
    stt_pre_roll_ms: int = 300  # Pre-roll buffer to capture speech start
    # WebRTC APM tuning
    apm_sample_rate_hz: int = 16000
    apm_stream_delay_ms: int = 60
    apm_enable_ns: bool = True
    apm_enable_agc: bool = True
    apm_enable_hpf: bool = True
    apm_lib_path: str | None = "/Users/krtinshet/Development/vera-assistant/native/webrtc_apm/dist/libapm_shim.dylib"

    # TTS settings
    tts_sample_rate: int = 24000
    # Optional audio device selection (sounddevice device id or name)
    audio_input_device: int | str | None = None
    audio_output_device: int | str | None = None

    # Logging settings
    log_level: str = "DEBUG"  # "DEBUG", "INFO", "WARNING", "ERROR"

    # LLM settings
    system_prompt: str = (
        "You are Lucy, a helpful voice assistant. "
        "Respond naturally and conversationally. "
        "Keep responses concise and clear. "
        "Do not use markdown formatting, emojis, or special characters. "
        "Speak as if having a natural conversation."
    )
    temperature: float = 0.7
    top_p: float = 0.9

    # Agent settings - Backend Configuration
    llm_backend: str = "ollama"  # "ollama" or "openai"
    memory_backend: str = "chroma"  # "chroma" or "inmemory"
    checkpointer_backend: str = "sqlite"  # "sqlite" or "memory"

    # Agent settings - LLM Configuration
    agent_model_provider: str = "ollama"  # "ollama" or "openai" (deprecated, use llm_backend)
    agent_model_name: str = "gemma3:4b"  # Model name for the provider
    openai_api_key: str | None = None  # OpenAI API key (if using OpenAI)
    openai_base_url: str | None = None  # Optional custom OpenAI base URL

    # Memory and persistence settings
    checkpoint_db_path: str = "./data/checkpoints.db"
    chroma_db_path: str = "./data/chroma_db"
    memory_collection_name: str = "agent_memories"
    memory_retrieval_limit: int = 5  # Number of memories to retrieve
    enable_semantic_memory: bool = True

    # Human-in-the-loop settings
    enable_human_approval: bool = True
    require_approval_for_tools: list[str] | None = None  # Tool names requiring approval
