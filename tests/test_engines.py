import pytest
from multimodal_assistant.engines.base import AudioChunk, ImageFrame, Transcription, VisionEmbedding
import numpy as np

def test_audio_chunk_creation():
    """Test AudioChunk dataclass"""
    data = np.zeros(1600, dtype=np.float32)
    chunk = AudioChunk(
        data=data,
        sample_rate=16000,
        timestamp=0.0,
        is_speech=True
    )

    assert chunk.sample_rate == 16000
    assert chunk.is_speech is True
    assert len(chunk.data) == 1600

def test_image_frame_creation():
    """Test ImageFrame dataclass"""
    data = np.zeros((224, 224, 3), dtype=np.uint8)
    frame = ImageFrame(
        data=data,
        timestamp=0.0,
        frame_id="frame_0"
    )

    assert frame.frame_id == "frame_0"
    assert frame.data.shape == (224, 224, 3)

def test_transcription_creation():
    """Test Transcription dataclass"""
    trans = Transcription(
        text="Hello world",
        confidence=0.95,
        is_final=True,
        timestamp=0.0
    )

    assert trans.text == "Hello world"
    assert trans.confidence == 0.95
    assert trans.is_final is True

def test_vision_embedding_creation():
    """Test VisionEmbedding dataclass"""
    embedding = VisionEmbedding(
        embedding=np.zeros(512),
        timestamp=0.0,
        image_id="img_0"
    )

    assert len(embedding.embedding) == 512
    assert embedding.image_id == "img_0"
