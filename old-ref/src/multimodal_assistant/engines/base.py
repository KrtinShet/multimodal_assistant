from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class AudioChunk:
    """Audio data container"""
    data: np.ndarray  # PCM float32
    sample_rate: int
    timestamp: float
    is_speech: bool = True

@dataclass
class ImageFrame:
    """Image data container"""
    data: np.ndarray  # RGB uint8
    timestamp: float
    frame_id: str

@dataclass
class Transcription:
    """STT output"""
    text: str
    confidence: float
    is_final: bool
    timestamp: float

@dataclass
class VisionEmbedding:
    """Vision encoder output"""
    embedding: np.ndarray
    timestamp: float
    image_id: str

class ISTTEngine(ABC):
    """Speech-to-Text engine interface"""

    @abstractmethod
    async def initialize(self): pass

    @abstractmethod
    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[Transcription]: pass

    @abstractmethod
    async def shutdown(self): pass

class IVisionEngine(ABC):
    """Vision encoding engine interface"""

    @abstractmethod
    async def initialize(self): pass

    @abstractmethod
    async def encode_image(
        self,
        frame: ImageFrame
    ) -> VisionEmbedding: pass

    @abstractmethod
    async def shutdown(self): pass

class ILLMEngine(ABC):
    """Large Language Model engine interface"""

    @abstractmethod
    async def initialize(self): pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        vision_embedding: Optional[VisionEmbedding] = None
    ) -> AsyncIterator[str]: pass

    @abstractmethod
    async def shutdown(self): pass

class ITTSEngine(ABC):
    """Text-to-Speech engine interface"""

    @abstractmethod
    async def initialize(self): pass

    @abstractmethod
    async def synthesize_stream(
        self,
        text_stream: AsyncIterator[str]
    ) -> AsyncIterator[AudioChunk]: pass

    @abstractmethod
    async def shutdown(self): pass
