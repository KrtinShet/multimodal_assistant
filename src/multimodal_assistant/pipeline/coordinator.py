from typing import Optional
from multimodal_assistant.core.event_bus import EventBus, Event
from multimodal_assistant.core.streams import AsyncStream
from multimodal_assistant.engines.base import *
import asyncio

class PipelineCoordinator:
    """Orchestrates the complete pipeline"""

    def __init__(
        self,
        stt_engine: ISTTEngine,
        vision_engine: IVisionEngine,
        llm_engine: ILLMEngine,
        tts_engine: ITTSEngine,
        event_bus: EventBus
    ):
        self.stt = stt_engine
        self.vision = vision_engine
        self.llm = llm_engine
        self.tts = tts_engine
        self.event_bus = event_bus

    async def process_multimodal(
        self,
        audio_stream: AsyncStream[AudioChunk],
        video_stream: Optional[AsyncStream[ImageFrame]] = None
    ):
        """Process multimodal input through pipeline"""

        # Step 1: Parallel STT and Vision processing
        transcription_task = asyncio.create_task(
            self._process_audio(audio_stream)
        )

        vision_task = None
        if video_stream:
            vision_task = asyncio.create_task(
                self._process_vision(video_stream)
            )

        # Wait for both to complete
        transcription = await transcription_task
        vision_embedding = await vision_task if vision_task else None

        # Step 2: LLM generation
        response_stream = await self._generate_response(
            transcription,
            vision_embedding
        )

        # Step 3: Parallel text display and TTS
        await asyncio.gather(
            self._display_text(response_stream),
            self._synthesize_speech(response_stream)
        )

    async def _process_audio(
        self,
        audio_stream: AsyncStream[AudioChunk]
    ) -> str:
        """Process audio to text"""
        await self.event_bus.publish(Event(
            event_type="stt_started",
            timestamp=asyncio.get_event_loop().time(),
            payload=None,
            source="coordinator"
        ))

        transcriptions = []
        async for transcription in self.stt.transcribe_stream(audio_stream):
            transcriptions.append(transcription.text)

        full_text = " ".join(transcriptions)

        await self.event_bus.publish(Event(
            event_type="stt_complete",
            timestamp=asyncio.get_event_loop().time(),
            payload={"text": full_text},
            source="coordinator"
        ))

        return full_text

    async def _process_vision(
        self,
        video_stream: AsyncStream[ImageFrame]
    ) -> VisionEmbedding:
        """Process latest frame"""
        latest_frame = None
        async for frame in video_stream:
            latest_frame = frame

        if latest_frame:
            return await self.vision.encode_image(latest_frame)
        return None

    async def _generate_response(
        self,
        text: str,
        vision_embedding: Optional[VisionEmbedding]
    ) -> AsyncStream[str]:
        """Generate LLM response"""
        await self.event_bus.publish(Event(
            event_type="llm_started",
            timestamp=asyncio.get_event_loop().time(),
            payload={"prompt": text},
            source="coordinator"
        ))

        response_stream = AsyncStream[str]()

        async def _generate():
            async for token in self.llm.generate_stream(text, vision_embedding):
                await response_stream.put(token)
            await response_stream.close()

        asyncio.create_task(_generate())
        return response_stream

    async def _display_text(self, text_stream: AsyncStream[str]):
        """Display text as it's generated"""
        async for token in text_stream:
            print(token, end='', flush=True)

    async def _synthesize_speech(self, text_stream: AsyncStream[str]):
        """Synthesize and play speech"""
        audio_stream = self.tts.synthesize_stream(text_stream)
        # Play audio (implementation depends on audio library)
        async for audio_chunk in audio_stream:
            # Play using sounddevice or other audio library
            pass
