from typing import Optional
from multimodal_assistant.core.event_bus import EventBus, Event
from multimodal_assistant.core.streams import AsyncStream
from multimodal_assistant.engines.base import *
from multimodal_assistant.pipeline.output_handler import AudioOutputHandler
import asyncio

class PipelineCoordinator:
    """Orchestrates the complete pipeline"""

    def __init__(
        self,
        stt_engine: ISTTEngine,
        vision_engine: IVisionEngine,
        llm_engine: ILLMEngine,
        tts_engine: ITTSEngine,
        event_bus: EventBus,
        audio_output: AudioOutputHandler | None = None
    ):
        self.stt = stt_engine
        self.vision = vision_engine
        self.llm = llm_engine
        self.tts = tts_engine
        self.event_bus = event_bus
        self.audio_output = audio_output

    async def process_multimodal(
        self,
        audio_stream: AsyncStream[AudioChunk],
        video_stream: Optional[AsyncStream[ImageFrame]] = None
    ):
        """Process multimodal input through pipeline"""

        transcription_task = asyncio.create_task(
            self._process_audio(audio_stream)
        )

        vision_task = None
        if video_stream:
            vision_task = asyncio.create_task(
                self._process_vision(video_stream)
            )

        generator_task = None

        try:
            transcription = await transcription_task
            vision_embedding = await vision_task if vision_task else None

            if not transcription.strip():
                return

            text_stream, tts_stream, generator_task = await self._generate_response(
                transcription,
                vision_embedding
            )

            await asyncio.gather(
                self._display_text(text_stream),
                self._synthesize_speech(tts_stream)
            )
        except asyncio.CancelledError:
            transcription_task.cancel()
            if vision_task:
                vision_task.cancel()
            if generator_task:
                generator_task.cancel()
            raise
        finally:
            if generator_task:
                try:
                    await generator_task
                except asyncio.CancelledError:
                    pass
                except Exception as exc:
                    print(f"Error in response generator: {exc}")

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
            if transcription.text:
                transcriptions.append(transcription.text.strip())

        full_text = " ".join(transcriptions).strip()

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

        while True:
            frame = await video_stream.next()
            if frame is None:
                break
            latest_frame = frame

        if latest_frame:
            return await self.vision.encode_image(latest_frame)
        return None

    async def _generate_response(
        self,
        text: str,
        vision_embedding: Optional[VisionEmbedding]
    ) -> tuple[AsyncStream[str], AsyncStream[str], asyncio.Task]:
        """Generate LLM response"""
        await self.event_bus.publish(Event(
            event_type="llm_started",
            timestamp=asyncio.get_event_loop().time(),
            payload={"prompt": text},
            source="coordinator"
        ))

        text_stream = AsyncStream[str]()
        tts_stream = AsyncStream[str]()

        async def _generate():
            async for token in self.llm.generate_stream(text, vision_embedding):
                await asyncio.gather(
                    text_stream.put(token),
                    tts_stream.put(token)
                )
            await asyncio.gather(
                text_stream.close(),
                tts_stream.close()
            )

        generator_task = asyncio.create_task(_generate())
        return text_stream, tts_stream, generator_task

    async def _display_text(self, text_stream: AsyncStream[str]):
        """Display text as it's generated"""
        async for token in text_stream:
            print(token, end='', flush=True)

        print()

    async def _synthesize_speech(self, text_stream: AsyncStream[str]):
        """Synthesize and play speech"""
        audio_stream = self.tts.synthesize_stream(text_stream)
        if self.audio_output is None:
            async for _ in audio_stream:
                pass
            return

        await self.audio_output.start_playback(audio_stream)
