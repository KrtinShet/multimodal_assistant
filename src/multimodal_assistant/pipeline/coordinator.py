from typing import Optional, TYPE_CHECKING
from multimodal_assistant.core.event_bus import EventBus, Event
from multimodal_assistant.core.streams import AsyncStream
from multimodal_assistant.engines.base import *
from multimodal_assistant.pipeline.output_handler import AudioOutputHandler, TextOutputHandler
from multimodal_assistant.utils.performance import PerformanceMonitor
from multimodal_assistant.utils.logger import setup_logger
import asyncio

if TYPE_CHECKING:
    from multimodal_assistant.pipeline.input_handler import AudioInputHandler

class PipelineCoordinator:
    """Orchestrates the complete pipeline"""

    def __init__(
        self,
        stt_engine: ISTTEngine,
        vision_engine: IVisionEngine,
        llm_engine: ILLMEngine,
        tts_engine: ITTSEngine,
        event_bus: EventBus,
        audio_output: AudioOutputHandler | None = None,
        text_output: TextOutputHandler | None = None,
        perf_monitor: PerformanceMonitor | None = None,
        audio_input: "AudioInputHandler | None" = None
    ):
        self.stt = stt_engine
        self.vision = vision_engine
        self.llm = llm_engine
        self.tts = tts_engine
        self.event_bus = event_bus
        self.audio_output = audio_output
        self.text_output = text_output or TextOutputHandler()
        self.perf_monitor = perf_monitor
        self.audio_input = audio_input
        self.logger = setup_logger("multimodal_assistant.coordinator")

    async def process_multimodal(
        self,
        audio_stream: AsyncStream[AudioChunk],
        video_stream: Optional[AsyncStream[ImageFrame]] = None
    ):
        """Process multimodal input through pipeline"""
        self.logger.debug("Starting multimodal processing")

        transcription_task = asyncio.create_task(
            self._process_audio(audio_stream)
        )

        vision_task = None
        if video_stream:
            self.logger.debug("Video stream detected, starting vision processing")
            vision_task = asyncio.create_task(
                self._process_vision(video_stream)
            )

        generator_task = None

        try:
            transcription = await transcription_task
            vision_embedding = await vision_task if vision_task else None

            if not transcription.strip():
                self.logger.debug("Empty transcription, skipping response generation")
                return

            self.logger.debug(f"Processing complete - Transcription: '{transcription[:50]}...' | Vision: {vision_embedding is not None}")

            text_stream, tts_stream, generator_task = await self._generate_response(
                transcription,
                vision_embedding
            )

            await asyncio.gather(
                self._display_text(text_stream),
                self._synthesize_speech(tts_stream)
            )
        except asyncio.CancelledError:
            self.logger.debug("Processing cancelled")
            transcription_task.cancel()
            if vision_task:
                vision_task.cancel()
            if generator_task:
                generator_task.cancel()
            raise
        except Exception as exc:
            self.logger.error(f"Error in multimodal processing: {exc}", exc_info=True)
            raise
        finally:
            if generator_task:
                try:
                    await generator_task
                except asyncio.CancelledError:
                    pass
                except Exception as exc:
                    self.logger.error(f"Error in response generator: {exc}", exc_info=True)

    async def _process_audio(
        self,
        audio_stream: AsyncStream[AudioChunk]
    ) -> str:
        """Process audio to text"""
        self.logger.debug("Starting STT processing")
        if self.perf_monitor:
            self.perf_monitor.start_timer("stt")

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
                self.logger.debug(f"STT chunk: {transcription.text.strip()}")

        full_text = " ".join(transcriptions).strip()

        if self.perf_monitor:
            latency = self.perf_monitor.end_timer("stt")
            self.logger.debug(f"STT completed in {latency:.2f}ms")

        await self.event_bus.publish(Event(
            event_type="stt_complete",
            timestamp=asyncio.get_event_loop().time(),
            payload={"text": full_text},
            source="coordinator"
        ))

        self.logger.info(f"Transcription: {full_text}")
        return full_text

    async def _process_vision(
        self,
        video_stream: AsyncStream[ImageFrame]
    ) -> VisionEmbedding:
        """Process latest frame"""
        self.logger.debug("Starting vision processing")
        if self.perf_monitor:
            self.perf_monitor.start_timer("vision")

        latest_frame = None
        frame_count = 0

        while True:
            frame = await video_stream.next()
            if frame is None:
                break
            latest_frame = frame
            frame_count += 1

        self.logger.debug(f"Processed {frame_count} frames, using latest")

        if latest_frame:
            embedding = await self.vision.encode_image(latest_frame)

            if self.perf_monitor:
                latency = self.perf_monitor.end_timer("vision")
                self.logger.debug(f"Vision encoding completed in {latency:.2f}ms")

            self.logger.info("Vision embedding generated")
            return embedding

        if self.perf_monitor:
            self.perf_monitor.end_timer("vision")

        self.logger.warning("No frames available for vision processing")
        return None

    async def _generate_response(
        self,
        text: str,
        vision_embedding: Optional[VisionEmbedding]
    ) -> tuple[AsyncStream[str], AsyncStream[str], asyncio.Task]:
        """Generate LLM response"""
        self.logger.debug(f"Starting LLM generation with vision={'enabled' if vision_embedding else 'disabled'}")
        if self.perf_monitor:
            self.perf_monitor.start_timer("llm")

        await self.event_bus.publish(Event(
            event_type="llm_started",
            timestamp=asyncio.get_event_loop().time(),
            payload={"prompt": text},
            source="coordinator"
        ))

        text_stream = AsyncStream[str]()
        tts_stream = AsyncStream[str]()

        async def _generate():
            token_count = 0
            async for token in self.llm.generate_stream(text, vision_embedding):
                token_count += 1
                await asyncio.gather(
                    text_stream.put(token),
                    tts_stream.put(token)
                )

            if self.perf_monitor:
                latency = self.perf_monitor.end_timer("llm")
                self.logger.debug(f"LLM generation completed: {token_count} tokens in {latency:.2f}ms")

            await asyncio.gather(
                text_stream.close(),
                tts_stream.close()
            )

        generator_task = asyncio.create_task(_generate())
        return text_stream, tts_stream, generator_task

    async def _display_text(self, text_stream: AsyncStream[str]):
        """Display text as it's generated"""
        self.logger.debug("Starting text display")
        await self.text_output.display_stream(text_stream)

    async def _synthesize_speech(self, text_stream: AsyncStream[str]):
        """Synthesize and play speech"""
        self.logger.debug("Starting TTS synthesis")
        if self.perf_monitor:
            self.perf_monitor.start_timer("tts")

        audio_stream = self.tts.synthesize_stream(text_stream)

        if self.audio_output is None:
            self.logger.debug("No audio output configured, consuming stream silently")
            async for _ in audio_stream:
                pass
            if self.perf_monitor:
                self.perf_monitor.end_timer("tts")
            return

        # Mute microphone during TTS playback to prevent echo
        if self.audio_input:
            self.audio_input.mute()

        try:
            await self.audio_output.start_playback(audio_stream)
        finally:
            # Unmute microphone after TTS completes
            if self.audio_input:
                self.audio_input.unmute()

        if self.perf_monitor:
            latency = self.perf_monitor.end_timer("tts")
            self.logger.debug(f"TTS synthesis completed in {latency:.2f}ms")
