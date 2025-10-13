from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np

from multimodal_assistant.config.settings import Settings
from multimodal_assistant.core.event_bus import EventBus, Event
from multimodal_assistant.core.streams import AsyncStream
from multimodal_assistant.engines.base import (
    ISTTEngine,
    IVisionEngine,
    ILLMEngine,
    ITTSEngine,
    AudioChunk,
    ImageFrame,
    VisionEmbedding,
)
from multimodal_assistant.pipeline.output_handler import AudioOutputHandler, TextOutputHandler
from multimodal_assistant.utils.logger import setup_logger
from multimodal_assistant.utils.performance import PerformanceMonitor

if TYPE_CHECKING:
    from multimodal_assistant.pipeline.input_handler import AudioInputHandler


@dataclass
class InputEvent:
    """Microphone-derived event pushed into the coordinator loop."""

    event_type: str
    audio: Optional[np.ndarray] = None


@dataclass
class ResponseSession:
    """Book-keeping for an in-flight assistant response."""

    user_text: str
    generated_text: str = ""
    vision_embedding: Optional[VisionEmbedding] = None
    text_stream: Optional[AsyncStream[str]] = None
    tts_text_stream: Optional[AsyncStream[str]] = None
    generator_task: Optional[asyncio.Task] = None
    text_task: Optional[asyncio.Task] = None
    audio_task: Optional[asyncio.Task] = None
    monitor_task: Optional[asyncio.Task] = None


class PipelineCoordinator:
    """Event-driven orchestrator for streaming audio, LLM, and TTS."""

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
        audio_input: "AudioInputHandler | None" = None,
        settings: Settings | None = None,
        audio_enabled_callback=None,
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
        self.settings = settings or Settings()
        self.audio_enabled_callback = audio_enabled_callback

        self.logger = setup_logger("multimodal_assistant.coordinator")

        # Runtime state
        self._state: str = "idle"
        self._event_queue: asyncio.Queue[InputEvent] = asyncio.Queue()
        self._microphone_task: Optional[asyncio.Task] = None
        self._session: Optional[ResponseSession] = None
        self._resume_buffer: str = ""
        self._noise_active: bool = False
        self._barge_active: bool = False

        # Derived thresholds from settings (convert ms to seconds)
        self._min_speech_seconds = self.settings.min_user_speech_ms / 1000.0
        self._silence_seconds = self.settings.user_silence_ms / 1000.0
        self._noise_pause_seconds = self.settings.noise_pause_ms / 1000.0
        self._noise_resume_seconds = self.settings.noise_resume_ms / 1000.0
        self._noise_threshold = self.settings.noise_rms_threshold

    def _is_audio_enabled(self) -> bool:
        """Check if audio is enabled via callback."""
        if self.audio_enabled_callback:
            return self.audio_enabled_callback()
        return True  # Default to enabled if no callback

    async def run(
        self,
        audio_stream: AsyncStream[AudioChunk],
        video_stream: Optional[AsyncStream[ImageFrame]] = None,
    ):
        """Run the continuous coordinator loop."""

        if self.audio_input and self.audio_output:
            # Wrap output reference to match the input sample rate and duration
            def _ref_provider(num_samples: int):
                try:
                    from multimodal_assistant.audio.resample import resample_linear
                    in_rate = getattr(self.audio_input, "sample_rate", 16000)
                    out_rate = getattr(self.audio_output, "sample_rate", in_rate)
                    # Request a time-equivalent window from the output buffer
                    req = int(max(1, round(num_samples * (out_rate / float(in_rate)))))
                    ref = self.audio_output.get_reference_audio(req)
                    if ref.size == 0 or out_rate == in_rate:
                        return ref
                    return resample_linear(ref, out_rate, in_rate)
                except Exception:
                    # Fall back to raw window; gating will best-effort
                    return self.audio_output.get_reference_audio(num_samples)

            self.audio_input.set_reference_provider(_ref_provider)

        self._microphone_task = asyncio.create_task(
            self._microphone_loop(audio_stream)
        )

        try:
            while True:
                event = await self._event_queue.get()
                await self._handle_event(event, video_stream)
        except asyncio.CancelledError:
            self.logger.debug("Coordinator loop cancelled")
            raise
        finally:
            if self._microphone_task:
                self._microphone_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._microphone_task

    async def _handle_event(
        self,
        event: InputEvent,
        video_stream: Optional[AsyncStream[ImageFrame]],
    ):
        if event.event_type == "barge_in_start":
            await self._suspend_active_response()
            return

        if event.event_type == "noise_start":
            if self._state == "speaking" and self.audio_output:
                await self.audio_output.pause()
                self._state = "paused_by_noise"
            return

        if event.event_type == "noise_end":
            if self._state == "paused_by_noise" and self.audio_output:
                await self.audio_output.resume()
                self._state = "speaking"
            return

        if event.event_type != "utterance" or event.audio is None:
            return

        transcription = await self._transcribe_audio(event.audio)
        if not transcription:
            self.logger.debug("Ignoring empty transcription from utterance event")
            self._barge_active = False
            return

        self.logger.info("User said: %s", transcription)
        normalized = transcription.lower().strip()

        if "continue" in normalized and self._resume_buffer:
            await self._resume_response(transcription, video_stream)
            return

        # Any other utterance is treated as a new user turn.
        self._resume_buffer = ""
        await self._start_new_response(transcription, video_stream)

    async def _microphone_loop(self, audio_stream: AsyncStream[AudioChunk]):
        """Monitor the microphone stream for utterances and noise events."""

        speech_buffer: list[np.ndarray] = []
        speech_duration = 0.0
        silence_duration = 0.0
        in_speech = False
        noise_duration = 0.0
        silence_after_noise = 0.0

        while True:
            chunk = await audio_stream.next()
            if chunk is None:
                await asyncio.sleep(0.01)
                continue

            data = chunk.data.astype(np.float32, copy=False)
            chunk_duration = len(data) / float(chunk.sample_rate)
            energy = float(np.sqrt(np.mean(data**2) + 1e-9))

            # Detect the onset of user speech during assistant playback.
            if chunk.is_speech and not in_speech and self._state in {"speaking", "paused_by_noise"}:
                if not self._barge_active:
                    self._barge_active = True
                    await self._event_queue.put(InputEvent(event_type="barge_in_start"))

            if chunk.is_speech:
                speech_buffer.append(data)
                speech_duration += chunk_duration
                silence_duration = 0.0
                in_speech = True
            else:
                if in_speech:
                    silence_duration += chunk_duration
                    if speech_duration >= self._min_speech_seconds and silence_duration >= self._silence_seconds:
                        utterance = np.concatenate(speech_buffer)
                        await self._event_queue.put(
                            InputEvent(event_type="utterance", audio=utterance)
                        )
                        speech_buffer.clear()
                        speech_duration = 0.0
                        silence_duration = 0.0
                        in_speech = False
                        noise_duration = 0.0
                        self._barge_active = False
                        continue
                else:
                    speech_buffer.clear()
                    speech_duration = 0.0
                    silence_duration = 0.0

            # Noise-triggered pause/resume logic (non-speech energy).
            if not chunk.is_speech and energy >= self._noise_threshold:
                noise_duration += chunk_duration
                silence_after_noise = 0.0
                if (
                    not self._noise_active
                    and self._state == "speaking"
                    and noise_duration >= self._noise_pause_seconds
                ):
                    self._noise_active = True
                    await self._event_queue.put(InputEvent(event_type="noise_start"))
            else:
                if self._noise_active:
                    noise_duration = max(0.0, noise_duration - chunk_duration)
                    silence_after_noise += chunk_duration
                    should_resume = (
                        silence_after_noise >= self._noise_resume_seconds
                        or energy < (0.5 * self._noise_threshold)
                        or chunk.is_speech
                    )
                    if should_resume:
                        self._noise_active = False
                        silence_after_noise = 0.0
                        await self._event_queue.put(InputEvent(event_type="noise_end"))

    async def _suspend_active_response(self):
        """Pause playback and cache partial assistant text for resuming."""

        if not self._session:
            return

        if self.audio_output:
            await self.audio_output.pause()
            await self.audio_output.flush()

        await self._cancel_session_tasks(preserve_for_resume=True)
        self._state = "interrupted"

    async def _cancel_session_tasks(self, preserve_for_resume: bool):
        """Cancel in-flight response tasks and optionally keep text for resume."""

        session = self._session
        if not session:
            return

        tasks = [
            session.generator_task,
            session.text_task,
            session.audio_task,
            session.monitor_task,
        ]

        for task in tasks:
            if task and not task.done():
                task.cancel()

        for task in tasks:
            if task:
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        if preserve_for_resume:
            self._resume_buffer = session.generated_text.strip()
        else:
            self._resume_buffer = ""

        self._session = None

    async def _transcribe_audio(self, audio: np.ndarray) -> str:
        if self.perf_monitor:
            self.perf_monitor.start_timer("stt")

        await self.event_bus.publish(
            Event(
                event_type="stt_started",
                timestamp=asyncio.get_event_loop().time(),
                payload=None,
                source="coordinator",
            )
        )

        text = await self.stt.transcribe_audio(audio)

        if self.perf_monitor:
            latency = self.perf_monitor.end_timer("stt")
            self.logger.debug("STT latency: %.2fms", latency)

        await self.event_bus.publish(
            Event(
                event_type="stt_complete",
                timestamp=asyncio.get_event_loop().time(),
                payload={"text": text},
                source="coordinator",
            )
        )

        return text.strip()

    async def _start_new_response(
        self,
        user_text: str,
        video_stream: Optional[AsyncStream[ImageFrame]],
    ):
        """Start responding to a fresh user turn."""

        await self._cancel_session_tasks(preserve_for_resume=False)

        vision_embedding = None
        if video_stream:
            vision_embedding = await self._process_vision(video_stream)

        session = ResponseSession(user_text=user_text, vision_embedding=vision_embedding)
        self._session = session
        self._state = "generating"

        await self.event_bus.publish(
            Event(
                event_type="llm_started",
                timestamp=asyncio.get_event_loop().time(),
                payload={"prompt": user_text},
                source="coordinator",
            )
        )

        session.text_stream, session.tts_text_stream, session.generator_task = await self._generate_response(
            user_text,
            vision_embedding,
            session,
        )

        session.text_task = asyncio.create_task(self._display_text(session.text_stream))

        if self.audio_output and self._is_audio_enabled():
            self.logger.debug("Starting TTS synthesis and audio playback...")
            try:
                audio_stream = self.tts.synthesize_stream(session.tts_text_stream)
                self.logger.debug(f"TTS stream created: {audio_stream}")
                session.audio_task = asyncio.create_task(
                    self.audio_output.start_playback(audio_stream)
                )
                self.logger.debug(f"Audio playback task created: {session.audio_task}")
                self._state = "speaking"
            except Exception as e:
                self.logger.error(f"Failed to start audio playback: {e}", exc_info=True)
        else:
            if not self._is_audio_enabled():
                self.logger.debug("Audio is disabled - TTS will be silent")
            else:
                self.logger.warning("No audio_output configured - TTS will be silent")

        session.monitor_task = asyncio.create_task(self._monitor_session(session))

    async def _resume_response(
        self,
        user_text: str,
        video_stream: Optional[AsyncStream[ImageFrame]],
    ):
        """Continue a previously interrupted assistant response."""

        if not self._resume_buffer:
            await self._start_new_response(user_text, video_stream)
            return

        resume_prompt = (
            "The assistant was responding with the following text before it was "
            "interrupted:\n"
            f"{self._resume_buffer}\n\n"
            "Continue the response naturally from where it stopped without "
            "repeating prior sentences."
        )

        await self._start_new_response(resume_prompt, video_stream)
        self._resume_buffer = ""

    async def _monitor_session(self, session: ResponseSession):
        """Wait for response tasks to settle and reset state."""

        tasks = [
            session.generator_task,
            session.text_task,
            session.audio_task,
        ]

        try:
            for task in tasks:
                if task:
                    await task
        except asyncio.CancelledError:
            raise
        finally:
            if self._session is session:
                self._session = None
                self._state = "idle"
                self._resume_buffer = ""

            await self.event_bus.publish(
                Event(
                    event_type="response_complete",
                    timestamp=asyncio.get_event_loop().time(),
                    payload=None,
                    source="coordinator",
                )
            )

    async def _process_vision(
        self,
        video_stream: AsyncStream[ImageFrame],
    ) -> Optional[VisionEmbedding]:
        """Grab the most recent frame from the video stream and encode it."""

        latest_frame = None

        while True:
            frame = await video_stream.next()
            if frame is None:
                break
            latest_frame = frame

        if not latest_frame:
            return None

        if self.perf_monitor:
            self.perf_monitor.start_timer("vision")

        embedding = await self.vision.encode_image(latest_frame)

        if self.perf_monitor:
            latency = self.perf_monitor.end_timer("vision")
            self.logger.debug("Vision latency: %.2fms", latency)

        return embedding

    async def _generate_response(
        self,
        prompt: str,
        vision_embedding: Optional[VisionEmbedding],
        session: ResponseSession,
    ) -> tuple[AsyncStream[str], AsyncStream[str], asyncio.Task]:
        """Stream tokens from the LLM and mirror them to text and TTS streams."""

        if self.perf_monitor:
            self.perf_monitor.start_timer("llm")

        text_stream = AsyncStream[str]()
        tts_stream = AsyncStream[str]()

        async def _generate():
            token_count = 0
            try:
                async for token in self.llm.generate_stream(prompt, vision_embedding):
                    token_count += 1
                    session.generated_text += token
                    await asyncio.gather(
                        text_stream.put(token),
                        tts_stream.put(token),
                    )
            finally:
                await asyncio.gather(text_stream.close(), tts_stream.close())

                if self.perf_monitor:
                    latency = self.perf_monitor.end_timer("llm")
                    self.logger.debug(
                        "LLM streamed %s tokens in %.2fms", token_count, latency
                    )

        generator_task = asyncio.create_task(_generate())
        return text_stream, tts_stream, generator_task

    async def _display_text(self, text_stream: AsyncStream[str]):
        await self.text_output.display_stream(text_stream)
