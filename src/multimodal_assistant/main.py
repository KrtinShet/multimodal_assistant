import asyncio
from multimodal_assistant.core.event_bus import EventBus
from multimodal_assistant.engines.stt_engine import FasterWhisperEngine
from multimodal_assistant.engines.vision_engine import CLIPVisionEngine
from multimodal_assistant.engines.llm_engine import OllamaLLMEngine
from multimodal_assistant.engines.tts_engine import KokoroTTSEngine
from multimodal_assistant.pipeline.input_handler import AudioInputHandler, VideoInputHandler
from multimodal_assistant.pipeline.coordinator import PipelineCoordinator

class MultimodalAssistant:
    """Main application class"""

    def __init__(self):
        # Initialize components
        self.event_bus = EventBus()
        self.stt = FasterWhisperEngine(model_size="small")
        self.vision = CLIPVisionEngine()
        self.llm = OllamaLLMEngine(model_name="gemma3:4b")
        self.tts = KokoroTTSEngine()

        self.audio_input = AudioInputHandler()
        self.video_input = VideoInputHandler(fps=1)

        self.coordinator = PipelineCoordinator(
            stt_engine=self.stt,
            vision_engine=self.vision,
            llm_engine=self.llm,
            tts_engine=self.tts,
            event_bus=self.event_bus
        )

    async def initialize(self):
        """Initialize all engines"""
        print("Initializing engines...")
        await self.event_bus.start()

        await asyncio.gather(
            self.stt.initialize(),
            self.vision.initialize(),
            self.llm.initialize(),
            self.tts.initialize()
        )

        print("All engines ready!")

    async def run(self):
        """Run the assistant"""
        # Start input capture
        audio_stream = await self.audio_input.start_capture()
        video_stream = await self.video_input.start_capture()

        print("Listening... (Press Ctrl+C to stop)")

        try:
            # Process indefinitely
            while True:
                await self.coordinator.process_multimodal(
                    audio_stream,
                    video_stream
                )
                await asyncio.sleep(0.1)

        except KeyboardInterrupt:
            print("\nShutting down...")
            await self.shutdown()

    async def shutdown(self):
        """Cleanup"""
        await self.audio_input.stop_capture()
        await self.video_input.stop_capture()

        await asyncio.gather(
            self.stt.shutdown(),
            self.vision.shutdown(),
            self.llm.shutdown(),
            self.tts.shutdown()
        )

        await self.event_bus.stop()

async def main():
    assistant = MultimodalAssistant()
    await assistant.initialize()
    await assistant.run()

if __name__ == "__main__":
    asyncio.run(main())
