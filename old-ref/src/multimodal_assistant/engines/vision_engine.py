import torch
from transformers import CLIPProcessor, CLIPVisionModel
from .base import IVisionEngine, ImageFrame, VisionEmbedding
import asyncio
from PIL import Image
from multimodal_assistant.utils.logger import setup_logger

class CLIPVisionEngine(IVisionEngine):
    """CLIP vision encoder with MPS acceleration"""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.logger = setup_logger("multimodal_assistant.engines.vision")

    async def initialize(self):
        """Load CLIP model"""
        self.logger.info(f"Initializing CLIP Vision (model={self.model_name}, device={self.device})")
        loop = asyncio.get_event_loop()

        def _load():
            processor = CLIPProcessor.from_pretrained(self.model_name)
            model = CLIPVisionModel.from_pretrained(
                self.model_name,
                use_safetensors=False  # avoid redownloading large safetensor checkpoint
            )
            model = model.to(self.device)
            model.eval()
            return processor, model

        self.processor, self.model = await loop.run_in_executor(None, _load)
        self.logger.info("CLIP Vision initialized successfully")

    async def encode_image(self, frame: ImageFrame) -> VisionEmbedding:
        """Encode image to embedding"""
        self.logger.debug(f"Encoding image frame {frame.frame_id}")
        # Convert numpy to PIL
        image = Image.fromarray(frame.data)

        # Process in thread pool
        loop = asyncio.get_event_loop()

        def _encode():
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.pooler_output.cpu().numpy()[0]

            return embedding

        embedding = await loop.run_in_executor(None, _encode)

        return VisionEmbedding(
            embedding=embedding,
            timestamp=frame.timestamp,
            image_id=frame.frame_id
        )

    async def shutdown(self):
        """Cleanup"""
        self.model = None
        self.processor = None
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
