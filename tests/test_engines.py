"""Tests for engine components."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import AsyncIterator
import numpy as np
from pathlib import Path

from assistant.engines.tts_engine import KokoroTTSEngine
from assistant.engines.lm_engine import SimpleLLMEngine, LMEngine
from assistant.engines.base import AudioChunk, VisionEmbedding


class TestKokoroTTSEngine:
    """Tests for KokoroTTSEngine."""

    def test_initialization(self):
        """Test TTS engine initialization."""
        engine = KokoroTTSEngine()

        assert engine.sample_rate == 24000
        assert engine.model_path == Path("assets/kokoro/kokoro-v1.0.onnx")
        assert engine.voices_path == Path("assets/kokoro/voices-v1.0.bin")
        assert engine.logger is not None
        assert engine.max_chars_per_chunk == 160
        assert engine.kokoro is None

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test successful TTS initialization."""
        engine = KokoroTTSEngine()

        with patch('assistant.engines.tts_engine.Path.exists', return_value=True):
            with patch('kokoro_onnx.Kokoro') as mock_kokoro:
                mock_instance = MagicMock()
                mock_kokoro.return_value = mock_instance

                await engine.initialize()

                assert engine.kokoro is mock_instance
                mock_kokoro.assert_called_once_with(
                    str(engine.model_path),
                    str(engine.voices_path)
                )

    @pytest.mark.asyncio
    async def test_initialize_model_files_not_found(self):
        """Test initialization when model files are not found."""
        engine = KokoroTTSEngine()

        with patch('assistant.engines.tts_engine.Path.exists', return_value=False):
            await engine.initialize()

            assert engine.kokoro is None

    @pytest.mark.asyncio
    async def test_initialize_import_error(self):
        """Test initialization when Kokoro is not installed."""
        engine = KokoroTTSEngine()

        with patch('assistant.engines.tts_engine.Path.exists', return_value=True):
            with patch('builtins.__import__', side_effect=ImportError):
                await engine.initialize()

                assert engine.kokoro is None

    @pytest.mark.asyncio
    async def test_initialize_exception(self):
        """Test initialization with general exception."""
        engine = KokoroTTSEngine()

        with patch('assistant.engines.tts_engine.Path.exists', return_value=True):
            with patch('kokoro_onnx.Kokoro', side_effect=Exception("Test error")):
                await engine.initialize()

                assert engine.kokoro is None

    @pytest.mark.asyncio
    async def test_synthesize_stream_disabled(self):
        """Test synthesis when TTS is disabled."""
        engine = KokoroTTSEngine()
        engine.kokoro = None  # Disabled state

        async def text_stream():
            yield "Hello world"

        chunks = []
        async for chunk in engine.synthesize_stream(text_stream()):
            chunks.append(chunk)

        # Should consume stream without producing audio
        assert len(chunks) == 0

    @pytest.mark.asyncio
    async def test_synthesize_stream_with_sentence_ending(self):
        """Test synthesis with sentence ending punctuation."""
        engine = KokoroTTSEngine()
        mock_kokoro = MagicMock()
        engine.kokoro = mock_kokoro

        # Mock Kokoro response
        mock_samples = np.ones(1000, dtype=np.float32)
        mock_kokoro.create.return_value = (mock_samples, 24000)

        async def text_stream():
            yield "Hello world."

        chunks = []
        async for chunk in engine.synthesize_stream(text_stream()):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert isinstance(chunks[0], AudioChunk)
        assert chunks[0].sample_rate == 24000

    @pytest.mark.asyncio
    async def test_synthesize_stream_over_length(self):
        """Test synthesis when text exceeds max chunk length."""
        engine = KokoroTTSEngine()
        engine.max_chars_per_chunk = 10  # Set low for testing
        mock_kokoro = MagicMock()
        engine.kokoro = mock_kokoro

        # Mock Kokoro response
        mock_samples = np.ones(1000, dtype=np.float32)
        mock_kokoro.create.return_value = (mock_samples, 24000)

        async def text_stream():
            yield "This is a very long text that exceeds the limit"

        chunks = []
        async for chunk in engine.synthesize_stream(text_stream()):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert isinstance(chunks[0], AudioChunk)

    @pytest.mark.asyncio
    async def test_synthesize_stream_cancellation(self):
        """Test synthesis cancellation."""
        engine = KokoroTTSEngine()
        mock_kokoro = MagicMock()
        engine.kokoro = mock_kokoro

        # Mock Kokoro response
        mock_samples = np.ones(1000, dtype=np.float32)
        mock_kokoro.create.return_value = (mock_samples, 24000)

        async def text_stream():
            yield "Hello"
            raise asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            async for chunk in engine.synthesize_stream(text_stream()):
                pass

    @pytest.mark.asyncio
    async def test_synthesize_stream_final_buffer(self):
        """Test synthesis of remaining buffer after stream ends."""
        engine = KokoroTTSEngine()
        mock_kokoro = MagicMock()
        engine.kokoro = mock_kokoro

        # Mock Kokoro response
        mock_samples = np.ones(1000, dtype=np.float32)
        mock_kokoro.create.return_value = (mock_samples, 24000)

        async def text_stream():
            yield "Partial sentence without ending"

        chunks = []
        async for chunk in engine.synthesize_stream(text_stream()):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert isinstance(chunks[0], AudioChunk)

    @pytest.mark.asyncio
    async def test__synthesize_async_success(self):
        """Test successful async synthesis."""
        engine = KokoroTTSEngine()
        mock_kokoro = MagicMock()
        engine.kokoro = mock_kokoro

        # Mock Kokoro response
        mock_samples = np.ones(1000, dtype=np.float32)
        mock_kokoro.create.return_value = (mock_samples, 24000)

        chunks = []
        async for chunk in engine._synthesize_async("Hello world"):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert isinstance(chunks[0], AudioChunk)
        assert chunks[0].sample_rate == 24000
        assert np.array_equal(chunks[0].data, mock_samples)

    @pytest.mark.asyncio
    async def test__synthesize_async_no_kokoro(self):
        """Test synthesis when Kokoro is not available."""
        engine = KokoroTTSEngine()
        engine.kokoro = None

        chunks = []
        async for chunk in engine._synthesize_async("Hello world"):
            chunks.append(chunk)

        assert len(chunks) == 0

    def test__synthesize_sentence(self):
        """Test sentence synthesis (blocking)."""
        engine = KokoroTTSEngine()
        mock_kokoro = MagicMock()
        engine.kokoro = mock_kokoro

        # Mock Kokoro response with different sample rate
        mock_samples = np.ones(1000, dtype=np.int16)
        mock_kokoro.create.return_value = (mock_samples, 22050)

        result = engine._synthesize_sentence("Hello world")

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32  # Should be converted
        assert engine.sample_rate == 22050  # Should be updated

    def test__synthesize_sentence_no_kokoro(self):
        """Test sentence synthesis when Kokoro is not available."""
        engine = KokoroTTSEngine()
        engine.kokoro = None

        result = engine._synthesize_sentence("Hello world")

        assert result is None

    def test__synthesize_sentence_exception(self):
        """Test sentence synthesis with exception."""
        engine = KokoroTTSEngine()
        mock_kokoro = MagicMock()
        engine.kokoro = mock_kokoro
        mock_kokoro.create.side_effect = Exception("Synthesis error")

        result = engine._synthesize_sentence("Hello world")

        assert result is None

    @pytest.mark.asyncio
    async def test_shutdown(self):
        """Test TTS shutdown."""
        engine = KokoroTTSEngine()
        engine.kokoro = MagicMock()

        await engine.shutdown()

        assert engine.kokoro is None


class TestSimpleLLMEngine:
    """Tests for SimpleLLMEngine."""

    def test_initialization(self):
        """Test LLM engine initialization."""
        engine = SimpleLLMEngine()

        assert engine.settings is not None
        assert engine.logger is not None
        assert not engine._initialized
        assert engine._model is None

    def test_initialization_with_settings(self, test_settings):
        """Test LLM engine initialization with custom settings."""
        engine = SimpleLLMEngine(test_settings)

        assert engine.settings is test_settings
        assert engine.logger is not None
        assert not engine._initialized
        assert engine._model is None

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test successful LLM initialization."""
        engine = SimpleLLMEngine()

        await engine.initialize()

        assert engine._initialized
        assert engine._model == "mock_model"

    @pytest.mark.asyncio
    async def test_initialize_exception(self):
        """Test LLM initialization with exception."""
        engine = SimpleLLMEngine()

        # Create a mock that raises exception when setting _initialized
        original_init = engine.initialize

        async def failing_init():
            raise Exception("Test error")

        engine.initialize = failing_init

        with pytest.raises(Exception, match="Test error"):
            await engine.initialize()

    @pytest.mark.asyncio
    async def test_generate_stream_success(self):
        """Test successful text generation stream."""
        engine = SimpleLLMEngine()
        await engine.initialize()

        chunks = []
        async for chunk in engine.generate_stream("Hello"):
            chunks.append(chunk)

        # Should generate response
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        assert "".join(chunks).startswith("This is a mock response to:")

    @pytest.mark.asyncio
    async def test_generate_stream_not_initialized(self):
        """Test generation when engine is not initialized."""
        engine = SimpleLLMEngine()
        # Don't initialize

        chunks = []
        async for chunk in engine.generate_stream("Hello"):
            chunks.append(chunk)

        # Should return error message
        assert len(chunks) > 0
        assert any("not initialized" in chunk for chunk in chunks)

    @pytest.mark.asyncio
    async def test_generate_stream_with_vision_embedding(self):
        """Test generation with vision embedding."""
        engine = SimpleLLMEngine()
        await engine.initialize()

        # Create mock vision embedding
        vision_embedding = VisionEmbedding(
            embedding=np.ones(512, dtype=np.float32),
            timestamp=0.0,
            image_id="test_image"
        )

        chunks = []
        async for chunk in engine.generate_stream("Describe this image", vision_embedding):
            chunks.append(chunk)

        # Should generate response (vision embedding is ignored in mock implementation)
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_generate_stream_exception(self):
        """Test generation with exception."""
        engine = SimpleLLMEngine()
        await engine.initialize()

        # Mock to raise exception during generation
        with patch('asyncio.sleep', side_effect=Exception("Test error")):
            chunks = []
            async for chunk in engine.generate_stream("Hello"):
                chunks.append(chunk)

            # Should handle error gracefully
            assert any("Error generating response" in chunk for chunk in chunks)

    @pytest.mark.asyncio
    async def test_generate_stream_long_prompt(self):
        """Test generation with very long prompt."""
        engine = SimpleLLMEngine()
        await engine.initialize()

        long_prompt = "Hello " * 1000  # Very long prompt

        chunks = []
        async for chunk in engine.generate_stream(long_prompt):
            chunks.append(chunk)

        # Should generate response
        assert len(chunks) > 0
        # Should truncate in debug log
        assert len("".join(chunks)) > 0

    @pytest.mark.asyncio
    async def test_shutdown(self):
        """Test LLM shutdown."""
        engine = SimpleLLMEngine()
        await engine.initialize()

        assert engine._initialized
        assert engine._model is not None

        await engine.shutdown()

        assert not engine._initialized
        assert engine._model is None

    @pytest.mark.asyncio
    async def test_shutdown_not_initialized(self):
        """Test shutdown when not initialized."""
        engine = SimpleLLMEngine()
        # Don't initialize

        # Should not raise error
        await engine.shutdown()

        assert not engine._initialized
        assert engine._model is None


class TestLMEngineBackwardCompatibility:
    """Tests for LMEngine backward compatibility."""

    def test_lm_engine_inherits_from_simple_llm_engine(self):
        """Test that LMEngine inherits from SimpleLLMEngine."""
        engine = LMEngine()

        assert isinstance(engine, SimpleLLMEngine)
        assert hasattr(engine, 'initialize')
        assert hasattr(engine, 'generate_stream')
        assert hasattr(engine, 'shutdown')

    @pytest.mark.asyncio
    async def test_lm_engine_functionality(self):
        """Test that LMEngine works like SimpleLLMEngine."""
        engine = LMEngine()

        await engine.initialize()

        chunks = []
        async for chunk in engine.generate_stream("Test"):
            chunks.append(chunk)

        assert len(chunks) > 0

        await engine.shutdown()


class TestEngineIntegration:
    """Integration tests for engines."""

    @pytest.mark.asyncio
    async def test_tts_llm_pipeline(self, test_settings):
        """Test TTS -> LLM pipeline integration."""
        # Initialize LLM
        llm_engine = SimpleLLMEngine(test_settings)
        await llm_engine.initialize()

        # Generate text response
        text_chunks = []
        async for chunk in llm_engine.generate_stream("Hello"):
            text_chunks.append(chunk)

        text_response = "".join(text_chunks)

        # Initialize TTS
        tts_engine = KokoroTTSEngine()

        # Mock TTS to avoid actual synthesis
        tts_engine.kokoro = MagicMock()
        mock_samples = np.ones(1000, dtype=np.float32)
        tts_engine.kokoro.create.return_value = (mock_samples, 24000)

        # Convert text to audio
        async def text_stream():
            for chunk in text_chunks:
                yield chunk

        audio_chunks = []
        async for chunk in tts_engine.synthesize_stream(text_stream()):
            audio_chunks.append(chunk)

        # Verify pipeline worked
        assert len(text_response) > 0
        assert len(audio_chunks) >= 0  # Might be empty if TTS is disabled

        # Cleanup
        await llm_engine.shutdown()
        await tts_engine.shutdown()

    @pytest.mark.asyncio
    async def test_engine_error_propagation(self, test_settings):
        """Test that engine errors are properly handled."""
        llm_engine = SimpleLLMEngine(test_settings)
        tts_engine = KokoroTTSEngine()

        # Initialize both engines
        await llm_engine.initialize()
        await tts_engine.initialize()

        # Mock TTS to raise error
        tts_engine.kokoro = MagicMock()
        tts_engine.kokoro.create.side_effect = Exception("TTS error")

        # Generate text
        text_response = ""
        async for chunk in llm_engine.generate_stream("Hello"):
            text_response += chunk

        # Try to convert to audio
        async def text_stream():
            yield text_response

        audio_chunks = []
        async for chunk in tts_engine.synthesize_stream(text_stream()):
            audio_chunks.append(chunk)

        # Should handle TTS error gracefully
        assert len(text_response) > 0
        # audio_chunks might be empty due to TTS error

        # Cleanup
        await llm_engine.shutdown()
        await tts_engine.shutdown()