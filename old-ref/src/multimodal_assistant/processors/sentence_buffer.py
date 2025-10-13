from typing import AsyncIterator

class SentenceBuffer:
    """Buffers text tokens into complete sentences"""

    def __init__(self):
        self.buffer = []
        self.sentence_terminators = {'.', '!', '?', '\n'}

    async def buffer_stream(
        self,
        text_stream: AsyncIterator[str]
    ) -> AsyncIterator[str]:
        """Buffer tokens into sentences"""
        sentence_buffer = []

        async for token in text_stream:
            sentence_buffer.append(token)

            # Check if we have a sentence terminator
            if any(term in token for term in self.sentence_terminators):
                sentence = ''.join(sentence_buffer).strip()
                if sentence:
                    yield sentence
                sentence_buffer = []

        # Yield any remaining text
        if sentence_buffer:
            sentence = ''.join(sentence_buffer).strip()
            if sentence:
                yield sentence
