from typing import AsyncIterator

from assistant.config.settings import Settings
from assistant.utils.logger import setup_logger


class TextOutputHandler:
    """Handles text output display with streaming support.

    This handler displays streaming text tokens to the console with
    basic formatting and cleanup (e.g., removing excessive newlines).
    """

    def __init__(self, settings: Settings):
        """Initialize text output handler.

        Args:
            settings: Application settings
        """
        self._buffer = []
        self.logger = setup_logger("assistant.output.text", settings.log_level)

    async def display_stream(self, text_stream: AsyncIterator[str]):
        """Display streaming text tokens to console.

        Args:
            text_stream: Async iterator yielding text tokens
        """
        self._buffer = []
        token_count = 0

        async for token in text_stream:
            # Clean up token: remove standalone backslashes and excessive newlines
            cleaned_token = token

            # Skip standalone backslashes
            if token == '\\':
                continue

            # Replace multiple consecutive newlines with max 2
            if '\n' in token:
                parts = token.split('\n')
                # Keep at most 2 consecutive newlines
                cleaned_parts = []
                for i, part in enumerate(parts):
                    if i > 0:
                        # Add newline, but limit consecutive ones
                        if not (i > 1 and parts[i-1] == '' and part == ''):
                            cleaned_parts.append('')
                    cleaned_parts.append(part)
                cleaned_token = '\n'.join(cleaned_parts)

            if cleaned_token:
                print(cleaned_token, end='', flush=True)
                self._buffer.append(cleaned_token)
                token_count += 1

        print()  # New line at end
        self.logger.debug(f"Displayed {token_count} tokens")

    def display_message(self, message: str):
        """Display a complete message to console.

        Args:
            message: Complete message to display
        """
        self.logger.debug(f"Displaying message: {message[:100]}...")
        print(message)

    def get_buffer(self) -> str:
        """Get the complete buffered text.

        Returns:
            Concatenated text from all buffered tokens
        """
        return ''.join(self._buffer)

    def clear_buffer(self):
        """Clear the internal buffer."""
        self._buffer = []
