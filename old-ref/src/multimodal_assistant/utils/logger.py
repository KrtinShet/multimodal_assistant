import logging
import sys

def setup_logger(name: str = "multimodal_assistant", log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)

    # Convert string log level to logging constant
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    # Prevent messages from being handled twice by parent/root handlers
    logger.propagate = False

    # Only add handler if logger doesn't have one already
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger
