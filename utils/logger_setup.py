import sys
from loguru import logger


def setup_logging():
    logger.remove()  # Remove the default handler
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
               "<level>{message}</level>",
        level="DEBUG"
    )

# Ensure the logger is set up when this module is imported
setup_logging()

