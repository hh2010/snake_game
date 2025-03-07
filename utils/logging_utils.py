import logging
import os
from datetime import datetime


def setup_logger(name: str, debug_mode: bool = False):
    """
    Configure a logger with consistent settings

    Args:
        name: The name of the logger
        debug_mode: Whether to enable debug logging (default: False)

    Returns:
        A configured logger instance
    """
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/{name}_{timestamp}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Always collect all logs

    # Remove any existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # Add file handler for all logs (DEBUG and above)
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    file_handler.setLevel(logging.DEBUG)  # Save all logs to file
    logger.addHandler(file_handler)

    # Add console handler (INFO by default, DEBUG if debug_mode)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    console_handler.setLevel(logging.DEBUG if debug_mode else logging.INFO)
    logger.addHandler(console_handler)

    logger.info(f"Initialized logger - Log file: {log_filename}")
    return logger, log_filename
