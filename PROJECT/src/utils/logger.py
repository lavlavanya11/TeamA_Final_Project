import logging
import os
from datetime import datetime

def setup_logger(name: str):
    """
    Creates a logger with timestamped logs.
    Logs are written to both console and log file.
    """
    # Create logs directory
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Log file name with date
    log_file = os.path.join(
        log_dir, f"bottrainer_{datetime.now().strftime('%Y-%m-%d')}.log"
    )

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate logs
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger