import logging
import os


def setup_logger():

    # create logs folder if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    logging.basicConfig(
        filename="logs/trainer.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    return logging.getLogger(__name__)