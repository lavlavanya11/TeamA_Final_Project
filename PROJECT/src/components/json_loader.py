import json
import os
from src.utils.logger import setup_logger

logger = setup_logger("JSON_LOADER")

class IntentJSONLoader:
    """
    Loads and validates the intents.json file.
    Checks that all required keys exist before passing data forward.
    """

    def __init__(self, intents_path: str):
        self.intents_path = intents_path

    def load(self) -> dict:
        """
        Reads intents.json and returns it as a dictionary.
        """
        if not os.path.exists(self.intents_path):
            logger.error(f"Intents file not found at: {self.intents_path}")
            raise FileNotFoundError(f"Intents file not found at: {self.intents_path}")

        with open(self.intents_path, "r") as f:
            data = json.load(f)

        self._validate(data)
        logger.info(f"Intents loaded successfully from {self.intents_path}")
        return data

    def _validate(self, data: dict):
        """
        Validates that required keys exist in the JSON.
        """
        if "intents" not in data:
            raise ValueError("Missing 'intents' key in intents.json")

        if "entities" not in data:
            raise ValueError("Missing 'entities' key in intents.json")

        for intent in data["intents"]:
            if "name" not in intent:
                raise ValueError(f"Intent missing 'name' key: {intent}")
            if "examples" not in intent:
                raise ValueError(f"Intent '{intent['name']}' missing 'examples' key")
            if "entities" not in intent:
                raise ValueError(f"Intent '{intent['name']}' missing 'entities' key")

        logger.info(f"Validation passed — {len(data['intents'])} intents found")