import json
import os


def load_intents(path="data/raw_data/intents.json"):
    """
    Load intents from JSON file
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Intent file not found at {path}")

    try:
        with open(path, "r") as file:
            data = json.load(file)

        return data.get("intents", [])

    except Exception as e:
        print("Error loading intents:", e)
        return []