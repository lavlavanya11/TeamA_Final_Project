import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from .config_loader import load_config


BASE_DIR = Path(__file__).resolve().parent.parent


def _validate(data: Dict[str, Any]) -> None:
    if "intents" not in data:
        raise ValueError("Missing 'intents' key in intents.json")
    if not isinstance(data["intents"], list):
        raise ValueError("'intents' must be a list in intents.json")
    for intent in data["intents"]:
        if not isinstance(intent, dict):
            raise ValueError(f"Intent must be an object: {intent}")
        if "name" not in intent:
            raise ValueError(f"Intent missing 'name': {intent}")
        if "examples" not in intent:
            raise ValueError(f"Intent '{intent.get('name')}' missing 'examples'")


@lru_cache(maxsize=1)
def load_intents_dataset() -> Dict:
    """
    Shared loader for the main NLU dataset (intents + entities).
    """
    cfg = load_config()
    intents_path = BASE_DIR / cfg.intents_path
    data = json.loads(intents_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("intents.json must contain a JSON object at the top level")
    _validate(data)
    return data

