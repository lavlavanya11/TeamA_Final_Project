from functools import lru_cache
from typing import Any, Dict, Set

from utils.data_loader import load_intents_dataset

def extract_entities(nlu_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Post-process entities:
    - ensure dict type
    - drop unknown entity keys (not in dataset)
    - keep only entities required for the predicted intent (if specified)
    - normalize values (strip strings, drop empty)
    """
    entities = nlu_result.get("entities") or {}
    if not isinstance(entities, dict):
        return {}

    intent = (nlu_result.get("intent") or "").strip()

    allowed_entity_keys = _allowed_entity_keys()
    required_for_intent = _required_entities_for_intent(intent)

    out: Dict[str, Any] = {}
    for k, v in entities.items():
        if k not in allowed_entity_keys:
            continue
        if required_for_intent is not None and k not in required_for_intent:
            continue

        if isinstance(v, str):
            vv = v.strip()
            if not vv:
                continue
            out[k] = vv
        else:
            if v is None:
                continue
            out[k] = v

    return out


@lru_cache(maxsize=1)
def _allowed_entity_keys() -> Set[str]:
    dataset = load_intents_dataset()
    entities = dataset.get("entities", {}) or {}
    return set(entities.keys())


@lru_cache(maxsize=64)
def _required_entities_for_intent(intent: str):
    if not intent:
        return None
    dataset = load_intents_dataset()
    for i in dataset.get("intents", []) or []:
        if i.get("name") == intent:
            required = i.get("entities", [])
            return set(required) if isinstance(required, list) else None
    return None

