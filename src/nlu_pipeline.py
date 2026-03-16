from typing import Any, Dict

from .intent_classifier import classify_and_extract
from .entity_extractor import extract_entities


def run_pipeline(user_text: str) -> Dict[str, Any]:
    """
    High-level NLU pipeline entrypoint.

    Returns a structured dict:
    {
      "intent": str,
      "confidence": float,
      "entities": { ... },
      "raw": original_result,
      "error": str | None
    }
    """
    if not isinstance(user_text, str):
        return {
            "intent": "unknown",
            "confidence": 0.0,
            "entities": {},
            "raw": {},
            "error": "Input must be a string.",
        }

    text = user_text.strip()
    if not text:
        return {
            "intent": "unknown",
            "confidence": 0.0,
            "entities": {},
            "raw": {},
            "error": "Input text is empty.",
        }

    if len(text) > 2000:
        return {
            "intent": "unknown",
            "confidence": 0.0,
            "entities": {},
            "raw": {},
            "error": "Input text is too long (max 2000 characters).",
        }

    result = classify_and_extract(text)
    entities = extract_entities(result)

    return {
        "intent": result.get("intent"),
        "confidence": float(result.get("confidence", 0.0)),
        "entities": entities,
        "raw": result,
        "error": result.get("error"),
    }

