from typing import Any, Dict, Optional

from utils.json_parser import safe_json_parse
from utils.logger import get_logger
from .llm_client import call_llm
from .prompt_builder import build_nlu_prompt


logger = get_logger(__name__)


FALLBACK_RESULT: Dict[str, Any] = {
    "intent": "unknown",
    "confidence": 0.0,
    "entities": {},
    "error": None,
}


def classify_and_extract(text: str) -> Dict[str, Any]:
    """
    Run intent classification + entity extraction via the LLM.
    """
    prompt = build_nlu_prompt(text)
    logger.info("Sending prompt to LLM.")
    raw, err = call_llm(prompt)

    if raw is None:
        logger.error("LLM returned no text, using fallback result.")
        out = FALLBACK_RESULT.copy()
        out["error"] = err or "LLM returned no text."
        return out

    parsed = safe_json_parse(raw)
    if parsed is None or not isinstance(parsed, dict):
        logger.error("Failed to parse JSON from LLM output, using fallback.")
        logger.debug("Raw LLM output: %s", raw)
        out = FALLBACK_RESULT.copy()
        out["error"] = "Failed to parse JSON from LLM output."
        return out

    # Basic normalization to ensure keys exist
    parsed.setdefault("intent", FALLBACK_RESULT["intent"])
    parsed.setdefault("confidence", FALLBACK_RESULT["confidence"])
    parsed.setdefault("entities", FALLBACK_RESULT["entities"])
    parsed.setdefault("error", None)

    logger.info(
        "Predicted intent=%s, confidence=%.3f",
        parsed.get("intent"),
        float(parsed.get("confidence", 0.0)),
    )
    return parsed

