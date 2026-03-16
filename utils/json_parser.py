import json
from typing import Any, Optional


def safe_json_parse(raw: str) -> Optional[Any]:
    """
    Try to robustly parse a JSON object from an LLM response.

    Strategy:
    1. Direct json.loads
    2. If that fails, try the substring between first '{' and last '}'.
    """
    if raw is None:
        return None

    candidate = raw.strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and start < end:
        try:
            return json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            return None

    return None

