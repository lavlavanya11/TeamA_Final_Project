from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

from utils.data_loader import load_intents_dataset


BASE_DIR = Path(__file__).resolve().parent.parent
PROMPTS_DIR = BASE_DIR / "prompts"


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def _intent_and_entity_context(dataset: Dict) -> str:
    intents: List[Dict] = dataset.get("intents", [])
    entities: Dict[str, List[str]] = dataset.get("entities", {})

    intent_lines: List[str] = []
    for i in intents:
        name = i.get("name", "").strip()
        if not name:
            continue
        intent_lines.append(f"- {name}")

    entity_lines: List[str] = []
    for ent_name, values in entities.items():
        if not ent_name:
            continue
        examples = ", ".join(values[:8]) if isinstance(values, list) else ""
        if examples:
            entity_lines.append(f"- {ent_name}: examples: {examples}")
        else:
            entity_lines.append(f"- {ent_name}")

    return (
        "Intents (choose exactly one):\n"
        + ("\n".join(intent_lines) if intent_lines else "- (none)\n")
        + "\n\n"
        + "Entities:\n"
        + ("\n".join(entity_lines) if entity_lines else "- (none)\n")
    )


@lru_cache(maxsize=1)
def _load_prompt_templates() -> Tuple[str, str]:
    base_prompt = _load_text(PROMPTS_DIR / "base_prompt.txt")
    few_shots = _load_text(PROMPTS_DIR / "few_shot_examples.txt")
    return base_prompt, few_shots


def build_nlu_prompt(user_text: str) -> str:
    """
    Build the full prompt for NLU:
    - base instructions
    - few-shot examples
    - current user query
    """
    base_prompt, few_shots = _load_prompt_templates()
    dataset = load_intents_dataset()
    context = _intent_and_entity_context(dataset)

    return (
        f"{base_prompt}\n\n"
        f"{context}\n\n"
        f"{few_shots}\n\n"
        f'User: "{user_text}"\n'
        f"Output:"
    )

