from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class AppConfig:
    llm_model_name: str
    llm_rpm: int
    llm_max_attempts: int
    intents_path: str
    test_data_path: str


def _validate(cfg: Dict[str, Any]) -> None:
    if not isinstance(cfg, dict):
        raise ValueError("config.yaml must be a YAML mapping (dict).")

    llm = cfg.get("llm")
    if not isinstance(llm, dict):
        raise ValueError("Missing or invalid 'llm' section in config.yaml")
    if not llm.get("model_name"):
        raise ValueError("Missing 'llm.model_name' in config.yaml")

    paths = cfg.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("Missing or invalid 'paths' section in config.yaml")
    if not paths.get("intents_path"):
        raise ValueError("Missing 'paths.intents_path' in config.yaml")
    if not paths.get("test_data_path"):
        raise ValueError("Missing 'paths.test_data_path' in config.yaml")


@lru_cache(maxsize=1)
def load_config() -> AppConfig:
    """
    Load app configuration from `config/config.yaml`.
    `.env` should only be used for secrets like GROQ_API_KEY.
    """
    base_dir = Path(__file__).resolve().parent.parent
    cfg_path = base_dir / "config" / "config.yaml"
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    _validate(raw)

    llm = raw["llm"]
    paths = raw["paths"]

    return AppConfig(
        llm_model_name=str(llm.get("model_name")),
        llm_rpm=int(llm.get("rpm", 30)),
        llm_max_attempts=int(llm.get("max_attempts", 3)),
        intents_path=str(paths.get("intents_path")),
        test_data_path=str(paths.get("test_data_path")),
    )

