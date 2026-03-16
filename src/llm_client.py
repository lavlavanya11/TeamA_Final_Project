import os
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from utils.config_loader import load_config

load_dotenv()


@dataclass(frozen=True)
class LlmConfig:
    api_key: Optional[str]
    model_name: Optional[str]
    max_requests_per_minute: int = 30
    max_attempts: int = 3


class GroqClient:
    """
    Groq client with token-bucket rate limiting, safe under concurrency.
    Intended to be used as a singleton via @st.cache_resource.
    """

    def __init__(self, config: LlmConfig):
        self.config = config
        self._client = None
        self._lock = threading.Lock()
        self._tokens = float(config.max_requests_per_minute)
        self._last_refill = time.monotonic()

    def _ensure_client(self) -> Tuple[bool, Optional[str]]:
        if not self.config.api_key:
            return False, "GROQ_API_KEY is not set in .env"
        if not self.config.model_name:
            return False, "Model name is missing in config/config.yaml"
        if self._client is None:
            self._client = Groq(api_key=self.config.api_key)
        return True, None

    def _acquire_token(self) -> None:
        """Block until a request token is available (token bucket algorithm)."""
        interval = 60.0 / max(1, self.config.max_requests_per_minute)
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(
                float(self.config.max_requests_per_minute),
                self._tokens + elapsed / interval
            )
            self._last_refill = now

            if self._tokens < 1.0:
                wait = (1.0 - self._tokens) * interval
                time.sleep(wait)
                self._tokens = 0.0
            else:
                self._tokens -= 1.0

    def generate(self, prompt: str) -> Tuple[Optional[str], Optional[str]]:
        ok, err = self._ensure_client()
        if not ok:
            return None, err

        for attempt in range(self.config.max_attempts):
            try:
                self._acquire_token()
                response = self._client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a strict NLU engine. Always return valid JSON only. No explanation, no markdown, no backticks."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.1,
                    max_tokens=500,
                )
                raw = response.choices[0].message.content.strip()

                # Strip markdown fences if model includes them anyway
                if raw.startswith("```"):
                    raw = raw.strip("`").strip()
                    if raw.startswith("json"):
                        raw = raw[4:].strip()

                return raw, None

            except Exception as e:
                err_str = str(e)
                if "rate_limit" in err_str.lower() or "429" in err_str:
                    backoff = 5 * (attempt + 1)
                    time.sleep(backoff)
                else:
                    return None, f"LLM call failed: {e}"

        return None, "LLM call failed after retries (rate limit/backoff)."


@st.cache_resource
def _get_client() -> GroqClient:
    cfg = load_config()
    config = LlmConfig(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name=cfg.llm_model_name,
        max_requests_per_minute=cfg.llm_rpm,
        max_attempts=cfg.llm_max_attempts,
    )
    return GroqClient(config)


def call_llm(prompt: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (text, error_message). If successful, error_message is None.
    """
    return _get_client().generate(prompt)

