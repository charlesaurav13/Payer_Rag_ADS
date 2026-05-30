"""
LLM client — OpenRouter and Groq, both serving llama-3.1-8b-instruct.

Responses are cached in memory by (messages + model) hash so identical
queries never hit the API twice within the same Python session.
"""

import hashlib
import json
import logging
import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from groq import Groq
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"),override=True)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model names
# ---------------------------------------------------------------------------
_MODELS = {
    "openrouter": "meta-llama/llama-3.1-8b-instruct",
    "groq":       "llama-3.1-8b-instant",
}

# ---------------------------------------------------------------------------
# Module-level response cache  {hash -> response_text}
# ---------------------------------------------------------------------------
_cache: Dict[str, str] = {}


def _key(messages: List[Dict], model: str) -> str:
    payload = json.dumps({"m": messages, "model": model}, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class LLMClient:
    """
    Usage:
        llm = LLMClient("openrouter")          # or "groq"
        text = llm.complete([{"role":"user","content":"hi"}])
    """

    def __init__(self, provider: str = "openrouter"):
        provider = provider.lower()
        if provider not in _MODELS:
            raise ValueError(f"provider must be 'openrouter' or 'groq', got '{provider}'")

        self.provider = provider
        self.model    = _MODELS[provider]

        if provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY", "")
            if not api_key:
                raise EnvironmentError("OPENROUTER_API_KEY not set in .env")
            self._client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
            )
        else:
            api_key = os.getenv("GROQ_API_KEY", "")
            if not api_key:
                raise EnvironmentError("GROQ_API_KEY not set in .env")
            self._client = Groq(api_key=api_key)

        log.info("LLMClient ready — provider=%s  model=%s", provider, self.model)

    # ------------------------------------------------------------------
    def complete(
        self,
        messages: List[Dict],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        use_cache: bool = True,
    ) -> str:
        """
        Send messages to the LLM and return the response text.
        Identical calls return the cached response without hitting the API.
        """
        cache_key = _key(messages, self.model)

        if use_cache and cache_key in _cache:
            log.debug("Cache hit [%s]", cache_key[:8])
            return _cache[cache_key]

        log.debug("Cache miss [%s] — calling API", cache_key[:8])
        response = self._call(messages, temperature, max_tokens)

        if use_cache:
            _cache[cache_key] = response

        return response

    # ------------------------------------------------------------------
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _call(self, messages: List[Dict], temperature: float, max_tokens: int) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()

    # ------------------------------------------------------------------
    @property
    def cache_size(self) -> int:
        return len(_cache)

    def clear_cache(self) -> None:
        _cache.clear()
        log.info("LLM cache cleared")
