"""Any server exposing an OpenAI-style /v1/completions endpoint
(vLLM, llama.cpp's llama-server, LM Studio...). We use the raw *completions*
endpoint, not chat, so the prompt format is exactly the one used in training.

These servers add <|begin_of_text|> themselves (vLLM's `add_special_tokens`
defaults to true), so we strip ours to avoid a double BOS token, which would
make the prompt differ from what the model was trained on.
"""
from __future__ import annotations

import os
import threading
from typing import Optional, Sequence

from ..prompts import BOS, STOP_STRINGS, format_llama31
from .base import Backend, Message


class OpenAICompatBackend(Backend):
    name = "openai"

    def __init__(self, base_url: str = "http://localhost:8080/v1", model: str = "fincopilot",
                 api_key: Optional[str] = None, timeout: Optional[float] = None):
        self.base_url, self.model = base_url.rstrip("/"), model
        self.api_key = api_key or os.environ.get("FINCOPILOT_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.timeout = float(timeout or os.environ.get("FINCOPILOT_TIMEOUT", 300))

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def build_payload(self, messages: Sequence[Message], max_new_tokens: int) -> dict:
        prompt = format_llama31(messages)
        if prompt.startswith(BOS):
            prompt = prompt[len(BOS):]
        return {"model": self.model, "prompt": prompt, "max_tokens": max_new_tokens,
                "temperature": 0, "stop": STOP_STRINGS}

    def generate(self, messages: Sequence[Message], max_new_tokens: int = 384) -> str:
        import requests

        try:
            r = requests.post(f"{self.base_url}/completions", json=self.build_payload(messages, max_new_tokens),
                              headers=self._headers(), timeout=self.timeout)
        except requests.ConnectionError as exc:
            raise RuntimeError(f"Can't reach the model server at {self.base_url}.") from exc
        except requests.Timeout as exc:
            raise RuntimeError("The model server took too long to answer. If it was asleep it may still be "
                               "starting up; try again in a minute.") from exc
        if r.status_code == 401:
            raise RuntimeError("The model server rejected the API key. Check FINCOPILOT_API_KEY.")
        if r.status_code == 404:
            raise RuntimeError(f"The model server has no model named '{self.model}'.")
        r.raise_for_status()
        return r.json()["choices"][0]["text"]

    def warm_up(self) -> None:
        """Wake a scale-to-zero server in the background so the first question waits less."""
        import requests

        def ping():
            try:
                requests.get(f"{self.base_url}/models", headers=self._headers(), timeout=self.timeout)
            except Exception:
                pass

        threading.Thread(target=ping, daemon=True).start()
