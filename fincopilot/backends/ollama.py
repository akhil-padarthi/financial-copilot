"""Run the model locally with Ollama (https://ollama.com).

We send the exact Llama 3.1 prompt ourselves (raw mode) so local inference
matches training byte for byte, whatever chat template the Ollama model has.
Works with the fine-tuned GGUF and with the stock `llama3.1:8b` as a baseline.
"""
from __future__ import annotations

from typing import Sequence

from ..prompts import BOS, STOP_STRINGS, format_llama31
from .base import Backend, Message


class OllamaBackend(Backend):
    name = "ollama"

    def __init__(self, model: str = "fincopilot", host: str = "http://localhost:11434",
                 timeout: float = 180.0, num_ctx: int = 8192):
        self.model, self.host, self.timeout, self.num_ctx = model, host.rstrip("/"), timeout, num_ctx

    def generate(self, messages: Sequence[Message], max_new_tokens: int = 384) -> str:
        import requests

        # llama.cpp adds <|begin_of_text|> itself, so drop ours to avoid a double BOS.
        prompt = format_llama31(messages)[len(BOS):]
        payload = {"model": self.model, "prompt": prompt, "raw": True, "stream": False,
                   "options": {"temperature": 0, "num_predict": max_new_tokens, "num_ctx": self.num_ctx,
                               "stop": STOP_STRINGS}}
        try:
            r = requests.post(f"{self.host}/api/generate", json=payload, timeout=self.timeout)
        except requests.ConnectionError as exc:
            raise RuntimeError(f"Can't reach Ollama at {self.host}. Is `ollama serve` running?") from exc
        if r.status_code == 404:
            raise RuntimeError(f"Ollama has no model named '{self.model}'. Run `ollama list` to see what's installed.")
        r.raise_for_status()
        return r.json().get("response", "")
