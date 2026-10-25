from __future__ import annotations

from .base import Backend, ScriptedBackend


def make_backend(kind: str, model: str = None, adapter: str = None, host: str = None) -> Backend:
    """Build a backend from command-line style options.

    hf      model = HF model id (default: Unsloth's 4-bit Llama 3.1 8B Instruct), adapter = LoRA folder
    ollama  model = Ollama model name (e.g. "fincopilot" or "llama3.1:8b"), host = Ollama URL
    openai  model = served model name, host = base URL of an OpenAI-compatible server
    """
    if kind == "hf":
        from .hf import HFBackend
        return HFBackend(model or "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", adapter=adapter)
    if kind == "ollama":
        from .ollama import OllamaBackend
        return OllamaBackend(model or "fincopilot", host or "http://localhost:11434")
    if kind == "openai":
        from .openai_compat import OpenAICompatBackend
        return OpenAICompatBackend(host or "http://localhost:8080/v1", model or "fincopilot")
    if kind == "demo":
        from .demo import DemoBackend
        return DemoBackend()
    raise ValueError(f"unknown backend {kind!r} (choose hf, ollama, openai or demo)")


BACKEND_CHOICES = ["ollama", "hf", "openai", "demo"]


def describe_backend(kind: str, model: str = None, adapter: str = None) -> str:
    if kind == "demo":
        return "Demo mode: recorded model replies, real calculations"
    if kind == "ollama":
        return f"{model or 'fincopilot'} via Ollama"
    if kind == "hf":
        return f"{adapter or model or 'Llama 3.1 8B'} via Hugging Face"
    return f"{model or 'fincopilot'} via OpenAI-compatible server"


__all__ = ["Backend", "ScriptedBackend", "make_backend", "describe_backend", "BACKEND_CHOICES"]
