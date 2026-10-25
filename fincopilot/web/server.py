"""Web app: chat with Financial Copilot and see each step of the calculation.

Local:
    python -m fincopilot.web --backend demo                          # try the interface, no model needed
    python -m fincopilot.web --backend ollama --model llama3.1:8b --prompt-style full   # base model
    python -m fincopilot.web --backend ollama --model fincopilot     # your fine-tuned model
Then open http://localhost:7860

Deployed (configured with environment variables, see DEPLOY.md):
    gunicorn -b 0.0.0.0:$PORT "fincopilot.web.server:create_app_from_env()"
"""
from __future__ import annotations

import argparse
import os
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Mapping, Optional, Tuple

from flask import Flask, jsonify, request, send_from_directory

from ..agent import Agent
from ..backends import BACKEND_CHOICES, describe_backend, make_backend

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
MAX_HISTORY = 24  # messages kept for context; older turns are dropped


class RateLimiter:
    """Sliding-window limiter: at most `max_requests` per `window` seconds per key."""

    def __init__(self, max_requests: int, window: float):
        self.max_requests, self.window = max_requests, window
        self._hits: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str, now: Optional[float] = None) -> Tuple[bool, int]:
        now = time.monotonic() if now is None else now
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and hits[0] <= now - self.window:
                hits.popleft()
            if len(hits) >= self.max_requests:
                return False, int(hits[0] + self.window - now) + 1
            hits.append(now)
            if len(self._hits) > 10_000:  # forget idle visitors
                for k in [k for k, v in self._hits.items() if not v]:
                    del self._hits[k]
            return True, 0


def parse_limit(spec: Optional[str]) -> Optional[RateLimiter]:
    """'30/600' -> 30 requests per 600 seconds. Empty or '0' disables the limit."""
    if not spec or spec.strip() in ("0", "off", "none"):
        return None
    count, _, seconds = spec.partition("/")
    return RateLimiter(int(count), float(seconds or 60))


def _client_key() -> str:
    forwarded = request.headers.get("X-Forwarded-For", "")
    return forwarded.split(",")[0].strip() or (request.remote_addr or "unknown")


def create_app(agent: Agent, description: str, per_visitor: Optional[RateLimiter] = None,
               daily: Optional[RateLimiter] = None) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.json.sort_keys = False  # keep each calculator's own output order (headline result first)

    @app.get("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @app.get("/healthz")
    def healthz():
        return jsonify({"ok": True})

    @app.get("/api/info")
    def info():
        agent.backend.warm_up()  # a visitor opened the page: start waking the model now
        return jsonify({"model": description, "explain_with": agent.explain_with, "prompt_style": agent.prompt_style})

    @app.post("/api/chat")
    def chat():
        body: Dict[str, Any] = request.get_json(silent=True) or {}
        message = str(body.get("message", "")).strip()
        if not message:
            return jsonify({"error": "Type a question first."}), 400
        if len(message) > 2000:
            return jsonify({"error": "That question is too long. Keep it under 2,000 characters."}), 400
        if per_visitor:
            ok, wait = per_visitor.check(_client_key())
            if not ok:
                return jsonify({"error": f"You've asked a lot of questions in a short time. Try again in {wait} seconds."}), 429
        if daily:
            ok, _ = daily.check("all")
            if not ok:
                return jsonify({"error": "This demo has reached its daily question limit. Please try again tomorrow."}), 429
        history: List[Dict[str, str]] = [
            {"role": m["role"], "content": str(m["content"])} for m in body.get("history", [])
            if isinstance(m, dict) and m.get("role") in ("user", "assistant", "tool") and "content" in m
        ][-MAX_HISTORY:]
        try:
            reply = agent.chat(history, message)
        except RuntimeError as exc:  # backend unreachable, model missing, ...
            return jsonify({"error": str(exc)}), 502
        return jsonify(reply.to_dict())

    return app


def create_app_from_env(env: Optional[Mapping[str, str]] = None) -> Flask:
    """Build the app from environment variables (for gunicorn, Docker and Modal).

    FINCOPILOT_BACKEND        demo | ollama | openai | hf            (default demo)
    FINCOPILOT_MODEL          model name on the server, e.g. fincopilot or base
    FINCOPILOT_HOST           server URL (Ollama URL, or OpenAI-compatible base URL ending in /v1)
    FINCOPILOT_ADAPTER        LoRA folder (hf backend only)
    FINCOPILOT_API_KEY        bearer token for the model server (openai backend)
    FINCOPILOT_PROMPT_STYLE   compact (fine-tuned model) | full (untuned base model)
    FINCOPILOT_EXPLAIN_WITH   model | template
    FINCOPILOT_LABEL          text shown in the header
    FINCOPILOT_RATE_LIMIT     per-visitor limit, e.g. 30/600 = 30 questions per 10 minutes (default off)
    FINCOPILOT_DAILY_LIMIT    total questions per day across all visitors, e.g. 500 (default off)
    """
    env = os.environ if env is None else env
    kind = env.get("FINCOPILOT_BACKEND", "demo")
    model, host, adapter = env.get("FINCOPILOT_MODEL"), env.get("FINCOPILOT_HOST"), env.get("FINCOPILOT_ADAPTER")
    backend = make_backend(kind, model=model, adapter=adapter, host=host)
    explain_with = env.get("FINCOPILOT_EXPLAIN_WITH") or ("template" if kind == "demo" else "model")
    agent = Agent(backend, prompt_style=env.get("FINCOPILOT_PROMPT_STYLE", "compact"), explain_with=explain_with)
    label = env.get("FINCOPILOT_LABEL") or describe_backend(kind, model, adapter)
    daily_spec = env.get("FINCOPILOT_DAILY_LIMIT")
    daily = parse_limit(f"{daily_spec}/86400") if daily_spec and daily_spec.strip() not in ("0", "off") else None
    return create_app(agent, label, per_visitor=parse_limit(env.get("FINCOPILOT_RATE_LIMIT")), daily=daily)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=BACKEND_CHOICES, default="demo")
    p.add_argument("--model", default=None)
    p.add_argument("--adapter", default=None)
    p.add_argument("--host", default=None, help="model server URL (Ollama or OpenAI-compatible)")
    p.add_argument("--prompt-style", choices=["compact", "full"], default="compact",
                   help="use 'full' for an untuned base model")
    p.add_argument("--explain-with", choices=["model", "template"], default=None,
                   help="who writes the explanation (default: model, or template in demo mode)")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    p.add_argument("--bind", default="127.0.0.1", help="use 0.0.0.0 to accept connections from other machines")
    args = p.parse_args(argv)

    explain_with = args.explain_with or ("template" if args.backend == "demo" else "model")
    backend = make_backend(args.backend, model=args.model, adapter=args.adapter, host=args.host)
    agent = Agent(backend, prompt_style=args.prompt_style, explain_with=explain_with)
    app = create_app(agent, describe_backend(args.backend, args.model, args.adapter))
    print(f"Financial Copilot is running at http://{args.bind}:{args.port}")
    app.run(host=args.bind, port=args.port, debug=False, threaded=args.backend != "hf")


if __name__ == "__main__":
    main()
