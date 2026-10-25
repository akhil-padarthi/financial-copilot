"""Chat with Financial Copilot in the terminal.

    python -m fincopilot.cli --backend demo                                   # no model needed
    python -m fincopilot.cli --backend ollama --model llama3.1:8b --prompt-style full   # untuned base model
    python -m fincopilot.cli --backend ollama --model fincopilot              # your fine-tuned model
    python -m fincopilot.cli --backend ollama --model fincopilot --ask "What's 10k at 7% for 10 years?"

Commands while chatting: /work (show or hide the steps), /reset, /quit
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .agent import Agent, AgentReply
from .backends import BACKEND_CHOICES, describe_backend, make_backend

STAGES = {"route": "Route", "extract": "Read values", "validate": "Check the form", "calculate": "Compute",
          "explain": "Explain", "clarify": "Ask instead of guess", "ground": "Check the numbers"}
MARKS = {"ok": "✓", "info": "✓", "fixed": "↺", "failed": "✗"}


_ANSI_OK = sys.stdout.isatty() and (os.name != "nt" or "WT_SESSION" in os.environ or "TERM" in os.environ)


def _color(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _ANSI_OK else text


def _safe_console() -> None:
    """On Windows, output redirected to a file uses a legacy encoding that can't print
    the step marks; replace unprintable characters instead of crashing."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def print_reply(reply: AgentReply, show_work: bool) -> None:
    if show_work and any(s.stage != "route" for s in reply.steps):
        for i, step in enumerate(reply.steps, 1):
            mark = MARKS.get(step.status, " ")
            mark = _color("31", mark) if step.status == "failed" else _color("32", mark)
            print(_color("2", f"  {i}. {STAGES.get(step.stage, step.stage):<16}") + f"{mark} {step.detail}")
            data = step.data.get("parameters") if step.stage == "extract" else step.data.get("result") if step.stage == "calculate" else None
            if data:
                print(_color("2", "       " + json.dumps(data)))
        print()
    print(_color("1", reply.text))
    print()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=BACKEND_CHOICES, default="demo")
    p.add_argument("--model", default=None)
    p.add_argument("--adapter", default=None)
    p.add_argument("--host", default=None)
    p.add_argument("--prompt-style", choices=["compact", "full"], default="compact")
    p.add_argument("--explain-with", choices=["model", "template"], default=None)
    p.add_argument("--ask", default=None, help="ask one question and exit")
    p.add_argument("--hide-work", action="store_true")
    args = p.parse_args(argv)
    _safe_console()

    explain_with = args.explain_with or ("template" if args.backend == "demo" else "model")
    agent = Agent(make_backend(args.backend, args.model, args.adapter, args.host),
                  prompt_style=args.prompt_style, explain_with=explain_with)
    show_work = not args.hide_work

    if args.ask:
        print_reply(agent.chat([], args.ask), show_work)
        return

    print(f"Financial Copilot ({describe_backend(args.backend, args.model, args.adapter)})")
    print("Ask a money question. Commands: /work, /reset, /quit\n")
    history = []
    while True:
        try:
            question = input(_color("36", "you › ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question:
            continue
        if question in ("/quit", "/exit"):
            return
        if question == "/reset":
            history = []
            print("Started a new conversation.\n")
            continue
        if question == "/work":
            show_work = not show_work
            print(f"Steps are now {'shown' if show_work else 'hidden'}.\n")
            continue
        try:
            reply = agent.chat(history, question)
        except RuntimeError as exc:
            print(_color("31", str(exc)) + "\n")
            continue
        history = reply.history
        print_reply(reply, show_work)


if __name__ == "__main__":
    main()
