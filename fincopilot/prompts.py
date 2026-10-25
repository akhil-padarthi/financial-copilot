"""System prompt, Llama 3.1 chat formatting, and training-example tokenization.

The same `format_llama31` function is used for training and for every inference
backend, so the model always sees byte-identical prompt formatting.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

from .tools import all_tools

BOS = "<|begin_of_text|>"
START_HEADER = "<|start_header_id|>"
END_HEADER = "<|end_header_id|>"
EOT = "<|eot_id|>"
STOP_STRINGS = ["<|eot_id|>", "<|eom_id|>", "<|start_header_id|>", "<|end_of_text|>"]

Message = Dict[str, str]

RULES = """You are Financial Copilot. You answer money questions by calling exact calculation tools. You never do arithmetic yourself.

How to respond:
1. If the question needs a calculation, reply with only a JSON tool call: {"name": "<tool>", "parameters": {...}}
2. Copy the user's numbers exactly. Write percentages as percents (7% -> 7), 10k as 10000 and 1.2 million as 1200000. Never compute or convert values yourself.
3. If a required value is missing, ask the user for it. Never guess.
4. After a tool result arrives, explain it in plain English using only numbers from the question and the result.
5. For general questions about financial concepts, answer briefly in plain text. Do not predict markets or recommend specific investments."""

_FULL_EXAMPLE = """Example:
User: If I invest $10,000 at 7% for 10 years, how much will I have?
Assistant: {"name": "compound_interest", "parameters": {"principal": 10000, "annual_rate": 7, "years": 10}}"""


def build_system_prompt(style: str = "compact") -> str:
    """compact: one signature per tool (used for fine-tuning and the fine-tuned model).
    full: descriptions for every parameter plus an example (gives an untuned base model a fair chance)."""
    tools = all_tools()
    if style == "compact":
        header = "Tools (a|b = give one of them, [..] = optional):"
        catalog = "\n".join(t.signature() for t in tools)
        return f"{RULES}\n\n{header}\n{catalog}"
    if style == "full":
        catalog = "\n\n".join(t.doc() for t in tools)
        return f"{RULES}\n\nTools:\n\n{catalog}\n\n{_FULL_EXAMPLE}"
    raise ValueError(f"unknown prompt style {style!r}")


def _role(role: str) -> str:
    # Llama 3.1 calls tool results the "ipython" role.
    return "ipython" if role in ("tool", "ipython") else role


def format_llama31(messages: Sequence[Message], add_generation_prompt: bool = True) -> str:
    out = [BOS]
    for m in messages:
        out.append(f"{START_HEADER}{_role(m['role'])}{END_HEADER}\n\n{m['content'].strip()}{EOT}")
    if add_generation_prompt:
        out.append(f"{START_HEADER}assistant{END_HEADER}\n\n")
    return "".join(out)


def with_system(messages: Sequence[Message], style: str = "compact") -> List[Message]:
    msgs = [m for m in messages if m["role"] != "system"]
    return [{"role": "system", "content": build_system_prompt(style)}] + msgs


def build_training_features(messages: Sequence[Message], encode: Callable[[str], List[int]],
                            max_length: Optional[int] = None) -> Optional[Dict[str, List[int]]]:
    """Tokenize a conversation and mask the loss to assistant replies only.

    Each message is encoded as [header tokens][body tokens + <|eot_id|>]. Labels are
    -100 (ignored) everywhere except the bodies of assistant messages, so the model
    learns to *write* tool calls and explanations but never to predict the user's
    words or the tool's results (predicting tool results would teach it to make up
    numbers, which is exactly what we are trying to prevent).

    An assistant message with "train": False is shown as context but not learned
    (used for deliberately wrong tool calls in self-correction examples).
    Returns None when the example is longer than max_length (dropped, not truncated).
    """
    input_ids: List[int] = []
    labels: List[int] = []

    def add(text: str, trainable: bool) -> None:
        ids = encode(text)
        input_ids.extend(ids)
        labels.extend(ids if trainable else [-100] * len(ids))

    add(BOS, False)
    for m in messages:
        role = _role(m["role"])
        add(f"{START_HEADER}{role}{END_HEADER}\n\n", False)
        add(m["content"].strip() + EOT, role == "assistant" and m.get("train", True))
    if max_length is not None and len(input_ids) > max_length:
        return None
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}
