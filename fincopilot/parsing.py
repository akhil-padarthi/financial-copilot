"""Turn raw model text into either a tool call or a plain-text reply."""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union


@dataclass
class ToolCall:
    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    raw: str = ""


@dataclass
class TextReply:
    text: str


class ParseError(ValueError):
    pass


_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.S)
_CALL_START = re.compile(r"\{\s*[\"'](?:name|tool|function|tool_calls|type)[\"']\s*:")
_FUNCTION_TAG = re.compile(r"<function=([A-Za-z_][\w.-]*)>")


def _first_json_object(text: str, start: int) -> Optional[str]:
    """Return the balanced {...} starting at `start`, respecting strings."""
    depth, in_str, escape, quote = 0, False, False, ""
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                in_str = False
        elif ch in "\"'":
            in_str, quote = True, ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _load(candidate: str) -> Any:
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    try:  # tolerate Python-style dicts with single quotes / True / None
        return ast.literal_eval(candidate)
    except (ValueError, SyntaxError) as exc:
        raise ParseError(f"Malformed JSON tool call: {exc}") from None


def parse_model_output(text: str) -> Union[ToolCall, TextReply]:
    cleaned = text.replace("<|python_tag|>", "").strip()
    fence = _FENCE.match(cleaned)
    if fence:
        cleaned = fence.group(1).strip()

    # Llama 3.1's alternative custom-tool format: <function=name>{"arg": ...}</function>
    tagged = _FUNCTION_TAG.search(cleaned)
    if tagged:
        body_start = cleaned.find("{", tagged.end())
        body = _first_json_object(cleaned, body_start) if body_start != -1 else "{}"
        if body is None:
            raise ParseError("Malformed JSON tool call: unbalanced braces")
        params = _load(body)
        if not isinstance(params, dict):
            raise ParseError('"parameters" must be a JSON object')
        return ToolCall(name=tagged.group(1), parameters=params, raw=cleaned[tagged.start():])

    if cleaned.startswith("{"):
        start = 0
    else:
        match = _CALL_START.search(cleaned)
        if not match:
            return TextReply(text.strip())
        start = match.start()

    candidate = _first_json_object(cleaned, start)
    if candidate is None:
        raise ParseError("Malformed JSON tool call: unbalanced braces")
    obj = _load(candidate)
    if not isinstance(obj, dict):
        raise ParseError("A tool call must be a JSON object")
    # OpenAI-style wrappers: {"tool_calls": [{"function": {...}}]} or {"function": {"name": ..., "arguments": ...}}
    if isinstance(obj.get("tool_calls"), list) and obj["tool_calls"] and isinstance(obj["tool_calls"][0], dict):
        obj = obj["tool_calls"][0]
    if isinstance(obj.get("function"), dict):
        obj = obj["function"]
    name = obj.get("name") or obj.get("tool") or obj.get("function")
    params = obj.get("parameters", obj.get("arguments", obj.get("args", {})))
    if isinstance(params, str):
        params = _load(params) if params.strip() else {}
    if not isinstance(name, str) or not name:
        raise ParseError('A tool call needs a "name"')
    if not isinstance(params, dict):
        raise ParseError('"parameters" must be a JSON object')
    return ToolCall(name=name.strip(), parameters=params, raw=candidate)
