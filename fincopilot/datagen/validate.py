"""Audit a generated dataset before you spend GPU time training on it.

    python -m fincopilot.datagen.validate data

Checks every example:
  * every tool call parses, validates, and reproduces the stored tool result
  * every gold value can be read from the user's own words (no impossible labels)
  * every explanation and plain-text reply passes the agent's grounding check
  * every test/real-world label computes
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import List, Tuple

from ..grounding import extract_numbers, numbers_in_messages, ungrounded_numbers
from ..parsing import ToolCall, parse_model_output
from ..tools import get_tool
from .render import int_words

_FREQ_WORDS = {1: ["annual", "yearly", "once a year", "every year", "each year", "a year", "per year"],
               2: ["semi-annual", "semiannual", "twice a year"], 4: ["quarter"], 12: ["month"],
               26: ["two weeks", "biweekly", "other week"], 52: ["week"], 365: ["daily", "every day"]}


def readable_numbers(text: str):
    lowered = text.lower()
    values = {v for _, v, _ in extract_numbers(text)}
    values |= {abs(v) for v in values}  # "-120,000 now" is an outflow of 120,000
    for m in re.finditer(r"negative\s+\$?(\d[\d,]*(?:\.\d+)?)", lowered):
        values.add(-float(m.group(1).replace(",", "")))
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*grand", lowered):
        values.add(float(m.group(1)) * 1000)
    for m in re.finditer(r"(\d+)\s*cents", lowered):
        values.add(int(m.group(1)) / 100)
    return values, lowered


def value_in_text(value, kind: str, text: str) -> bool:
    values, lowered = readable_numbers(text)
    if kind == "freq":
        return any(w in lowered for w in _FREQ_WORDS[int(value)])
    if any(abs(v - value) < 1e-6 for v in values):
        return True
    if float(value).is_integer():
        n = int(value)
        words = [int_words(abs(n))]
        if n >= 1000 and n % 1000 == 0:
            words.append(int_words(n // 1000) + " thousand")
        words += {10: ["decade"], 20: ["two decades"], 1: ["a year", "one year"],
                  1_000_000: ["a million", "one million"]}.get(n, [])
        return any(w in lowered for w in words)
    return False


def audit_training_file(path: str) -> Tuple[int, int, List[Tuple[str, str]]]:
    problems: List[Tuple[str, str]] = []
    n_calls = n_values = 0
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    for ex in rows:
        msgs = ex["messages"]
        user_text = " ".join(m["content"] for m in msgs if m["role"] == "user")
        for i, m in enumerate(msgs):
            if m["role"] != "assistant" or m.get("train") is False:
                continue
            parsed = parse_model_output(m["content"])
            if isinstance(parsed, ToolCall):
                n_calls += 1
                tool = get_tool(parsed.name)
                args = tool.validate(parsed.parameters)
                result = json.loads(msgs[i + 1]["content"])
                if "error" not in result:
                    if tool.execute(args) != result:
                        problems.append((ex["id"], "stored result does not match the tool"))
                    bad = ungrounded_numbers(msgs[i + 2]["content"], tool.allowed_numbers(args, result))
                    if bad:
                        problems.append((ex["id"], f"ungrounded explanation numbers {bad}"))
                for field, value in parsed.parameters.items():
                    for v in (value if isinstance(value, list) else [value]):
                        n_values += 1
                        if not value_in_text(v, tool.kind(field), user_text):
                            problems.append((ex["id"], f"{field}={v} not found in the question"))
            elif msgs[i - 1]["role"] != "tool":
                bad = ungrounded_numbers(m["content"], numbers_in_messages(msgs[:i]), small_int_max=100)
                if bad:
                    problems.append((ex["id"], f"ungrounded reply numbers {bad}"))
    return n_calls, n_values, problems


def main(argv=None) -> int:
    folder = (argv or sys.argv[1:] or ["data"])[0]
    total_problems = []
    for name in ("train.jsonl", "val.jsonl"):
        path = os.path.join(folder, name)
        if os.path.exists(path):
            calls, values, problems = audit_training_file(path)
            print(f"{name}: {calls} tool calls, {values} extracted values, {len(problems)} problems")
            total_problems += problems
    for name in ("test.jsonl", "realworld_test.jsonl"):
        path = os.path.join(folder, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f]
            for row in rows:
                exp = row["expected"]
                if exp["action"] == "tool":
                    get_tool(exp["name"]).run(exp["parameters"])
            print(f"{name}: {len(rows)} labels compute")
    for pid, what in total_problems[:20]:
        print(f"  {pid}: {what}")
    print("OK" if not total_problems else f"{len(total_problems)} problems")
    return 1 if total_problems else 0


if __name__ == "__main__":
    sys.exit(main())
