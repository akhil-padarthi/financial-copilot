"""Generate the synthetic training, validation and test sets.

    python -m fincopilot.datagen.generate --out data

Example types (train):
  tool       question -> tool call -> tool result -> explanation
  followup   question missing a value -> clarifying question -> user answers -> tool call -> result -> explanation
  clarify    question missing a value -> clarifying question
  repair     question -> wrong call (not learned) -> error -> corrected call -> result -> explanation
  domain     question with impossible inputs -> tool call -> "cannot compute" -> plain explanation
  concept / advice / offtopic / greeting   plain-text replies, no tool

Test examples use held-out templates, so they measure generalization to new phrasings.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import ValidationError

from ..prompts import build_system_prompt
from ..responses import (clarification_text, domain_error_payload, unknown_tool_payload, validation_error_payload)
from ..tools import Tool, ToolDomainError, all_tools, clean_numbers, get_tool, to_json
from . import conversational as conv
from .realworld import REALWORLD
from .render import _ONES, _TENS, render_value
from .samplers import DOMAIN_SAMPLERS, SAMPLERS_WITH_SLOTS
from .templates import TEMPLATES

SLOT = re.compile(r"\{(\w+)(?::(\w+))?\}")

PREFIXES = ["Quick question: ", "Hey, ", "Hi! ", "Can you help me? ", "I'm trying to plan ahead. ", "Help me out: ",
            "Okay so ", "Question for you: ", "Hello. ", "Need some help. ", "Hi there, "]

TRAIN_PLAN = {"tool": 160, "followup": 460, "clarify": 260, "repair": 240, "domain": 100,
              "concept": 360, "advice": 130, "offtopic": 90, "greeting": 60}  # "tool" is per tool
TEST_PLAN = {"tool": 25, "followup": 40, "clarify": 40, "advice": 8, "offtopic": 8, "greeting": 6}

RENAMES = {"annual_rate": ["rate", "interest_rate"], "principal": ["amount", "loan_amount"], "years": ["term", "n_years"],
           "monthly_payment": ["payment"], "initial_investment": ["investment", "cost"], "cash_flows": ["flows"],
           "contribution": ["monthly_contribution"], "home_price": ["price"], "balance": ["amount_owed"],
           "discount_rate": ["rate"], "target_amount": ["goal"], "share_price": ["price"]}

FOLLOWUP_HINTS = {
    "gross_monthly_income": ["I make {v} a month", "{v} a month before taxes"],
    "gross_annual_income": ["I make {v} a year", "{v} a year before taxes"],
    "monthly_income": ["{v} a month"], "annual_income": ["{v} a year"],
    "annual_expenses": ["about {v} a year", "{v} per year"], "monthly_expenses": ["{v} a month", "{v} per month"],
    "annual_withdrawal": ["{v} a year"], "monthly_payment": ["{v} a month", "{v} per month"],
    "extra_monthly_payment": ["{v} extra a month"], "annual_cash_flow": ["{v} a year"],
    "cash_flows": ["the cash flows are {v}", "{v}"], "share_prices": ["the prices were {v}", "{v}"],
    "final_value": ["it's worth {v} now", "now it's {v}"], "share_price": ["the stock is at {v}", "it trades at {v}"],
}


def slots_of(template: str) -> List[str]:
    seen: List[str] = []
    for m in SLOT.finditer(template):
        if m.group(1) not in seen:
            seen.append(m.group(1))
    return seen


def render(template: str, tool: Tool, values: Dict[str, Any], rng: random.Random) -> str:
    return SLOT.sub(lambda m: render_value(tool.kind(m.group(1)), values[m.group(1)], rng, m.group(2) or ""), template)


_PROTECTED = set(_ONES) | set(_TENS) | {
    "hundred", "thousand", "million", "billion", "decade", "decades", "grand", "cents", "dollars", "bucks", "percent",
    "monthly", "quarterly", "annually", "yearly", "weekly", "daily", "biweekly", "semiannually", "twice", "annual",
    "month", "months", "years", "quarter", "every", "other", "weeks"}


def _typo(text: str, rng: random.Random) -> str:
    """Swap two letters in one ordinary word. Never touches words that carry a value."""
    words = text.split(" ")
    idx = [i for i, w in enumerate(words) if len(w) >= 5 and w.isalpha() and w.lower() not in _PROTECTED]
    if not idx:
        return text
    i = rng.choice(idx)
    w = words[i]
    j = rng.randrange(1, len(w) - 2)
    words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


def augment(text: str, rng: random.Random) -> str:
    """Light noise so the model doesn't depend on perfect grammar."""
    if rng.random() < 0.12:
        prefix = rng.choice(PREFIXES)
        if text[:1].isupper() and not re.match(r"^I\b|^I'", text) and not text[:2].isupper():
            text = text[0].lower() + text[1:]
        text = prefix + text
    if rng.random() < 0.08:
        text = text.lower()
    if rng.random() < 0.08:
        text = text.rstrip("?.!")
    if rng.random() < 0.06:
        text = _typo(text, rng)
    return text


def _ordered(tool: Tool, values: Dict[str, Any], keep: Sequence[str]) -> Dict[str, Any]:
    return {f: values[f] for f in tool.fields if f in keep}


def _msg(role: str, content: str, train: Optional[bool] = None) -> Dict[str, Any]:
    m: Dict[str, Any] = {"role": role, "content": content}
    if train is False:
        m["train"] = False
    return m


def call_json(name: str, params: Dict[str, Any]) -> str:
    return to_json({"name": name, "parameters": params})


class Generator:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.system = build_system_prompt("compact")

    # --- building blocks -------------------------------------------------------------
    def sample_call(self, tool: Tool, split: str, domain: bool = False):
        """Pick a template, sample values, and return (question, params, args, result_or_error)."""
        rng = self.rng
        sampler = (DOMAIN_SAMPLERS if domain else SAMPLERS_WITH_SLOTS)[tool.name]
        templates = TEMPLATES[tool.name][split]
        for _ in range(200):
            template = rng.choice(templates)
            slots = slots_of(template)
            if domain and tool.name == "payback_period" and "annual_cash_flow" in slots:
                continue
            values = sampler(rng, frozenset(slots))
            params = _ordered(tool, values, slots)
            try:
                args = tool.validate(params)
            except ValidationError:
                continue
            try:
                outcome: Any = tool.execute(args)
                if domain:
                    continue
            except ToolDomainError as exc:
                if not domain:
                    continue
                outcome = str(exc)
            question = augment(render(template, tool, values, rng), rng)
            return question, clean_numbers(params), args, outcome
        raise RuntimeError(f"could not sample a valid example for {tool.name}")

    def answer_turns(self, tool: Tool, params, args, result) -> List[Dict[str, Any]]:
        return [_msg("assistant", call_json(tool.name, params)), _msg("tool", to_json(result)),
                _msg("assistant", tool.explain(args, result))]

    def followup_text(self, tool: Tool, missing: Sequence[str], values: Dict[str, Any]) -> str:
        rng = self.rng
        parts = []
        for f in missing:
            kind = tool.kind(f)
            v = render_value(kind, values[f], rng)
            if f in FOLLOWUP_HINTS:
                parts.append(rng.choice(FOLLOWUP_HINTS[f]).format(v=v))
            elif kind == "freq":
                parts.append(rng.choice([f"it compounds {v}", f"compounded {v}"]))
            elif kind in ("years", "months"):
                parts.append(rng.choice([v, f"over {v}", f"{tool.label(f)} is {v}"]))
            else:
                parts.append(rng.choice([v, v, f"{tool.label(f)} is {v}"]))
        text = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + rng.choice([" and ", ", "]) + parts[-1]
        text = rng.choice(["", "", "", "Sure, ", "Oh, ", "Sorry, ", "Right, "]) + text
        text = text[0].upper() + text[1:]
        return text + rng.choice(["", ".", "."])

    def partial(self, tool: Tool):
        rng = self.rng
        template, missing = rng.choice(TEMPLATES[tool.name]["partial"])
        slots = slots_of(template)
        for _ in range(200):
            values = SAMPLERS_WITH_SLOTS[tool.name](rng, frozenset(slots + list(missing)))
            full = _ordered(tool, values, slots + list(missing))
            try:
                args = tool.validate(full)
                result = tool.execute(args)
            except (ValidationError, ToolDomainError):
                continue
            question = augment(render(template, tool, values, rng), rng)
            return question, missing, values, clean_numbers(full), args, result
        raise RuntimeError(f"could not sample a partial example for {tool.name}")

    def partial_tools(self) -> List[Tool]:
        return [t for t in all_tools() if TEMPLATES[t.name].get("partial")]

    # --- example types ------------------------------------------------------------------
    def ex_tool(self, tool: Tool, split: str):
        q, params, args, result = self.sample_call(tool, split)
        return ({"type": "tool", "tool": tool.name, "messages": [_msg("user", q)] + self.answer_turns(tool, params, args, result)},
                {"action": "tool", "name": tool.name, "parameters": params})

    def ex_clarify(self, tool: Tool):
        q, missing, _values, _full, _args, _result = self.partial(tool)
        clarification = clarification_text(tool, missing, variant=self.rng.randrange(4))
        return ({"type": "clarify", "tool": tool.name, "messages": [_msg("user", q), _msg("assistant", clarification)]},
                {"action": "clarify", "name": tool.name, "missing": list(missing)})

    def ex_followup(self, tool: Tool):
        q, missing, values, full, args, result = self.partial(tool)
        clarification = clarification_text(tool, missing, variant=self.rng.randrange(4))
        msgs = [_msg("user", q), _msg("assistant", clarification), _msg("user", self.followup_text(tool, missing, values))]
        return ({"type": "followup", "tool": tool.name, "messages": msgs + self.answer_turns(tool, full, args, result)},
                {"action": "tool", "name": tool.name, "parameters": full})

    def ex_repair(self, tool: Tool):
        rng = self.rng
        q, params, args, result = self.sample_call(tool, "train")
        if rng.random() < 0.65 and params:
            key = rng.choice(list(params))
            new_key = rng.choice(RENAMES.get(key, [key.split("_")[-1], key + "_amount"]))
            if new_key in tool.fields or new_key == key:
                new_key = key + "_value"
            bad = {(new_key if k == key else k): v for k, v in params.items()}
            try:
                tool.validate(bad)
                raise AssertionError("renamed call unexpectedly validated")
            except ValidationError as exc:
                payload = validation_error_payload(tool.name, exc)
            bad_call = call_json(tool.name, bad)
        else:
            bad_name = rng.choice([f"calculate_{tool.name}", f"{tool.name}_calculator", tool.name.replace("_", ""),
                                   f"get_{tool.name}"])
            payload = unknown_tool_payload(bad_name)
            bad_call = call_json(bad_name, params)
        msgs = [_msg("user", q), _msg("assistant", bad_call, train=False), _msg("tool", to_json(payload))]
        return ({"type": "repair", "tool": tool.name, "messages": msgs + self.answer_turns(tool, params, args, result)}, None)

    def ex_domain(self, tool: Tool):
        q, params, _args, error = self.sample_call(tool, "train", domain=True)
        msgs = [_msg("user", q), _msg("assistant", call_json(tool.name, params)),
                _msg("tool", to_json(domain_error_payload(error))), _msg("assistant", error)]
        return ({"type": "domain", "tool": tool.name, "messages": msgs},
                {"action": "tool", "name": tool.name, "parameters": params})

    def ex_text(self, kind: str, question: str, answer: str, augment_text: bool = True):
        q = augment(question, self.rng) if augment_text else question
        return ({"type": kind, "tool": None, "messages": [_msg("user", q), _msg("assistant", answer)]}, {"action": "text"})

    # --- datasets -------------------------------------------------------------------------
    def build_train(self, scale: float = 1.0) -> List[Dict[str, Any]]:
        rng, out = self.rng, []
        n = {k: max(1, round(v * scale)) for k, v in TRAIN_PLAN.items()}
        tools = all_tools()
        for tool in tools:
            out += [self.ex_tool(tool, "train")[0] for _ in range(n["tool"])]
        partial_tools = self.partial_tools()
        out += [self.ex_followup(rng.choice(partial_tools))[0] for _ in range(n["followup"])]
        out += [self.ex_clarify(rng.choice(partial_tools))[0] for _ in range(n["clarify"])]
        out += [self.ex_repair(rng.choice(tools))[0] for _ in range(n["repair"])]
        domain_tools = [get_tool(name) for name in DOMAIN_SAMPLERS]
        out += [self.ex_domain(rng.choice(domain_tools))[0] for _ in range(n["domain"])]
        concept_pairs = [(q, a) for qs, _t, a in conv.CONCEPTS for q in qs]
        for _ in range(n["concept"]):
            q, a = rng.choice(concept_pairs)
            out.append(self.ex_text("concept", q, a)[0])
        out += [self.ex_text("advice", rng.choice(conv.ADVICE_TRAIN), rng.choice(conv.ADVICE_RESPONSES))[0] for _ in range(n["advice"])]
        out += [self.ex_text("offtopic", rng.choice(conv.OFFTOPIC_TRAIN), rng.choice(conv.OFFTOPIC_RESPONSES))[0] for _ in range(n["offtopic"])]
        out += [self.ex_text("greeting", rng.choice(conv.GREETINGS_TRAIN), rng.choice(conv.GREETING_RESPONSES))[0] for _ in range(n["greeting"])]
        rng.shuffle(out)
        for i, ex in enumerate(out):
            ex["messages"] = [_msg("system", self.system)] + ex["messages"]
            ex["id"] = f"train-{i:05d}"
        return out

    def build_test(self) -> List[Dict[str, Any]]:
        rng, out = self.rng, []

        def add(pair):
            ex, expected = pair
            # The model is shown everything up to (not including) the assistant turn being tested.
            first_assistant = max(i for i, m in enumerate(ex["messages"]) if m["role"] == "user")
            ex["messages"] = [{"role": m["role"], "content": m["content"]} for m in ex["messages"][:first_assistant + 1]]
            ex["expected"] = expected
            out.append(ex)

        for tool in all_tools():
            for _ in range(TEST_PLAN["tool"]):
                add(self.ex_tool(tool, "test"))
        partial_tools = self.partial_tools()
        for _ in range(TEST_PLAN["followup"]):
            add(self.ex_followup(rng.choice(partial_tools)))
        for _ in range(TEST_PLAN["clarify"]):
            add(self.ex_clarify(rng.choice(partial_tools)))
        for _qs, test_qs, answer in conv.CONCEPTS:
            for q in test_qs:
                add(self.ex_text("concept", q, answer, augment_text=False))
        for kind, pool, responses in (("advice", conv.ADVICE_TEST, conv.ADVICE_RESPONSES),
                                      ("offtopic", conv.OFFTOPIC_TEST, conv.OFFTOPIC_RESPONSES),
                                      ("greeting", conv.GREETINGS_TEST, conv.GREETING_RESPONSES)):
            for i in range(TEST_PLAN[kind]):
                add(self.ex_text(kind, pool[i % len(pool)], responses[0], augment_text=i >= len(pool)))
        for i, ex in enumerate(out):
            ex["id"] = f"test-{i:05d}"
        return out


def build_realworld() -> List[Dict[str, Any]]:
    out = []
    for i, (question, spec) in enumerate(REALWORLD):
        if spec[0] == "tool":
            _, name, params = spec
            tool = get_tool(name)
            tool.run(params)  # gold labels must validate and compute
            expected = {"action": "tool", "name": name, "parameters": clean_numbers(params)}
            kind = "tool"
        elif spec[0] == "clarify":
            expected, kind = {"action": "clarify", "name": spec[1]}, "clarify"
        else:
            expected, kind = {"action": "text"}, "text"
        out.append({"id": f"realworld-{i:03d}", "type": kind, "tool": spec[1] if spec[0] == "tool" else None,
                    "messages": [{"role": "user", "content": question}], "expected": expected})
    return out


def write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale", type=float, default=1.0, help="multiply training-set size (1.0 = ~7.7k examples)")
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    train = Generator(args.seed).build_train(args.scale)
    val = Generator(args.seed + 1).build_train(0.03)
    test = Generator(args.seed + 2).build_test()
    realworld = build_realworld()

    train_users = {m["content"].strip().lower() for ex in train for m in ex["messages"] if m["role"] == "user"}
    leaked = [ex["id"] for ex in test + realworld
              if ex["expected"]["action"] == "tool" and ex["messages"][-1]["content"].strip().lower() in train_users]

    for name, rows in (("train", train), ("val", val), ("test", test), ("realworld_test", realworld)):
        write_jsonl(os.path.join(args.out, f"{name}.jsonl"), rows)

    stats = {
        "train_examples": len(train), "val_examples": len(val), "test_examples": len(test),
        "realworld_examples": len(realworld), "tools": len(all_tools()),
        "train_by_type": dict(Counter(ex["type"] for ex in train)),
        "train_by_tool": dict(sorted(Counter(ex["tool"] for ex in train if ex["tool"]).items())),
        "test_by_type": dict(Counter(ex["type"] for ex in test)),
        "test_tool_questions_also_in_train": len(leaked),
        "approx_tokens_per_train_example": round(sum(len("".join(m["content"] for m in ex["messages"])) for ex in train) / len(train) / 3.6),
    }
    with open(os.path.join(args.out, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps({k: v for k, v in stats.items() if k != "train_by_tool"}, indent=2))


if __name__ == "__main__":
    main()
