"""The agent loop: question -> route -> extract -> validate -> calculate -> explain.

    agent = Agent(backend)
    reply = agent.chat(history=[], user_text="If I invest $10,000 at 7% for 10 years, how much will I have?")
    reply.text      # plain-English answer
    reply.steps     # what happened at each stage (shown in the web UI)
    reply.history   # pass back in for follow-up questions

Safety nets, in order:
1. Parse errors, unknown tools and rejected arguments are sent back to the model
   once as a tool error so it can correct itself (self-repair).
2. If the model still leaves out something the user never said, we ask the user
   instead of guessing.
3. Python does every calculation.
4. The model's explanation must pass the grounding check; otherwise we use a
   template explanation built only from the tool's output.
5. A plain-text reply that contains numbers the user never gave (i.e. the model
   tried to do math in its head) is rejected the same way.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from pydantic import ValidationError

from .backends.base import Backend, Message
from .grounding import numbers_in_messages, ungrounded_numbers
from .parsing import ParseError, TextReply, ToolCall, parse_model_output
from .prompts import with_system
from .responses import (FAILED_CALL_FALLBACK, UNGROUNDED_FALLBACK, clarification_text, domain_error_payload,
                        format_validation_error, missing_fields_from_error, parse_error_payload, ungrounded_payload,
                        unknown_tool_payload, validation_error_payload)
from .tools import ToolDomainError, clean_numbers, get_tool, to_json


@dataclass
class Step:
    stage: str    # route | extract | validate | calculate | explain | clarify | ground
    status: str   # ok | fixed | failed | info
    detail: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentReply:
    text: str
    kind: str  # answer | clarification | text | cannot_compute | error
    history: List[Message]
    steps: List[Step] = field(default_factory=list)
    tool: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    explanation_source: Optional[str] = None  # model | template
    model_calls: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "kind": self.kind, "tool": self.tool, "arguments": self.arguments,
                "result": self.result, "explanation_source": self.explanation_source,
                "model_calls": self.model_calls,
                "steps": [s.__dict__ for s in self.steps], "history": self.history}


def _assistant(text: str) -> Message:
    return {"role": "assistant", "content": text}


def _tool(payload: Any) -> Message:
    return {"role": "tool", "content": to_json(payload)}


class Agent:
    def __init__(self, backend: Backend, prompt_style: str = "compact", explain_with: str = "model",
                 max_repairs: int = 1, max_new_tokens: int = 384):
        if explain_with not in ("model", "template"):
            raise ValueError("explain_with must be 'model' or 'template'")
        self.backend, self.prompt_style, self.explain_with = backend, prompt_style, explain_with
        self.max_repairs, self.max_new_tokens = max_repairs, max_new_tokens

    def _generate(self, messages: Sequence[Message]) -> str:
        self._calls += 1
        return self.backend.generate(with_system(messages, self.prompt_style), self.max_new_tokens)

    def chat(self, history: Sequence[Message], user_text: str) -> AgentReply:
        self._calls = 0
        convo: List[Message] = [m for m in history if m["role"] != "system"]
        convo.append({"role": "user", "content": user_text.strip()})
        steps: List[Step] = []
        work = list(convo)  # what the model sees this turn, including any repair feedback

        def finish(text: str, kind: str, extra: Sequence[Message] = (), **kw: Any) -> AgentReply:
            return AgentReply(text=text, kind=kind, history=convo + list(extra) + [_assistant(text)],
                              steps=steps, model_calls=self._calls, **kw)

        raw = self._generate(work)
        for attempt in range(self.max_repairs + 1):
            can_retry = attempt < self.max_repairs
            try:
                parsed = parse_model_output(raw)
            except ParseError as exc:
                steps.append(Step("route", "failed", f"Couldn't read the tool call: {exc}", {"raw": raw}))
                if can_retry:
                    work += [_assistant(raw), _tool(parse_error_payload(str(exc)))]
                    raw = self._generate(work)
                    continue
                return finish(FAILED_CALL_FALLBACK, "error")

            # --- a plain-text reply (clarifying question, concept answer, refusal) ---
            if isinstance(parsed, TextReply):
                bad = ungrounded_numbers(parsed.text, numbers_in_messages(convo), small_int_max=100)
                if bad:
                    steps.append(Step("ground", "failed",
                                      f"The reply contained numbers no tool produced ({', '.join(bad)}).", {"raw": raw}))
                    if can_retry:
                        work += [_assistant(raw), _tool(ungrounded_payload(bad))]
                        raw = self._generate(work)
                        continue
                    return finish(UNGROUNDED_FALLBACK, "error")
                steps.append(Step("route", "ok", "No calculation needed, so the model answered directly."))
                return finish(parsed.text, "text")

            # --- a tool call ---
            assert isinstance(parsed, ToolCall)
            tool = get_tool(parsed.name)
            if tool is None:
                steps.append(Step("route", "failed", f"The model asked for a tool that doesn't exist: '{parsed.name}'."))
                if can_retry:
                    work += [_assistant(raw), _tool(unknown_tool_payload(parsed.name))]
                    raw = self._generate(work)
                    continue
                return finish(FAILED_CALL_FALLBACK, "error")

            steps.append(Step("route", "ok", f"Picked the {tool.name} calculator.", {"tool": tool.name}))
            steps.append(Step("extract", "info", "Pulled these values from the question.",
                              {"parameters": parsed.parameters}))
            try:
                args = tool.validate(parsed.parameters)
            except ValidationError as exc:
                details = format_validation_error(exc)
                steps.append(Step("validate", "failed", "The form was rejected: " + "; ".join(details)))
                if can_retry:
                    work += [_assistant(raw), _tool(validation_error_payload(tool.name, exc))]
                    raw = self._generate(work)
                    continue
                missing = missing_fields_from_error(exc)
                if missing:
                    text = clarification_text(tool, missing)
                    steps.append(Step("clarify", "info", "Asked for the missing information instead of guessing.",
                                      {"missing": missing}))
                    return finish(text, "clarification", tool=tool.name)
                return finish(FAILED_CALL_FALLBACK, "error", tool=tool.name)

            used = {k: getattr(args, k) for k in parsed.parameters if k in tool.fields}
            arguments = clean_numbers(args.model_dump(exclude_none=True))
            steps.append(Step("validate", "ok" if attempt == 0 else "fixed",
                              "Every field passed validation." if attempt == 0 else
                              "The model corrected its call after feedback, and it passed validation.",
                              {"arguments": arguments}))
            call = [_assistant(to_json({"name": tool.name, "parameters": used}))]

            try:
                result = tool.execute(args)
            except ToolDomainError as exc:
                steps.append(Step("calculate", "failed", str(exc)))
                return finish(str(exc), "cannot_compute", call + [_tool(domain_error_payload(str(exc)))],
                              tool=tool.name, arguments=arguments)
            steps.append(Step("calculate", "ok", "Python computed the result.", {"result": result}))
            context = call + [_tool(result)]

            explanation, source = tool.explain(args, result), "template"
            if self.explain_with == "model":
                text = self._generate(convo + context).strip()
                bad = ungrounded_numbers(text, tool.allowed_numbers(args, result))
                is_call = not isinstance(_safe_parse(text), TextReply)
                if text and not bad and not is_call:
                    explanation, source = text, "model"
                    steps.append(Step("explain", "ok", "The model's explanation passed the grounding check."))
                else:
                    why = f"it mentioned numbers no tool produced ({', '.join(bad)})" if bad else "it wasn't a usable explanation"
                    steps.append(Step("explain", "fixed", f"Replaced the model's explanation because {why}.",
                                      {"rejected": text}))
            else:
                steps.append(Step("explain", "info", "Explanation built from the tool output."))
            return finish(explanation, "answer", context, tool=tool.name, arguments=arguments, result=result,
                          explanation_source=source)

        return finish(FAILED_CALL_FALLBACK, "error")  # unreachable, kept for safety


def _safe_parse(text: str):
    try:
        return parse_model_output(text)
    except ParseError:
        return None
