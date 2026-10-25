"""Fixed phrasing shared by the agent (at inference) and the data generator (at training).

Keeping these in one place guarantees the model is trained on exactly the
error messages and clarification style it will see when deployed.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

from pydantic import ValidationError

from .tools import Tool, get_tool
from .tools.fmt import join_and

CLARIFY_VARIANTS = [
    "I can {task}, but I need {missing} first. Could you share {pron}?",
    "Happy to help! To {task}, I just need {missing}.",
    "To {task}, I still need {missing}. What {be} {pron2}?",
    "Sure. Before I {task}, could you tell me {missing}?",
]


def missing_labels(tool: Tool, fields: Sequence[str]) -> List[str]:
    labels: List[str] = []
    for f in fields:
        label = tool.label(f) if f in tool.fields else f.replace("_", " ")
        if label not in labels:
            labels.append(label)
    return labels


def clarification_text(tool: Tool, fields: Sequence[str], variant: int = 0) -> str:
    labels = missing_labels(tool, fields)
    one = len(labels) == 1
    return CLARIFY_VARIANTS[variant % len(CLARIFY_VARIANTS)].format(
        task=tool.task, missing=join_and(labels), pron="it" if one else "them",
        be="is" if one else "are", pron2="it" if one else "they")


def missing_fields_from_error(error: ValidationError) -> List[str]:
    fields: List[str] = []
    for e in error.errors():
        if e["type"] == "missing" and e["loc"]:
            fields.append(str(e["loc"][0]))
        elif e["type"] == "missing_one_of":
            fields.append(str(e.get("ctx", {}).get("fields", "")).split(", ")[0])
    return [f for f in fields if f]


def format_validation_error(error: ValidationError) -> List[str]:
    details = []
    for e in error.errors():
        loc = ".".join(str(p) for p in e["loc"]) or "arguments"
        details.append(f"{loc}: {e['msg']}")
    return details


def validation_error_payload(tool_name: str, error: ValidationError) -> Dict[str, Any]:
    return {"error": "invalid_arguments", "tool": tool_name, "details": format_validation_error(error),
            "hint": "Fix the parameters and call the tool again, or ask the user for any missing value."}


def unknown_tool_payload(name: str) -> Dict[str, Any]:
    return {"error": "unknown_tool", "details": f"There is no tool named '{name}'.",
            "hint": "Use one of the tools listed in the system prompt."}


def parse_error_payload(message: str) -> Dict[str, Any]:
    return {"error": "invalid_json", "details": message,
            "hint": 'Reply with only {"name": "<tool>", "parameters": {...}}.'}


def ungrounded_payload(numbers: Sequence[str]) -> Dict[str, Any]:
    return {"error": "calculation_required",
            "details": f"Your reply contains numbers that no tool produced: {', '.join(numbers)}.",
            "hint": "Call a tool for any calculation instead of writing the result yourself."}


def domain_error_payload(message: str) -> Dict[str, Any]:
    return {"error": "cannot_compute", "details": message}


CAPABILITIES = ("I'm Financial Copilot. I answer money questions with exact calculations: loan and mortgage "
                "payments, compound interest, savings goals, retirement planning, investment returns like ROI, "
                "CAGR, NPV and IRR, budgeting, and small-business math like break-even and margins. Ask me "
                "something like \"How much will I have if I invest ten thousand dollars at 7% for 20 years?\"")

UNGROUNDED_FALLBACK = ("I don't want to give you a number I haven't calculated precisely. Could you rephrase the "
                       "question with the amounts, rate and time period involved?")

FAILED_CALL_FALLBACK = ("Sorry, I couldn't set up that calculation correctly. Could you rephrase the question, "
                        "including the amounts, rates and time period?")


def tool_by_name(name: str) -> Tool:
    tool = get_tool(name)
    if tool is None:
        raise KeyError(name)
    return tool
