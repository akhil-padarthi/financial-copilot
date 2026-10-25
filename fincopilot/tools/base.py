"""Tool registry, shared argument types and the execution contract.

Every calculator in Financial Copilot is a *tool*: a plain Python function plus
a Pydantic model describing its arguments. The language model never does math.
It only picks a tool name and fills in the argument form; this module checks
that form (types, ranges, required fields) before any code runs.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Annotated, Any, Callable, ClassVar, Dict, Iterable, List, Optional, Sequence, Tuple, Type

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

# ---------------------------------------------------------------------------
# Number coercion: accept what an LLM (or a person) might write, e.g. "$10,000",
# "7%", "10k", "1.2 million". Anything we cannot read unambiguously is rejected.
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"^(-?\d+(?:\.\d+)?)\s*(k|thousand|m|mm|mn|million|b|bn|billion)?$")
_SCALE = {
    "k": 1e3, "thousand": 1e3,
    "m": 1e6, "mm": 1e6, "mn": 1e6, "million": 1e6,
    "b": 1e9, "bn": 1e9, "billion": 1e9,
}


def coerce_number(value: Any) -> Any:
    """Convert friendly numeric strings to numbers; leave real numbers alone."""
    if isinstance(value, bool):
        raise ValueError("expected a number, not true/false")
    if isinstance(value, str):
        text = value.strip().lower()
        for junk in (",", "$", "%", "usd", " "):
            text = text.replace(junk, "")
        match = _NUMBER_RE.match(text)
        if not match:
            raise ValueError(f"could not read {value!r} as a number")
        number = float(match.group(1))
        if match.group(2):
            number *= _SCALE[match.group(2)]
        return number
    return value


def coerce_number_list(value: Any) -> Any:
    """Accept a JSON list, or a string like "15000, 20000 and 25000"."""
    if isinstance(value, str):
        parts = re.split(r"\s*(?:,|;|\band\b)\s*", value.strip())
        return [coerce_number(p) for p in parts if p]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [value]
    return value


Number = Annotated[float, BeforeValidator(coerce_number)]
Integer = Annotated[int, BeforeValidator(coerce_number)]
NumberList = Annotated[List[Number], BeforeValidator(coerce_number_list)]


def arg(kind: str, label: str, description: str, default: Any = ..., **constraints: Any) -> Any:
    """Declare a tool argument.

    kind        -- how the value is written in questions (money, percent, years, ...);
                   used by the synthetic data generator.
    label       -- noun phrase used when asking the user for a missing value,
                   e.g. "the annual interest rate".
    description -- shown to the model in the tool catalog.
    """
    return Field(default, description=description,
                 json_schema_extra={"kind": kind, "label": label}, **constraints)


class ToolArgs(BaseModel):
    """Base class for every tool's argument form.

    * extra="forbid": an invented parameter name is an error, not silently ignored.
    * allow_inf_nan=False: NaN/Infinity are rejected.
    * one_of: groups of alternative fields, e.g. (("years", "term_months"), True)
      means exactly one of years/term_months must be given.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    one_of: ClassVar[Tuple[Tuple[Tuple[str, ...], bool], ...]] = ()

    @model_validator(mode="after")
    def _check_one_of(self) -> "ToolArgs":
        for fields, required in self.one_of:
            given = [f for f in fields if getattr(self, f) is not None]
            if len(given) > 1:
                raise PydanticCustomError(
                    "conflicting_fields", "Provide only one of: {fields}.", {"fields": ", ".join(fields)})
            if required and not given:
                raise PydanticCustomError(
                    "missing_one_of", "Provide one of: {fields}.", {"fields": ", ".join(fields)})
        return self


class ToolDomainError(ValueError):
    """The inputs are valid, but the scenario has no meaningful answer.

    Example: a monthly payment smaller than the interest charged, so the loan
    is never paid off. The message is written for the end user.
    """


# ---------------------------------------------------------------------------
# Rounding and JSON helpers shared by tools, prompts and the data generator.
# ---------------------------------------------------------------------------

def round_floats(obj: Any, ndigits: int = 2) -> Any:
    if isinstance(obj, bool) or obj is None:
        return obj
    if isinstance(obj, float):
        value = round(obj, ndigits)
        return 0.0 if value == 0 else value
    if isinstance(obj, dict):
        return {k: round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_floats(v, ndigits) for v in obj]
    return obj


def clean_numbers(obj: Any) -> Any:
    """Render integral floats as ints (10000.0 -> 10000) for compact JSON."""
    if isinstance(obj, bool) or obj is None:
        return obj
    if isinstance(obj, float) and math.isfinite(obj) and obj == int(obj) and abs(obj) < 1e15:
        return int(obj)
    if isinstance(obj, dict):
        return {k: clean_numbers(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_numbers(v) for v in obj]
    return obj


def to_json(obj: Any) -> str:
    import json
    return json.dumps(clean_numbers(obj), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Tool:
    name: str
    category: str
    description: str
    task: str  # infinitive phrase for clarifications: "work out your monthly loan payment"
    args_model: Type[ToolArgs]
    func: Callable[..., Dict[str, Any]]
    explain_fn: Callable[[Any, Dict[str, Any]], str]
    constants: Tuple[float, ...] = ()

    # --- introspection -----------------------------------------------------
    @property
    def fields(self) -> Dict[str, Any]:
        return self.args_model.model_fields

    def field_meta(self, name: str) -> Dict[str, Any]:
        extra = self.fields[name].json_schema_extra or {}
        return dict(extra) if isinstance(extra, dict) else {}

    def label(self, name: str) -> str:
        return self.field_meta(name).get("label", name.replace("_", " "))

    def kind(self, name: str) -> str:
        return self.field_meta(name).get("kind", "number")

    def one_of_groups(self) -> Sequence[Tuple[Tuple[str, ...], bool]]:
        return self.args_model.one_of

    def required_fields(self) -> List[str]:
        return [n for n, f in self.fields.items() if f.is_required()]

    def signature(self) -> str:
        """Compact signature used in the system prompt, e.g.
        loan_payment(principal, annual_rate, years|term_months, payments_per_year=12)"""
        grouped = {f: g for g in self.one_of_groups() for f in g[0]}
        parts, done = [], set()
        for name, info in self.fields.items():
            if name in done:
                continue
            if name in grouped:
                fields, required = grouped[name]
                done.update(fields)
                alt = "|".join(fields)
                parts.append(alt if required else f"[{alt}]")
                continue
            if info.is_required():
                parts.append(name)
            else:
                default = info.default
                parts.append(f"{name}={clean_numbers(default)}")
        return f"{self.name}({', '.join(parts)})"

    def doc(self) -> str:
        """Longer description used by the 'full' prompt style (for base models)."""
        lines = [f"{self.name}: {self.description}"]
        grouped = {f: g for g in self.one_of_groups() for f in g[0]}
        for name, info in self.fields.items():
            if name in grouped:
                fields, required = grouped[name]
                need = "give exactly one of " if required else "optional, at most one of "
                tag = f"{need}{' / '.join(fields)}"
            elif info.is_required():
                tag = "required"
            else:
                tag = f"optional, default {clean_numbers(info.default)}"
            lines.append(f"  - {name} ({tag}): {info.description}")
        return "\n".join(lines)

    # --- execution ---------------------------------------------------------
    def validate(self, raw: Dict[str, Any]) -> ToolArgs:
        return self.args_model.model_validate(raw)

    def execute(self, args: ToolArgs) -> Dict[str, Any]:
        result = self.func(**args.model_dump())
        return round_floats(result)

    def run(self, raw: Dict[str, Any]) -> "ToolRun":
        args = self.validate(raw)
        return ToolRun(tool=self, args=args, result=self.execute(args))

    def explain(self, args: ToolArgs, result: Dict[str, Any]) -> str:
        return self.explain_fn(args, result)

    def allowed_numbers(self, args: ToolArgs, result: Dict[str, Any]) -> List[float]:
        """Numbers an explanation may mention (for the grounding check)."""
        values: List[float] = list(self.constants)
        _collect_numbers(args.model_dump(), values)
        _collect_numbers(result, values)
        return values


def _collect_numbers(obj: Any, out: List[float]) -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_numbers(v, out)


@dataclass
class ToolRun:
    tool: Tool
    args: ToolArgs
    result: Dict[str, Any]

    @property
    def explanation(self) -> str:
        return self.tool.explain(self.args, self.result)


REGISTRY: Dict[str, Tool] = {}


def register(name: str, *, category: str, description: str, task: str,
             args: Type[ToolArgs], explain: Callable[[Any, Dict[str, Any]], str],
             constants: Iterable[float] = ()) -> Callable:
    def decorator(func: Callable[..., Dict[str, Any]]) -> Callable[..., Dict[str, Any]]:
        if name in REGISTRY:
            raise ValueError(f"duplicate tool name {name}")
        REGISTRY[name] = Tool(name=name, category=category, description=description, task=task,
                              args_model=args, func=func, explain_fn=explain,
                              constants=tuple(float(c) for c in constants))
        return func
    return decorator


def get_tool(name: str) -> Optional[Tool]:
    return REGISTRY.get(name)


def all_tools() -> List[Tool]:
    return list(REGISTRY.values())


# ---------------------------------------------------------------------------
# Shared financial primitives
# ---------------------------------------------------------------------------

def payment(principal: float, annual_rate_pct: float, n_payments: int, periods_per_year: int = 12) -> float:
    """Level payment that amortizes `principal` over `n_payments` periods."""
    i = annual_rate_pct / 100 / periods_per_year
    if n_payments <= 0:
        raise ToolDomainError("The loan term must cover at least one payment.")
    if i == 0:
        return principal / n_payments
    return principal * i / (1 - (1 + i) ** -n_payments)


def simulate_payoff(balance: float, monthly_rate: float, monthly_payment: float,
                    max_months: int = 1200) -> Tuple[int, float]:
    """Month-by-month payoff. Returns (months, total_interest). The last payment
    may be partial. Raises ToolDomainError if the debt never gets paid off."""
    months, total_interest, b = 0, 0.0, balance
    while b > 0.005:
        if months >= max_months:
            raise ToolDomainError("At that payment the debt would take more than 100 years to pay off.")
        interest = b * monthly_rate
        total_interest += interest
        b = b + interest - monthly_payment
        months += 1
    return months, total_interest
