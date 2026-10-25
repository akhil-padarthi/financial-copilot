"""Formatting helpers for plain-English explanations.

Explanations are built only from tool inputs and outputs, so every number the
user reads was produced by Python, not by the language model.
"""
from __future__ import annotations

from typing import Iterable


def _trim(x: float, places: int = 2) -> str:
    text = f"{x:,.{places}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def money(x: float) -> str:
    sign = "-" if x < 0 else ""
    x = abs(x)
    body = f"{x:,.0f}" if abs(x - round(x)) < 0.005 else f"{x:,.2f}"
    return f"{sign}${body}"


def pct(x: float) -> str:
    return f"{_trim(x)}%"


def num(x: float) -> str:
    return _trim(x)


def years_phrase(x: float) -> str:
    return f"{num(x)} year" + ("" if x == 1 else "s")


def months_phrase(x: float) -> str:
    return f"{num(x)} month" + ("" if x == 1 else "s")


_COMPOUND_WORD = {1: "annually", 2: "semi-annually", 4: "quarterly", 12: "monthly",
                  24: "twice a month", 26: "every two weeks", 52: "weekly", 365: "daily"}
_EVERY_WORD = {1: "every year", 2: "twice a year", 4: "every quarter", 12: "every month",
               24: "twice a month", 26: "every two weeks", 52: "every week", 365: "every day"}
_PER_WORD = {1: "per year", 2: "twice a year", 4: "per quarter", 12: "per month",
             24: "twice a month", 26: "every two weeks", 52: "per week", 365: "per day"}


def compounding_word(n: int) -> str:
    return _COMPOUND_WORD.get(int(n), f"{n} times a year")


def every_word(n: int) -> str:
    return _EVERY_WORD.get(int(n), f"{n} times a year")


def per_word(n: int) -> str:
    return _PER_WORD.get(int(n), f"{n} times a year")


def join_and(items: Iterable[str]) -> str:
    items = list(items)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def money_list(values: Iterable[float]) -> str:
    return join_and(money(v) for v in values)
