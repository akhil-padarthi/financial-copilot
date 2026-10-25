"""Grounding check: every number in a reply must come from the user or from Python.

If the model writes "$1,195" when the tool returned 1199.10, or invents "2036"
by adding years in its head, the check flags it and the agent falls back to
a template explanation built purely from tool output.

Rounding is allowed: "$19,672" or "about $19.7k" for 19671.51 pass, because the
written number is a correct rounding of a real value. A wrong number does not.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Tuple

# Product names that contain digits but aren't amounts.
_IGNORE = re.compile(r"\b40[13]\s?\(\s?[kb]\s?\)|\b40[13][kb]\b|\b529\s+(?:plan|account|savings)|S&P\s?500"
                     r"|\bW-?2\b|\bW-?4\b|\b1099\b", re.IGNORECASE)
_NUM = re.compile(
    r"(?<![\w.])(-)?\$?\s?(-)?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?"
    r"(?:\s?(k|K|thousand|million|mn|m|M|billion|bn|b|B)\b)?")
_SCALE = {"k": 1e3, "thousand": 1e3, "million": 1e6, "mn": 1e6, "m": 1e6, "billion": 1e9, "bn": 1e9, "b": 1e9}


def extract_numbers(text: str) -> List[Tuple[str, float, float]]:
    """Return (token, value, resolution) for every number in `text`.

    resolution is half the place value of the last written digit, e.g.
    "1,199.10" -> 0.005, "19,700" -> 50, "20k" -> 5000.
    """
    text = _IGNORE.sub(" ", text)
    found = []
    for m in _NUM.finditer(text):
        sign = m.group(1) or m.group(2)
        whole, frac, suffix = m.group(3).replace(",", ""), m.group(4) or "", m.group(5)
        value = float(whole + frac)
        if frac:
            res = 0.5 * 10 ** -(len(frac) - 1)
        else:
            stripped = whole.rstrip("0")
            res = 0.5 * 10 ** (len(whole) - len(stripped)) if stripped else 0.5
        mult = _SCALE.get(suffix.lower(), 1) if suffix else 1
        value *= mult
        res *= mult
        if sign:
            value = -value
        found.append((m.group(0).strip(), value, res))
    return found


def _matches(value: float, res: float, allowed: float) -> bool:
    diff = abs(abs(value) - abs(allowed))
    if diff > res + 1e-9 * max(1.0, abs(allowed)):
        return False
    # A coarse rounding ("$20,000") must still be within 5% of the real value.
    return res <= 0.5 or diff <= 0.05 * abs(allowed)


def ungrounded_numbers(text: str, allowed: Iterable[float], small_int_max: int = 12) -> List[str]:
    """Numbers in `text` that don't match any allowed value.

    Small whole numbers (0..small_int_max) are always allowed: "12 months",
    "1 year", "the 2 loans" are not calculations.
    """
    pool = [float(a) for a in allowed]
    bad = []
    for token, value, res in extract_numbers(text):
        if value == int(value) and 0 <= abs(value) <= small_int_max:
            continue
        if any(_matches(value, res, a) for a in pool):
            continue
        bad.append(token)
    return bad


def numbers_in_messages(messages: Sequence[dict], roles: Sequence[str] = ("user",)) -> List[float]:
    values: List[float] = []
    for m in messages:
        if m.get("role") in roles:
            values.extend(v for _, v, _ in extract_numbers(m.get("content", "")))
    return values
