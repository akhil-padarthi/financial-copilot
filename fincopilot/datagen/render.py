"""Render numbers the many ways people actually write them.

The gold label is always the plain number; the question text varies:
10000 -> "$10,000", "10k", "$10K", "10 grand", "ten thousand dollars", ...
That variety is what teaches the model to extract values robustly.
"""
from __future__ import annotations

import random
from typing import List, Sequence, Tuple

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
         "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def int_words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else "-" + _ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return _ONES[n // 100] + " hundred" + ("" if rest == 0 else " " + int_words(rest))
    if n < 1_000_000:
        rest = n % 1000
        return int_words(n // 1000) + " thousand" + ("" if rest == 0 else " " + int_words(rest))
    rest = n % 1_000_000
    return int_words(n // 1_000_000) + " million" + ("" if rest == 0 else " " + int_words(rest))


def trim(x: float, places: int = 3) -> str:
    text = f"{x:.{places}f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-0") else "0"


def _is_int(x: float) -> bool:
    return abs(x - round(x)) < 1e-9


def _pick(rng: random.Random, options: Sequence[Tuple[str, float]]) -> str:
    texts, weights = zip(*options)
    return rng.choices(texts, weights=weights)[0]


def plain_number(x: float) -> str:
    return f"{int(round(x)):,}" if _is_int(x) else f"{x:,.2f}".rstrip("0").rstrip(".")


def money(x: float, rng: random.Random) -> str:
    if x < 0:
        return _pick(rng, [("-" + money(-x, rng), 3.0), ("negative " + money(-x, rng), 1.0)])
    if not _is_int(x):
        options = [(f"${x:,.2f}", 8.0)]
        cents = round(x * 100)
        if x < 1 and abs(cents - x * 100) < 1e-6:
            options.append((f"{cents} cents", 2.0))
        return _pick(rng, options)
    n = int(round(x))
    options: List[Tuple[str, float]] = [(f"${n:,}", 8.0), (f"{n:,} dollars", 0.8)]
    if n >= 1000:
        options.append((f"${n}", 0.8))
        options.append((f"{n:,}", 0.6))
    if 1000 <= n < 1_000_000 and n % 1000 == 0:
        k = n // 1000
        options += [(f"${k}k", 2.5), (f"{k}k", 1.2), (f"${k}K", 0.8)]
        if k <= 100:
            options.append((f"{k} grand", 0.4))
        if k <= 999:
            options.append((f"{int_words(k)} thousand dollars", 0.5))
    elif 1000 <= n < 1_000_000 and n % 100 == 0:
        k = trim(n / 1000, 1)
        options += [(f"${k}k", 1.5), (f"{k}k", 0.6)]
    if n >= 1_000_000 and n % 100_000 == 0:
        m = trim(n / 1_000_000, 1)
        options += [(f"${m} million", 2.5), (f"{m} million dollars", 1.0), (f"${m}M", 1.5), (f"{m} million", 0.8)]
        if n == 1_000_000:
            options += [("a million dollars", 1.5), ("one million dollars", 0.5)]
    if n < 1000:
        options += [(f"{n} bucks", 0.3)]
        if n <= 100:
            options.append((f"{int_words(n)} dollars", 0.3))
    return _pick(rng, options)


def percent(x: float, rng: random.Random) -> str:
    s = trim(x)
    options = [(f"{s}%", 8.0), (f"{s} percent", 2.0)]
    if _is_int(x) and 0 < x <= 20:
        options.append((f"{int_words(int(x))} percent", 0.6))
    return _pick(rng, options)


def years(x: float, rng: random.Random) -> str:
    if not _is_int(x):
        return f"{trim(x)} years"
    n = int(x)
    if n == 1:
        return _pick(rng, [("1 year", 4.0), ("one year", 2.0), ("a year", 1.0)])
    options = [(f"{n} years", 8.0), (f"{n} yrs", 0.4)]
    if n <= 30:
        options.append((f"{int_words(n)} years", 1.6))
    if n == 10:
        options.append(("a decade", 0.8))
    if n == 20:
        options.append(("two decades", 0.4))
    return _pick(rng, options)


def months(x: float, rng: random.Random) -> str:
    n = int(round(x))
    if n == 1:
        return "1 month"
    options = [(f"{n} months", 8.0)]
    if n <= 24:
        options.append((f"{int_words(n)} months", 1.5))
    return _pick(rng, options)


FREQ_ADV = {1: [("annually", 4), ("yearly", 1), ("once a year", 1)], 2: [("semi-annually", 3), ("twice a year", 2), ("semiannually", 1)],
            4: [("quarterly", 1)], 12: [("monthly", 1)], 26: [("every two weeks", 2), ("biweekly", 1)],
            52: [("weekly", 1)], 365: [("daily", 1)]}
FREQ_ADJ = {1: [("annual", 3), ("yearly", 1)], 2: [("semi-annual", 1)], 4: [("quarterly", 1)], 12: [("monthly", 1)],
            26: [("biweekly", 1)], 52: [("weekly", 1)], 365: [("daily", 1)]}
FREQ_EVERY = {1: [("every year", 2), ("each year", 1), ("a year", 2), ("per year", 1)], 2: [("twice a year", 1)],
              4: [("every quarter", 2), ("each quarter", 1), ("per quarter", 1)],
              12: [("every month", 3), ("each month", 1), ("a month", 3), ("per month", 1)],
              26: [("every two weeks", 2), ("every other week", 1)], 52: [("every week", 2), ("each week", 1), ("a week", 1)],
              365: [("every day", 1)]}


def frequency(n: int, rng: random.Random, form: str = "adv") -> str:
    table = {"adv": FREQ_ADV, "adj": FREQ_ADJ, "every": FREQ_EVERY}[form]
    return _pick(rng, table[int(n)])


def _money_formatter(values: Sequence[float], rng: random.Random):
    if all(_is_int(v) and v >= 1000 and v % 1000 == 0 for v in values) and rng.random() < 0.35:
        return lambda v: f"{int(v) // 1000}k" if rng.random() < 0.5 else f"${int(v) // 1000}k"
    if rng.random() < 0.15 and all(_is_int(v) for v in values):
        return lambda v: f"{int(v):,}"
    return lambda v: f"${int(round(v)):,}" if _is_int(v) else f"${v:,.2f}"


def money_list(values: Sequence[float], rng: random.Random) -> str:
    fmt = _money_formatter(values, rng)
    n = len(values)
    if n >= 2 and len(set(values)) == 1 and rng.random() < 0.7:
        each, count = fmt(values[0]), rng.choice([str(n), int_words(n)])
        return rng.choice([f"{each} a year for {count} years", f"{each} per year for {count} years",
                           f"{each} annually for {count} years", f"{each} each year for the next {count} years"])
    items = [fmt(v) for v in values]
    style = rng.random()
    if style < 0.40:
        return ", ".join(items[:-1]) + (", and " if rng.random() < 0.5 else " and ") + items[-1]
    if style < 0.60:
        return ", ".join(items)
    if style < 0.80:
        parts = [f"{x} in year {i}" for i, x in enumerate(items, 1)]
        return ", ".join(parts[:-1]) + " and " + parts[-1]
    return ", ".join(items[:-1]) + " and " + items[-1] + f" over the next {int_words(n)} years"


def price_list(values: Sequence[float], rng: random.Random) -> str:
    items = [f"${int(round(v)):,}" if _is_int(v) else f"${v:,.2f}" for v in values]
    style = rng.random()
    if style < 0.5:
        return ", ".join(items[:-1]) + " and " + items[-1]
    if style < 0.8:
        return ", ".join(items)
    return ", then ".join(items)


def render_value(kind: str, value, rng: random.Random, modifier: str = "") -> str:
    """Render one slot value. modifier: n (bare number), money, adv/adj/every (frequencies)."""
    if modifier == "n":
        return plain_number(value) if kind != "percent" else trim(value)
    if modifier == "money":
        return money(value, rng)
    if kind == "money":
        return money(value, rng)
    if kind == "percent":
        return percent(value, rng)
    if kind == "years":
        return years(value, rng)
    if kind in ("months", "count"):
        return months(value, rng)
    if kind == "freq":
        return frequency(value, rng, modifier or "adv")
    if kind == "money_list":
        return money_list(value, rng)
    if kind == "price_list":
        return price_list(value, rng)
    return plain_number(value)
