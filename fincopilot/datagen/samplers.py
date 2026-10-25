"""Realistic random inputs for every tool.

Each sampler returns values for *all* fields, including both sides of
alternatives (e.g. years and term_months); the template decides which
ones actually appear in the question, and only those become the gold label.
"""
from __future__ import annotations

import math
import random
from typing import Callable, Dict, FrozenSet

Values = Dict[str, object]


def nice(rng: random.Random, lo: float, hi: float, sig: int = None) -> float:
    """Log-uniform value rounded to 1-4 significant figures (people use round numbers)."""
    x = math.exp(rng.uniform(math.log(lo), math.log(hi)))
    if sig is None:
        sig = rng.choices([1, 2, 3, 4], weights=[0.15, 0.45, 0.3, 0.1])[0]
    mag = 10 ** (math.floor(math.log10(x)) - sig + 1)
    v = round(x / mag) * mag
    v = float(int(round(v))) if mag >= 1 else round(v, 2)
    return v if v > 0 else round(lo, 2)


def price(rng: random.Random, lo: float, hi: float) -> float:
    """Share prices and unit prices: often with cents."""
    if rng.random() < 0.5:
        return round(math.exp(rng.uniform(math.log(lo), math.log(hi))), 2)
    return nice(rng, lo, hi, sig=rng.choice([2, 3]))


def rate(rng: random.Random, lo: float, hi: float, fine: bool = False) -> float:
    steps = [(1, 0.35), (0.5, 0.25), (0.25, 0.2), (0.01, 0.15), (0.125, 0.05)] if not fine else \
        [(0.125, 0.4), (0.25, 0.3), (0.01, 0.2), (0.5, 0.1)]
    step = rng.choices([s for s, _ in steps], weights=[w for _, w in steps])[0]
    v = round(round(rng.uniform(lo, hi) / step) * step, 3)
    return max(v, lo) if lo > 0 else max(v, 0.0)


def years(rng: random.Random, lo: int, hi: int) -> float:
    return float(rng.randint(lo, hi))


def _flows(rng: random.Random, initial: float, lo_n: int = 2, hi_n: int = 7, scale=(0.15, 0.6)) -> list:
    n = rng.randint(lo_n, hi_n)
    base = initial * rng.uniform(*scale)
    if rng.random() < 0.3:
        v = nice(rng, base * 0.8, base * 1.2, sig=2)
        return [v] * n
    return [nice(rng, base * 0.6, base * 1.4, sig=2) for _ in range(n)]


def s_compound_interest(rng, slots):
    return {"principal": nice(rng, 500, 2e6), "annual_rate": rate(rng, 0.5, 12),
            "years": years(rng, 1, 40) if rng.random() < 0.92 else rng.choice([1.5, 2.5, 7.5, 0.5]),
            "compounds_per_year": rng.choices([1, 2, 4, 12, 365], weights=[2, 1, 2, 4, 2])[0]}


def s_simple_interest(rng, slots):
    return {"principal": nice(rng, 100, 5e5), "annual_rate": rate(rng, 1, 15), "years": years(rng, 1, 10)}


_CONTRIB = {12: (50, 3000), 1: (1000, 25000), 26: (25, 1500), 52: (10, 500), 4: (250, 6000)}


def s_future_value_with_contributions(rng, slots):
    m = rng.choices([12, 1, 26, 52, 4], weights=[6, 2, 1, 0.5, 1])[0] if "contributions_per_year" in slots else 12
    return {"contribution": nice(rng, *_CONTRIB[m]), "annual_rate": rate(rng, 2, 12), "years": years(rng, 1, 45),
            "initial_balance": nice(rng, 1000, 3e5), "contributions_per_year": m}


def s_present_value(rng, slots):
    return {"future_value": nice(rng, 1000, 2e6), "annual_rate": rate(rng, 1, 12), "years": years(rng, 1, 40),
            "compounds_per_year": rng.choice([1, 2, 4, 12, 365])}


_ANNUITY = {12: (100, 10000), 1: (1000, 200000), 4: (500, 30000), 52: (50, 2000), 26: (100, 4000)}


def s_present_value_of_annuity(rng, slots):
    m = rng.choice([12, 1, 4, 52, 26]) if "payments_per_year" in slots else 12
    return {"payment": nice(rng, *_ANNUITY[m]), "annual_rate": rate(rng, 1, 10), "years": years(rng, 1, 40),
            "payments_per_year": m}


def s_continuous_compounding(rng, slots):
    return {"principal": nice(rng, 100, 1e6), "annual_rate": rate(rng, 0.5, 15), "years": years(rng, 1, 40)}


def s_effective_annual_rate(rng, slots):
    return {"nominal_rate": rate(rng, 0.5, 30), "compounds_per_year": rng.choice([2, 4, 12, 12, 52, 365, 365])}


def s_rule_of_72(rng, slots):
    return {"annual_rate": rate(rng, 1, 24)}


def s_years_to_goal(rng, slots):
    current = nice(rng, 1000, 1e6)
    return {"current_amount": current, "target_amount": nice(rng, current * 1.3, current * 20, sig=rng.choice([1, 2])),
            "annual_rate": rate(rng, 1, 15), "compounds_per_year": rng.choice([1, 4, 12, 365])}


def s_inflation_adjusted_value(rng, slots):
    return {"amount": nice(rng, 10, 2e6), "inflation_rate": rate(rng, 1, 8), "years": years(rng, 1, 40)}


def s_loan_payment(rng, slots):
    return {"principal": nice(rng, 1000, 8e5), "annual_rate": 0.0 if rng.random() < 0.02 else rate(rng, 1.9, 25),
            "years": float(rng.choices([1, 2, 3, 4, 5, 6, 7, 10, 15, 20, 25, 30], weights=[1, 2, 3, 3, 5, 2, 2, 3, 2, 2, 1, 2])[0]),
            "term_months": rng.choice([12, 24, 36, 48, 60, 60, 72, 72, 84]),
            "payments_per_year": rng.choices([12, 26, 52, 4], weights=[6, 2, 1, 1])[0] if "payments_per_year" in slots else 12}


def s_mortgage_payment(rng, slots):
    home = nice(rng, 80_000, 3e6, sig=rng.choice([2, 3]))
    return {"home_price": home, "annual_rate": rate(rng, 2.5, 9, fine=True),
            "years": float(rng.choice([30, 30, 30, 15, 15, 20, 25, 10])),
            "down_payment": nice(rng, home * 0.03, home * 0.4, sig=rng.choice([1, 2])),
            "down_payment_percent": float(rng.choice([3, 3.5, 5, 10, 10, 15, 20, 20, 20, 25])),
            "annual_property_tax": nice(rng, home * 0.004, home * 0.025, sig=2),
            "annual_insurance": nice(rng, 500, 6000, sig=2), "monthly_hoa": nice(rng, 40, 900, sig=2)}


def s_remaining_loan_balance(rng, slots):
    term = rng.choice([5, 7, 10, 15, 20, 30, 30])
    return {"principal": nice(rng, 5000, 1.5e6), "annual_rate": rate(rng, 1, 12), "years": float(term),
            "elapsed_years": float(rng.randint(1, term - 1)), "elapsed_months": rng.randint(1, term * 12 - 1)}


def s_loan_payoff_time(rng, slots):
    balance = nice(rng, 500, 80_000)
    r = rate(rng, 0, 30)
    interest = balance * r / 1200
    lo = max(interest * 1.25, balance / 300, 15)
    hi = max(balance / 3, lo * 2)
    payment = nice(rng, lo, hi, sig=rng.choice([1, 2, 2, 3]))
    while payment <= interest * 1.05:
        payment = round(payment * 1.5 + 10)
    return {"balance": balance, "annual_rate": r, "monthly_payment": payment}


def s_extra_payment_savings(rng, slots):
    principal = nice(rng, 5000, 1e6)
    return {"principal": principal, "annual_rate": rate(rng, 2, 12), "years": float(rng.choice([3, 5, 7, 10, 15, 20, 30, 30])),
            "extra_monthly_payment": nice(rng, 20, min(3000, principal / 20), sig=rng.choice([1, 2]))}


def s_max_affordable_loan(rng, slots):
    return {"monthly_payment": nice(rng, 100, 10_000), "annual_rate": rate(rng, 1, 12),
            "years": float(rng.choice([3, 4, 5, 6, 10, 15, 20, 30, 30])), "term_months": rng.choice([36, 48, 60, 72, 84])}


def s_refinance_breakeven(rng, slots):
    current = rate(rng, 3, 9, fine=True)
    new = current - rate(rng, 0.25, 2.5, fine=True) if rng.random() < 0.85 else current + rate(rng, 0.25, 1, fine=True)
    return {"current_balance": nice(rng, 50_000, 1.5e6), "current_rate": current, "remaining_years": years(rng, 5, 29),
            "new_rate": round(max(new, 1.5), 3), "new_term_years": float(rng.choice([10, 15, 20, 30, 30])),
            "closing_costs": nice(rng, 1500, 15_000, sig=2)}


def s_roi(rng, slots):
    initial = nice(rng, 100, 1e6)
    return {"initial_investment": initial, "final_value": nice(rng, initial * 0.3, initial * 4, sig=rng.choice([2, 3, 4]))}


def s_cagr(rng, slots):
    begin, y = nice(rng, 1000, 5e7), rng.randint(1, 30)
    end = begin * (1 + rng.uniform(-0.1, 0.3)) ** y
    return {"beginning_value": begin, "ending_value": nice(rng, end * 0.98, end * 1.02, sig=rng.choice([2, 3])), "years": float(y)}


def s_npv(rng, slots):
    initial = nice(rng, 1000, 5e6, sig=rng.choice([1, 2]))
    return {"discount_rate": rate(rng, 3, 15), "initial_investment": initial, "cash_flows": _flows(rng, initial)}


def s_irr(rng, slots):
    initial = nice(rng, 1000, 5e6, sig=rng.choice([1, 2]))
    return {"initial_investment": initial, "cash_flows": _flows(rng, initial, scale=(0.2, 0.6))}


def s_payback_period(rng, slots):
    initial = nice(rng, 1000, 5e6, sig=rng.choice([1, 2]))
    flows = _flows(rng, initial, 3, 6, scale=(0.25, 0.6))
    while sum(flows) < initial * 1.05:
        flows.append(nice(rng, initial * 0.2, initial * 0.5, sig=2))
    return {"initial_investment": initial, "cash_flows": flows,
            "annual_cash_flow": nice(rng, initial * 0.05, initial * 0.5, sig=2)}


def s_dividend_yield(rng, slots):
    p = price(rng, 5, 800)
    m = rng.choice([1, 4, 4, 12, 2]) if "payments_per_year" in slots else 1
    per = max(round(p * rng.uniform(0.005, 0.08) / m, 2), 0.01)
    return {"dividend_per_share": per, "share_price": p, "payments_per_year": m}


def s_pe_ratio(rng, slots):
    p = price(rng, 5, 1500)
    eps = max(round(p / rng.uniform(5, 60), 2), 0.01)
    return {"share_price": p, "earnings_per_share": -eps if rng.random() < 0.04 else eps}


def s_sharpe_ratio(rng, slots):
    return {"portfolio_return": rate(rng, -5, 25), "risk_free_rate": rate(rng, 0.5, 6), "volatility": rate(rng, 4, 40)}


def s_average_cost_per_share(rng, slots):
    base = price(rng, 5, 600)
    cents = rng.random() < 0.4
    prices = [round(base * rng.uniform(0.7, 1.3), 2 if cents else 0) or 1.0 for _ in range(rng.randint(2, 7))]
    return {"amount_per_purchase": nice(rng, 25, 5000, sig=rng.choice([1, 2])), "share_prices": [float(p) for p in prices]}


def s_investment_fee_impact(rng, slots):
    return {"initial_investment": nice(rng, 1000, 5e6), "annual_return": rate(rng, 3, 12),
            "annual_fee": rng.choice([0.03, 0.05, 0.1, 0.2, 0.25, 0.5, 0.75, 0.85, 1.0, 1.0, 1.25, 1.5, 2.0]),
            "years": years(rng, 5, 45)}


def s_savings_goal_monthly(rng, slots):
    target = nice(rng, 1000, 5e6, sig=rng.choice([1, 2]))
    return {"target_amount": target, "years": years(rng, 1, 40), "annual_rate": 0.0 if rng.random() < 0.08 else rate(rng, 0.5, 10),
            "current_savings": nice(rng, 300, max(target * 0.3, 400), sig=rng.choice([1, 2]))}


def s_retirement_nest_egg(rng, slots):
    return {"annual_expenses": nice(rng, 20_000, 300_000, sig=rng.choice([1, 2])),
            "monthly_expenses": nice(rng, 1500, 25_000, sig=rng.choice([1, 2])),
            "withdrawal_rate": rng.choice([3.0, 3.25, 3.5, 4.0, 4.0, 4.5, 5.0])}


def s_retirement_drawdown(rng, slots):
    balance = nice(rng, 50_000, 5e6, sig=rng.choice([1, 2]))
    annual = nice(rng, balance * 0.03, balance * 0.15, sig=2)
    return {"balance": balance, "annual_withdrawal": annual, "monthly_withdrawal": nice(rng, annual / 14, annual / 10, sig=2),
            "annual_return": rate(rng, 0, 9)}


def s_emergency_fund(rng, slots):
    monthly = nice(rng, 800, 15_000, sig=rng.choice([2, 3]))
    return {"monthly_expenses": monthly, "months": float(rng.choice([3, 4, 6, 6, 6, 9, 12])),
            "current_savings": nice(rng, 300, monthly * 8, sig=2)}


def s_debt_to_income(rng, slots):
    return {"monthly_debt_payments": nice(rng, 100, 8000, sig=rng.choice([2, 3])),
            "gross_monthly_income": nice(rng, 2000, 40_000, sig=rng.choice([2, 3])),
            "gross_annual_income": nice(rng, 25_000, 500_000, sig=rng.choice([2, 3]))}


def s_budget_50_30_20(rng, slots):
    return {"monthly_income": nice(rng, 1500, 25_000, sig=rng.choice([2, 3])),
            "annual_income": nice(rng, 20_000, 300_000, sig=rng.choice([2, 3]))}


def s_break_even_units(rng, slots):
    p = price(rng, 2, 5000)
    return {"fixed_costs": nice(rng, max(500, p * 40), max(p * 5000, 5000)), "price_per_unit": p,
            "variable_cost_per_unit": max(round(p * rng.uniform(0.1, 0.85), 2 if rng.random() < 0.5 else 0), 0.5)}


def s_profit_margin(rng, slots):
    revenue = nice(rng, 1000, 1e9)
    return {"revenue": revenue, "cost": nice(rng, revenue * 0.4, revenue * 1.12, sig=rng.choice([2, 3]))}


def s_markup_percentage(rng, slots):
    cost = price(rng, 1, 5000)
    return {"cost": cost, "selling_price": price(rng, cost * 1.05, cost * 4)}


def s_straight_line_depreciation(rng, slots):
    cost = nice(rng, 1000, 5e6, sig=rng.choice([1, 2, 3]))
    salvage = 0.0 if rng.random() < 0.2 else nice(rng, cost * 0.02, cost * 0.3, sig=rng.choice([1, 2]))
    return {"asset_cost": cost, "salvage_value": salvage, "useful_life_years": float(rng.choice([3, 4, 5, 5, 7, 8, 10, 15, 20]))}


def s_percentage_change(rng, slots):
    old = price(rng, 1, 2000) if rng.random() < 0.5 else nice(rng, 10, 1e6)
    new = price(rng, old * 0.4, old * 2) if not float(old).is_integer() else nice(rng, old * 0.4, old * 2, sig=rng.choice([2, 3]))
    return {"old_value": old, "new_value": new if new != old else round(new * 1.1, 2)}


SAMPLERS_WITH_SLOTS: Dict[str, Callable[[random.Random, FrozenSet[str]], Values]] = {
    name[2:]: fn for name, fn in globals().items() if name.startswith("s_") and callable(fn)}

# Simpler signature (rng only) for tests and quick experiments.
SAMPLERS: Dict[str, Callable[[random.Random], Values]] = {
    name: (lambda fn: (lambda rng: fn(rng, frozenset({"contributions_per_year", "payments_per_year"}))))(fn)
    for name, fn in SAMPLERS_WITH_SLOTS.items()}


# Inputs that are valid but impossible, to teach graceful "can't compute" answers.
def d_loan_payoff_time(rng, slots):
    v = s_loan_payoff_time(rng, slots)
    v["annual_rate"] = rate(rng, 12, 30)
    interest = v["balance"] * v["annual_rate"] / 1200
    v["monthly_payment"] = max(nice(rng, interest * 0.3, interest * 0.95, sig=2), 1.0)
    return v


def d_break_even_units(rng, slots):
    v = s_break_even_units(rng, slots)
    v["variable_cost_per_unit"] = round(v["price_per_unit"] * rng.uniform(1.0, 1.4), 2)
    return v


def d_payback_period(rng, slots):
    initial = nice(rng, 5000, 1e6, sig=2)
    return {"initial_investment": initial, "cash_flows": [nice(rng, initial * 0.05, initial * 0.2, sig=2) for _ in range(3)],
            "annual_cash_flow": None}


def d_years_to_goal(rng, slots):
    v = s_years_to_goal(rng, slots)
    v["target_amount"] = nice(rng, v["current_amount"] * 0.3, v["current_amount"] * 0.95, sig=2)
    return v


DOMAIN_SAMPLERS = {"loan_payoff_time": d_loan_payoff_time, "break_even_units": d_break_even_units,
                   "payback_period": d_payback_period, "years_to_goal": d_years_to_goal}
