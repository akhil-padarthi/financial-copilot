"""Small-business math: break-even, margins, depreciation, percent change."""
from __future__ import annotations

import math

from .base import Number, ToolArgs, ToolDomainError, arg, register
from .fmt import money, num, pct, years_phrase

CATEGORY = "Business"


# --- break_even_units -------------------------------------------------------------

class BreakEvenArgs(ToolArgs):
    fixed_costs: Number = arg("money", "your fixed costs", "Fixed costs for the period (rent, salaries...), in dollars.", gt=0, le=1e12)
    price_per_unit: Number = arg("money", "the selling price per unit", "Selling price of one unit, in dollars.", gt=0, le=1e9)
    variable_cost_per_unit: Number = arg("money", "the variable cost per unit", "Cost to produce one unit, in dollars.", ge=0, le=1e9)


def _explain_break_even(a, r):
    text = (f"Each unit contributes {money(r['contribution_margin'])} toward fixed costs ({money(a.price_per_unit)} price "
            f"minus {money(a.variable_cost_per_unit)} variable cost). ")
    units = "unit" if r["units_to_sell"] == 1 else "units"
    if abs(r["break_even_units"] - r["units_to_sell"]) < 0.005:
        text += (f"Covering {money(a.fixed_costs)} of fixed costs takes {num(r['units_to_sell'])} {units}, about "
                 f"{money(r['break_even_revenue'])} in revenue.")
    else:
        text += (f"Covering {money(a.fixed_costs)} of fixed costs takes {num(r['break_even_units'])} units, so you'd "
                 f"need to sell {num(r['units_to_sell'])} {units} (about {money(r['break_even_revenue'])} in revenue) to break even.")
    return text


@register("break_even_units", category=CATEGORY, args=BreakEvenArgs, explain=_explain_break_even,
          task="work out your break-even point",
          description="Units (and revenue) needed to cover fixed costs given price and variable cost per unit.")
def break_even_units(fixed_costs, price_per_unit, variable_cost_per_unit):
    margin = price_per_unit - variable_cost_per_unit
    if margin <= 0:
        raise ToolDomainError(f"Each unit sells for {money(price_per_unit)} but costs {money(variable_cost_per_unit)} "
                              "to make, so you lose money on every sale and can never break even.")
    units = fixed_costs / margin
    return {"break_even_units": units, "units_to_sell": math.ceil(round(units, 6)),
            "contribution_margin": margin, "break_even_revenue": units * price_per_unit}


# --- profit_margin -----------------------------------------------------------------

class ProfitMarginArgs(ToolArgs):
    revenue: Number = arg("money", "your revenue", "Total revenue / sales, in dollars.", gt=0, le=1e14)
    cost: Number = arg("money", "your total costs", "Total costs / expenses, in dollars.", ge=0, le=1e14)


def _explain_profit_margin(a, r):
    if r["profit"] >= 0:
        return (f"Revenue of {money(a.revenue)} minus costs of {money(a.cost)} leaves a profit of {money(r['profit'])}, "
                f"a profit margin of {pct(r['profit_margin_percent'])}.")
    return (f"Revenue of {money(a.revenue)} minus costs of {money(a.cost)} is a loss of {money(-r['profit'])}, "
            f"a profit margin of {pct(r['profit_margin_percent'])}.")


@register("profit_margin", category=CATEGORY, args=ProfitMarginArgs, explain=_explain_profit_margin,
          task="work out your profit margin",
          description="Profit and profit margin from revenue and total costs.")
def profit_margin(revenue, cost):
    profit = revenue - cost
    return {"profit": profit, "profit_margin_percent": profit / revenue * 100}


# --- markup_percentage --------------------------------------------------------------

class MarkupArgs(ToolArgs):
    cost: Number = arg("money", "what the item costs you", "Unit cost, in dollars.", gt=0, le=1e10)
    selling_price: Number = arg("money", "the selling price", "Unit selling price, in dollars.", gt=0, le=1e10)


def _explain_markup(a, r):
    verb = "earns" if r["profit_per_unit"] >= 0 else "loses"
    return (f"Buying at {money(a.cost)} and selling at {money(a.selling_price)} {verb} {money(abs(r['profit_per_unit']))} "
            f"per unit. That's a markup of {pct(r['markup_percent'])} on cost and a profit margin of "
            f"{pct(r['margin_percent'])} of the selling price.")


@register("markup_percentage", category=CATEGORY, args=MarkupArgs, explain=_explain_markup,
          task="work out your markup",
          description="Markup on cost and margin on price from unit cost and selling price.")
def markup_percentage(cost, selling_price):
    profit = selling_price - cost
    return {"profit_per_unit": profit, "markup_percent": profit / cost * 100, "margin_percent": profit / selling_price * 100}


# --- straight_line_depreciation --------------------------------------------------------

class DepreciationArgs(ToolArgs):
    asset_cost: Number = arg("money", "the asset's cost", "Purchase cost of the asset, in dollars.", gt=0, le=1e12)
    useful_life_years: Number = arg("years", "the useful life", "Useful life in years.", gt=0, le=100)
    salvage_value: Number = arg("money", "the salvage value", "Value at the end of its useful life, in dollars.", 0, ge=0, le=1e12)


def _explain_depreciation(a, r):
    basis = (f"{money(a.asset_cost)} cost minus {money(a.salvage_value)} salvage value" if a.salvage_value
             else f"{money(a.asset_cost)} cost with no salvage value")
    return (f"Spreading the {money(r['total_depreciation'])} depreciable amount ({basis}) evenly over "
            f"{years_phrase(a.useful_life_years)}, straight-line depreciation is {money(r['annual_depreciation'])} per "
            f"year ({money(r['monthly_depreciation'])} per month).")


@register("straight_line_depreciation", category=CATEGORY, args=DepreciationArgs, explain=_explain_depreciation,
          task="work out the depreciation",
          description="Yearly straight-line depreciation from cost, salvage value and useful life.")
def straight_line_depreciation(asset_cost, salvage_value, useful_life_years):
    if salvage_value > asset_cost:
        raise ToolDomainError("The salvage value can't be higher than the asset's cost.")
    total = asset_cost - salvage_value
    annual = total / useful_life_years
    return {"annual_depreciation": annual, "monthly_depreciation": annual / 12, "total_depreciation": total}


# --- percentage_change ------------------------------------------------------------------------

class PercentChangeArgs(ToolArgs):
    old_value: Number = arg("number", "the starting value", "Original value.", ge=-1e15, le=1e15)
    new_value: Number = arg("number", "the new value", "New value.", ge=-1e15, le=1e15)


def _explain_percent_change(a, r):
    change = r["change"]
    if change == 0:
        return f"Going from {num(a.old_value)} to {num(a.new_value)} is no change (0%)."
    direction = "an increase" if change > 0 else "a decrease"
    return (f"Going from {num(a.old_value)} to {num(a.new_value)} is {direction} of {num(abs(change))}, or "
            f"{pct(abs(r['percent_change']))}.")


@register("percentage_change", category=CATEGORY, args=PercentChangeArgs, explain=_explain_percent_change,
          task="work out the percentage change",
          description="Percent increase or decrease between an old and a new value (prices, salary, rent, sales, users).")
def percentage_change(old_value, new_value):
    if old_value == 0:
        raise ToolDomainError("Percentage change is undefined when the starting value is zero.")
    change = new_value - old_value
    return {"change": change, "percent_change": change / abs(old_value) * 100}
