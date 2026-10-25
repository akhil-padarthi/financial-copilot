"""Time value of money: how money grows, and what future money is worth today."""
from __future__ import annotations

import math

from .base import Integer, Number, ToolArgs, ToolDomainError, arg, register
from .fmt import compounding_word, every_word, money, num, pct, years_phrase

CATEGORY = "Time value of money"

FREQ_DESC = "Times per year (1=annually, 2=semi-annually, 4=quarterly, 12=monthly, 26=every two weeks, 52=weekly, 365=daily)."


def _rate(label="the annual interest rate", desc="Annual interest rate in percent (7 means 7%).", **kw):
    kw.setdefault("ge", 0)
    kw.setdefault("le", 100)
    return arg("percent", label, desc, **kw)


def _years(label="the number of years", desc="Number of years.", **kw):
    kw.setdefault("gt", 0)
    kw.setdefault("le", 100)
    return arg("years", label, desc, **kw)


# --- compound_interest ------------------------------------------------------

class CompoundInterestArgs(ToolArgs):
    principal: Number = arg("money", "the starting amount", "One-time amount invested today, in dollars.", gt=0, le=1e12)
    annual_rate: Number = _rate()
    years: Number = _years()
    compounds_per_year: Integer = arg("freq", "how often interest compounds", FREQ_DESC, 1, ge=1, le=365)


def _explain_compound_interest(a, r):
    return (f"Investing {money(a.principal)} at {pct(a.annual_rate)} a year, compounded "
            f"{compounding_word(a.compounds_per_year)}, grows to {money(r['future_value'])} after "
            f"{years_phrase(a.years)}. That's {money(r['interest_earned'])} of interest on top of "
            f"your original {money(a.principal)}.")


@register("compound_interest", category=CATEGORY, args=CompoundInterestArgs,
          explain=_explain_compound_interest, task="work out how much a lump sum will grow to",
          description="Future value of a one-time lump sum growing with compound interest.")
def compound_interest(principal, annual_rate, years, compounds_per_year=1):
    n = compounds_per_year
    fv = principal * (1 + annual_rate / 100 / n) ** (n * years)
    return {"future_value": fv, "interest_earned": fv - principal}


# --- simple_interest --------------------------------------------------------

class SimpleInterestArgs(ToolArgs):
    principal: Number = arg("money", "the principal amount", "Amount lent, borrowed or deposited, in dollars.", gt=0, le=1e12)
    annual_rate: Number = _rate()
    years: Number = _years()


def _explain_simple_interest(a, r):
    return (f"At {pct(a.annual_rate)} simple interest, {money(a.principal)} earns {money(r['interest'])} over "
            f"{years_phrase(a.years)}, for a total of {money(r['total_amount'])}. Simple interest is charged "
            f"only on the original amount, never on interest already earned.")


@register("simple_interest", category=CATEGORY, args=SimpleInterestArgs, explain=_explain_simple_interest,
          task="work out the simple interest",
          description="Interest charged only on the original principal (no compounding).")
def simple_interest(principal, annual_rate, years):
    interest = principal * annual_rate / 100 * years
    return {"interest": interest, "total_amount": principal + interest}


# --- future_value_with_contributions ---------------------------------------

class FutureValueContributionsArgs(ToolArgs):
    contribution: Number = arg("money", "how much you'll add each time", "Amount added every period, in dollars.", gt=0, le=1e9)
    annual_rate: Number = _rate("the expected annual return", "Expected annual return in percent.")
    years: Number = _years()
    initial_balance: Number = arg("money", "your starting balance", "Amount already invested at the start, in dollars.", 0, ge=0, le=1e12)
    contributions_per_year: Integer = arg("freq", "how often you contribute",
                                          "Contributions per year; growth compounds at the same frequency. " + FREQ_DESC,
                                          12, ge=1, le=365)


def _explain_fv_contributions(a, r):
    start = f", starting from {money(a.initial_balance)}," if a.initial_balance else ""
    yours = "money you put in (including your starting balance)" if a.initial_balance else "money you put in"
    return (f"Contributing {money(a.contribution)} {every_word(a.contributions_per_year)} for "
            f"{years_phrase(a.years)}{start} at a {pct(a.annual_rate)} annual return grows to "
            f"{money(r['future_value'])} after {r['number_of_contributions']} contributions. Of that, "
            f"{money(r['total_contributed'])} is {yours} and {money(r['investment_growth'])} is investment growth.")


@register("future_value_with_contributions", category=CATEGORY, args=FutureValueContributionsArgs,
          explain=_explain_fv_contributions, task="project how much your regular savings will grow to",
          description="Future value of regular contributions (plus an optional starting balance) growing at a fixed return.")
def future_value_with_contributions(contribution, annual_rate, years, initial_balance=0, contributions_per_year=12):
    m = contributions_per_year
    i = annual_rate / 100 / m
    n = round(m * years)
    if n < 1:
        raise ToolDomainError("That time period is too short for even one contribution.")
    growth = (1 + i) ** n
    fv_contributions = contribution * ((growth - 1) / i if i else n)
    fv = initial_balance * growth + fv_contributions
    total_in = initial_balance + contribution * n
    return {"future_value": fv, "total_contributed": total_in, "investment_growth": fv - total_in,
            "number_of_contributions": n}


# --- present_value ---------------------------------------------------------

class PresentValueArgs(ToolArgs):
    future_value: Number = arg("money", "the future amount", "Amount needed or received in the future, in dollars.", gt=0, le=1e12)
    annual_rate: Number = _rate("the interest or discount rate", "Annual interest/discount rate in percent.")
    years: Number = _years("how many years away it is", "Years until the money is needed or received.")
    compounds_per_year: Integer = arg("freq", "how often interest compounds", FREQ_DESC, 1, ge=1, le=365)


def _explain_present_value(a, r):
    return (f"To have {money(a.future_value)} in {years_phrase(a.years)}, you'd need to invest "
            f"{money(r['present_value'])} today at {pct(a.annual_rate)} a year (compounded "
            f"{compounding_word(a.compounds_per_year)}). Growth makes up the other {money(r['discount'])}.")


@register("present_value", category=CATEGORY, args=PresentValueArgs, explain=_explain_present_value,
          task="work out what that future amount is worth today",
          description="Lump sum needed today to reach a future amount (or today's value of a future payment).")
def present_value(future_value, annual_rate, years, compounds_per_year=1):
    n = compounds_per_year
    pv = future_value / (1 + annual_rate / 100 / n) ** (n * years)
    return {"present_value": pv, "discount": future_value - pv}


# --- present_value_of_annuity ----------------------------------------------

class PresentValueAnnuityArgs(ToolArgs):
    payment: Number = arg("money", "the payment amount", "Amount received each period, in dollars.", gt=0, le=1e10)
    annual_rate: Number = _rate("the discount rate", "Annual discount/interest rate in percent.")
    years: Number = _years("how many years the payments last", "Number of years payments continue.")
    payments_per_year: Integer = arg("freq", "how often payments arrive", "Payments per year. " + FREQ_DESC, 12, ge=1, le=365)


def _explain_pv_annuity(a, r):
    return (f"Receiving {money(a.payment)} {every_word(a.payments_per_year)} for {years_phrase(a.years)} "
            f"is worth {money(r['present_value'])} today at a {pct(a.annual_rate)} discount rate. You'd collect "
            f"{money(r['total_payments'])} across {r['number_of_payments']} payments, but money received later is "
            f"worth less than money in hand today.")


@register("present_value_of_annuity", category=CATEGORY, args=PresentValueAnnuityArgs, explain=_explain_pv_annuity,
          task="work out what that stream of payments is worth today",
          description="Today's value of a series of equal payments (pension, annuity, settlement, lottery payout).")
def present_value_of_annuity(payment, annual_rate, years, payments_per_year=12):
    m = payments_per_year
    i = annual_rate / 100 / m
    n = round(m * years)
    if n < 1:
        raise ToolDomainError("That time period is too short for even one payment.")
    pv = payment * ((1 - (1 + i) ** -n) / i if i else n)
    return {"present_value": pv, "total_payments": payment * n, "number_of_payments": n}


# --- continuous_compounding ------------------------------------------------

class ContinuousCompoundingArgs(ToolArgs):
    principal: Number = arg("money", "the starting amount", "Amount invested today, in dollars.", gt=0, le=1e12)
    annual_rate: Number = _rate()
    years: Number = _years()


def _explain_continuous(a, r):
    return (f"With continuous compounding at {pct(a.annual_rate)}, {money(a.principal)} grows to "
            f"{money(r['future_value'])} after {years_phrase(a.years)}, earning {money(r['interest_earned'])} in interest.")


@register("continuous_compounding", category=CATEGORY, args=ContinuousCompoundingArgs, explain=_explain_continuous,
          task="work out the continuously compounded value",
          description="Future value with continuously compounded interest (A = P·e^(rt)).")
def continuous_compounding(principal, annual_rate, years):
    fv = principal * math.exp(annual_rate / 100 * years)
    return {"future_value": fv, "interest_earned": fv - principal}


# --- effective_annual_rate -------------------------------------------------

class EffectiveAnnualRateArgs(ToolArgs):
    nominal_rate: Number = _rate("the stated (nominal) rate", "Stated annual rate / APR in percent.")
    compounds_per_year: Integer = arg("freq", "how often interest compounds", FREQ_DESC, ge=1, le=365)


def _explain_ear(a, r):
    base = (f"A nominal rate of {pct(a.nominal_rate)} compounded {compounding_word(a.compounds_per_year)} "
            f"works out to an effective annual rate (APY) of {pct(r['effective_annual_rate'])}")
    if r["difference"] > 0:
        return (base + f", which is {num(r['difference'])} percentage points higher than the stated rate "
                "because interest earns interest during the year.")
    return base + ", the same as the stated rate."


@register("effective_annual_rate", category=CATEGORY, args=EffectiveAnnualRateArgs, explain=_explain_ear,
          task="convert that rate to an effective annual rate",
          description="Convert a nominal rate/APR with a compounding frequency into the effective annual rate (APY).")
def effective_annual_rate(nominal_rate, compounds_per_year):
    n = compounds_per_year
    ear = ((1 + nominal_rate / 100 / n) ** n - 1) * 100
    return {"effective_annual_rate": ear, "difference": ear - nominal_rate}


# --- rule_of_72 -------------------------------------------------------------

class RuleOf72Args(ToolArgs):
    annual_rate: Number = _rate("the annual growth rate", "Annual growth/interest rate in percent.", gt=0, ge=None)


def _explain_rule_of_72(a, r):
    return (f"By the Rule of 72, money growing at {pct(a.annual_rate)} a year doubles in about "
            f"{num(r['estimated_years'])} years (72 ÷ {num(a.annual_rate)}). The exact doubling time with annual "
            f"compounding is {num(r['exact_years'])} years.")


@register("rule_of_72", category=CATEGORY, args=RuleOf72Args, explain=_explain_rule_of_72, constants=(72,),
          task="estimate how long it takes your money to double",
          description="Years for money to double at a given rate (Rule of 72 estimate plus the exact answer).")
def rule_of_72(annual_rate):
    return {"estimated_years": 72 / annual_rate,
            "exact_years": math.log(2) / math.log(1 + annual_rate / 100)}


# --- years_to_goal -----------------------------------------------------------

class YearsToGoalArgs(ToolArgs):
    current_amount: Number = arg("money", "how much you have now", "Amount invested today, in dollars.", gt=0, le=1e12)
    target_amount: Number = arg("money", "your target amount", "Goal amount, in dollars.", gt=0, le=1e13)
    annual_rate: Number = _rate("the annual growth rate", "Annual growth/interest rate in percent.", gt=0, ge=None)
    compounds_per_year: Integer = arg("freq", "how often interest compounds", FREQ_DESC, 1, ge=1, le=365)


def _explain_years_to_goal(a, r):
    return (f"At {pct(a.annual_rate)} a year (compounded {compounding_word(a.compounds_per_year)}), "
            f"{money(a.current_amount)} grows to {money(a.target_amount)} in about {num(r['years'])} years, "
            f"or roughly {r['months']} months, without any further contributions.")


@register("years_to_goal", category=CATEGORY, args=YearsToGoalArgs, explain=_explain_years_to_goal,
          task="work out how long it takes to reach your goal",
          description="How long a lump sum takes to grow to a target amount (no further contributions).")
def years_to_goal(current_amount, target_amount, annual_rate, compounds_per_year=1):
    if target_amount <= current_amount:
        raise ToolDomainError(f"You already have {money(current_amount)}, which meets the {money(target_amount)} target.")
    n = compounds_per_year
    years = math.log(target_amount / current_amount) / (n * math.log(1 + annual_rate / 100 / n))
    return {"years": years, "months": math.ceil(round(years * 12, 6))}


# --- inflation_adjusted_value ----------------------------------------------

class InflationArgs(ToolArgs):
    amount: Number = arg("money", "the dollar amount", "Amount in today's dollars, in dollars.", gt=0, le=1e13)
    inflation_rate: Number = _rate("the inflation rate", "Average annual inflation in percent.", ge=-20, le=100)
    years: Number = _years()


def _explain_inflation(a, r):
    return (f"With {pct(a.inflation_rate)} annual inflation over {years_phrase(a.years)}, something that costs "
            f"{money(a.amount)} today would cost about {money(r['future_cost'])}. Put the other way, "
            f"{money(a.amount)} received {years_phrase(a.years)} from now would only buy what "
            f"{money(r['future_purchasing_power'])} buys today.")


@register("inflation_adjusted_value", category=CATEGORY, args=InflationArgs, explain=_explain_inflation,
          task="work out the effect of inflation",
          description="Effect of inflation: the future cost of today's prices, and today's value of future dollars.")
def inflation_adjusted_value(amount, inflation_rate, years):
    factor = (1 + inflation_rate / 100) ** years
    return {"future_cost": amount * factor, "future_purchasing_power": amount / factor}
