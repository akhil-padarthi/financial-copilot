"""Loans and mortgages."""
from __future__ import annotations

import math
from typing import Optional

from .base import Integer, Number, ToolArgs, ToolDomainError, arg, payment, register, simulate_payoff
from .fmt import join_and, money, num, pct, per_word, years_phrase

CATEGORY = "Loans and mortgages"


def _loan_rate(label="the interest rate", desc="Annual interest rate (APR) in percent."):
    return arg("percent", label, desc, ge=0, le=100)


def _term_months(n_years: Optional[float], n_months: Optional[int]) -> int:
    return int(n_months) if n_months is not None else round(n_years * 12)


def _term_text(years: Optional[float], months: Optional[int]) -> str:
    return years_phrase(years) if years is not None else f"{months} months"


# --- loan_payment -------------------------------------------------------------

class LoanPaymentArgs(ToolArgs):
    principal: Number = arg("money", "the loan amount", "Amount borrowed, in dollars.", gt=0, le=1e11)
    annual_rate: Number = _loan_rate()
    years: Optional[Number] = arg("years", "the loan term", "Loan term in years. Use this OR term_months.", None, gt=0, le=50)
    term_months: Optional[Integer] = arg("months", "the loan term", "Loan term in months. Use this OR years.", None, ge=1, le=600)
    payments_per_year: Integer = arg("freq", "how often you make payments",
                                     "Payments per year (12=monthly, 26=every two weeks, 52=weekly).", 12, ge=1, le=365)

    one_of = ((("years", "term_months"), True),)


def _explain_loan_payment(a, r):
    term = _term_text(a.years, a.term_months)
    return (f"A {money(a.principal)} loan at {pct(a.annual_rate)} over {term} costs {money(r['payment'])} "
            f"{per_word(a.payments_per_year)} ({r['number_of_payments']} payments). Over the life of the loan you'd "
            f"pay {money(r['total_paid'])} in total, of which {money(r['total_interest'])} is interest.")


@register("loan_payment", category=CATEGORY, args=LoanPaymentArgs, explain=_explain_loan_payment,
          task="work out your loan payment",
          description="Payment, total paid and total interest for an amortizing loan (car, personal, student loan).")
def loan_payment(principal, annual_rate, years=None, term_months=None, payments_per_year=12):
    months = term_months if term_months is not None else years * 12
    n = round(months * payments_per_year / 12)
    pay = payment(principal, annual_rate, n, payments_per_year)
    total = pay * n
    return {"payment": pay, "number_of_payments": n, "total_paid": total, "total_interest": total - principal}


# --- mortgage_payment ---------------------------------------------------------

class MortgageArgs(ToolArgs):
    home_price: Number = arg("money", "the home price", "Purchase price of the home, in dollars.", gt=0, le=1e10)
    annual_rate: Number = _loan_rate("the mortgage rate", "Mortgage interest rate in percent.")
    years: Number = arg("years", "the loan term", "Mortgage term in years (e.g. 30 or 15).", gt=0, le=50)
    down_payment: Optional[Number] = arg("money", "your down payment", "Down payment in dollars. Use this OR down_payment_percent.", None, ge=0)
    down_payment_percent: Optional[Number] = arg("percent", "your down payment", "Down payment as a percent of the price. Use this OR down_payment.", None, ge=0, lt=100)
    annual_property_tax: Number = arg("money", "the annual property tax", "Property tax per year, in dollars.", 0, ge=0)
    annual_insurance: Number = arg("money", "the annual homeowners insurance", "Homeowners insurance per year, in dollars.", 0, ge=0)
    monthly_hoa: Number = arg("money", "the monthly HOA dues", "HOA dues per month, in dollars.", 0, ge=0)

    one_of = ((("down_payment", "down_payment_percent"), False),)


def _explain_mortgage(a, r):
    if r["down_payment"] == 0:
        opening = f"With no down payment, you'd borrow the full {money(a.home_price)}."
    elif a.down_payment_percent is not None:
        opening = (f"With {pct(a.down_payment_percent)} down ({money(r['down_payment'])}) on a {money(a.home_price)} "
                   f"home, you'd borrow {money(r['loan_amount'])}.")
    else:
        opening = (f"With {money(r['down_payment'])} down on a {money(a.home_price)} home, you'd borrow "
                   f"{money(r['loan_amount'])}.")
    parts = [opening, f"At {pct(a.annual_rate)} for {years_phrase(a.years)}, principal and interest come to "
                      f"{money(r['principal_and_interest'])} per month."]
    extras = [(r["monthly_property_tax"], "property tax"), (r["monthly_insurance"], "insurance"),
              (r["monthly_hoa"], "HOA dues")]
    extras = [f"{money(v)} for {label}" for v, label in extras if v]
    if extras:
        parts.append(f"Adding {join_and(extras)} brings the total to {money(r['total_monthly_payment'])} per month.")
    parts.append(f"Over the full term you'd pay {money(r['total_interest'])} in interest.")
    return " ".join(parts)


@register("mortgage_payment", category=CATEGORY, args=MortgageArgs, explain=_explain_mortgage,
          task="estimate your monthly mortgage payment",
          description="Monthly mortgage payment from home price and down payment, optionally with tax, insurance and HOA (PITI).")
def mortgage_payment(home_price, annual_rate, years, down_payment=None, down_payment_percent=None,
                     annual_property_tax=0, annual_insurance=0, monthly_hoa=0):
    if down_payment is not None:
        dp = down_payment
    elif down_payment_percent is not None:
        dp = home_price * down_payment_percent / 100
    else:
        dp = 0.0
    if dp >= home_price:
        raise ToolDomainError(f"A down payment of {money(dp)} covers the whole {money(home_price)} price, so there's no mortgage to pay.")
    loan = home_price - dp
    n = round(years * 12)
    pi = payment(loan, annual_rate, n)
    tax, ins = annual_property_tax / 12, annual_insurance / 12
    return {"down_payment": dp, "loan_amount": loan, "principal_and_interest": pi,
            "monthly_property_tax": tax, "monthly_insurance": ins, "monthly_hoa": monthly_hoa,
            "total_monthly_payment": pi + tax + ins + monthly_hoa, "total_interest": pi * n - loan}


# --- remaining_loan_balance ---------------------------------------------------

class RemainingBalanceArgs(ToolArgs):
    principal: Number = arg("money", "the original loan amount", "Original amount borrowed, in dollars.", gt=0, le=1e11)
    annual_rate: Number = _loan_rate()
    years: Number = arg("years", "the original loan term", "Original loan term in years.", gt=0, le=50)
    elapsed_years: Optional[Number] = arg("years", "how long you've been paying", "Years of payments made so far. Use this OR elapsed_months.", None, ge=0, le=50)
    elapsed_months: Optional[Integer] = arg("months", "how long you've been paying", "Monthly payments made so far. Use this OR elapsed_years.", None, ge=0, le=600)

    one_of = ((("elapsed_years", "elapsed_months"), True),)


def _explain_remaining_balance(a, r):
    when = (f"After {years_phrase(a.elapsed_years)} ({r['payments_made']} monthly payments)"
            if a.elapsed_years is not None else f"After {r['payments_made']} monthly payments")
    return (f"{when} of {money(r['monthly_payment'])} on a {money(a.principal)}, {num(a.years)}-year loan at "
            f"{pct(a.annual_rate)}, you'd still owe {money(r['remaining_balance'])}. So far you'd have paid off "
            f"{money(r['principal_paid'])} of principal and {money(r['interest_paid'])} in interest.")


@register("remaining_loan_balance", category=CATEGORY, args=RemainingBalanceArgs, explain=_explain_remaining_balance,
          task="work out how much you still owe",
          description="Balance still owed on an amortizing loan after a number of payments.")
def remaining_loan_balance(principal, annual_rate, years, elapsed_years=None, elapsed_months=None):
    n = round(years * 12)
    k = _term_months(elapsed_years, elapsed_months)
    if k > n:
        raise ToolDomainError(f"A {num(years)}-year loan only has {n} monthly payments, so after {k} payments it would already be paid off.")
    pay = payment(principal, annual_rate, n)
    i = annual_rate / 100 / 12
    if i:
        balance = principal * (1 + i) ** k - pay * ((1 + i) ** k - 1) / i
    else:
        balance = principal - pay * k
    balance = max(balance, 0.0)
    principal_paid = principal - balance
    return {"remaining_balance": balance, "monthly_payment": pay, "payments_made": k,
            "principal_paid": principal_paid, "interest_paid": pay * k - principal_paid}


# --- loan_payoff_time ---------------------------------------------------------

class PayoffTimeArgs(ToolArgs):
    balance: Number = arg("money", "the balance you owe", "Current balance owed, in dollars.", gt=0, le=1e10)
    annual_rate: Number = _loan_rate("the interest rate (APR)", "Annual interest rate / APR in percent.")
    monthly_payment: Number = arg("money", "your monthly payment", "Amount paid every month, in dollars.", gt=0, le=1e10)


def _explain_payoff_time(a, r):
    return (f"Paying {money(a.monthly_payment)} a month on a {money(a.balance)} balance at {pct(a.annual_rate)} "
            f"clears the debt in {r['months']} months (about {num(r['years'])} years). You'd pay "
            f"{money(r['total_interest'])} in interest, {money(r['total_paid'])} in total.")


@register("loan_payoff_time", category=CATEGORY, args=PayoffTimeArgs, explain=_explain_payoff_time,
          task="work out how long it takes to pay off that debt",
          description="Months to pay off a balance (credit card or loan) with a fixed monthly payment, and the interest paid.")
def loan_payoff_time(balance, annual_rate, monthly_payment):
    i = annual_rate / 100 / 12
    first_interest = balance * i
    if monthly_payment <= first_interest:
        raise ToolDomainError(
            f"A payment of {money(monthly_payment)} doesn't cover the {money(first_interest)} of interest charged "
            f"each month, so the balance would never go down. You'd need to pay more than {money(first_interest)} a month.")
    months, interest = simulate_payoff(balance, i, monthly_payment)
    return {"months": months, "years": months / 12, "total_interest": interest, "total_paid": balance + interest}


# --- extra_payment_savings ------------------------------------------------------

class ExtraPaymentArgs(ToolArgs):
    principal: Number = arg("money", "the loan amount", "Loan amount (or current balance), in dollars.", gt=0, le=1e11)
    annual_rate: Number = _loan_rate()
    years: Number = arg("years", "the loan term", "Loan term (or remaining term) in years.", gt=0, le=50)
    extra_monthly_payment: Number = arg("money", "the extra amount you'd pay each month", "Extra amount added to each monthly payment, in dollars.", gt=0, le=1e10)


def _explain_extra_payment(a, r):
    return (f"Adding {money(a.extra_monthly_payment)} a month to the regular {money(r['regular_payment'])} payment on a "
            f"{money(a.principal)} loan at {pct(a.annual_rate)} pays it off in {r['new_months']} months instead of "
            f"{r['original_months']}, which is {r['months_saved']} months sooner. You'd save "
            f"{money(r['interest_saved'])} in interest ({money(r['new_interest'])} instead of {money(r['original_interest'])}).")


@register("extra_payment_savings", category=CATEGORY, args=ExtraPaymentArgs, explain=_explain_extra_payment,
          task="work out what extra payments would save you",
          description="Time and interest saved by adding an extra amount to every monthly loan payment.")
def extra_payment_savings(principal, annual_rate, years, extra_monthly_payment):
    n = round(years * 12)
    regular = payment(principal, annual_rate, n)
    i = annual_rate / 100 / 12
    original_interest = regular * n - principal
    new_months, new_interest = simulate_payoff(principal, i, regular + extra_monthly_payment)
    return {"regular_payment": regular, "new_payment": regular + extra_monthly_payment,
            "original_months": n, "new_months": new_months, "months_saved": n - new_months,
            "original_interest": original_interest, "new_interest": new_interest,
            "interest_saved": original_interest - new_interest}


# --- max_affordable_loan ------------------------------------------------------------

class AffordableLoanArgs(ToolArgs):
    monthly_payment: Number = arg("money", "the monthly payment you can afford", "Monthly principal-and-interest budget, in dollars.", gt=0, le=1e9)
    annual_rate: Number = _loan_rate()
    years: Optional[Number] = arg("years", "the loan term", "Loan term in years. Use this OR term_months.", None, gt=0, le=50)
    term_months: Optional[Integer] = arg("months", "the loan term", "Loan term in months. Use this OR years.", None, ge=1, le=600)

    one_of = ((("years", "term_months"), True),)


def _explain_affordable_loan(a, r):
    return (f"If you can afford {money(a.monthly_payment)} a month, at {pct(a.annual_rate)} over "
            f"{_term_text(a.years, a.term_months)} you could borrow up to {money(r['max_loan_amount'])}. Repaying it "
            f"would cost {money(r['total_paid'])} in total, including {money(r['total_interest'])} of interest. "
            "This covers principal and interest only, not taxes, insurance or fees.")


@register("max_affordable_loan", category=CATEGORY, args=AffordableLoanArgs, explain=_explain_affordable_loan,
          task="work out how much you can borrow",
          description="Largest loan a given monthly payment can support (how much house or car you can afford).")
def max_affordable_loan(monthly_payment, annual_rate, years=None, term_months=None):
    n = _term_months(years, term_months)
    i = annual_rate / 100 / 12
    loan = monthly_payment * ((1 - (1 + i) ** -n) / i if i else n)
    total = monthly_payment * n
    return {"max_loan_amount": loan, "total_paid": total, "total_interest": total - loan, "number_of_payments": n}


# --- refinance_breakeven ---------------------------------------------------------------

class RefinanceArgs(ToolArgs):
    current_balance: Number = arg("money", "your current loan balance", "Balance owed today, in dollars.", gt=0, le=1e11)
    current_rate: Number = _loan_rate("your current interest rate", "Current loan rate in percent.")
    remaining_years: Number = arg("years", "the years left on your current loan", "Years remaining on the current loan.", gt=0, le=50)
    new_rate: Number = _loan_rate("the new interest rate", "Rate on the new loan in percent.")
    new_term_years: Number = arg("years", "the new loan term", "Term of the new loan in years.", gt=0, le=50)
    closing_costs: Number = arg("money", "the closing costs", "Upfront refinancing costs, in dollars.", 0, ge=0)


def _explain_refinance(a, r):
    if r["monthly_savings"] > 0:
        text = (f"Refinancing {money(a.current_balance)} from {pct(a.current_rate)} to {pct(a.new_rate)} over "
                f"{years_phrase(a.new_term_years)} lowers your monthly payment from {money(r['current_payment'])} to "
                f"{money(r['new_payment'])}, saving {money(r['monthly_savings'])} a month.")
        if a.closing_costs:
            text += f" With {money(a.closing_costs)} in closing costs, you'd break even after {r['breakeven_months']} months."
    else:
        text = (f"Refinancing would raise your monthly payment from {money(r['current_payment'])} to "
                f"{money(r['new_payment'])}, so there's no monthly break-even point.")
    net = r["net_lifetime_savings"]
    if net >= 0:
        text += f" Counting total interest and closing costs, you'd come out {money(net)} ahead over the life of the loans."
    else:
        text += (f" Counting total interest and closing costs, it would cost you {money(-net)} more over the life of "
                 "the loans, mostly because of the extra interest from the new term.")
    return text


@register("refinance_breakeven", category=CATEGORY, args=RefinanceArgs, explain=_explain_refinance,
          task="work out whether refinancing pays off",
          description="Monthly savings, break-even months and lifetime cost of refinancing a loan.")
def refinance_breakeven(current_balance, current_rate, remaining_years, new_rate, new_term_years, closing_costs=0):
    n1, n2 = round(remaining_years * 12), round(new_term_years * 12)
    p1 = payment(current_balance, current_rate, n1)
    p2 = payment(current_balance, new_rate, n2)
    savings = p1 - p2
    if savings > 0:
        breakeven = math.ceil(round(closing_costs / savings, 6)) if closing_costs else 0
    else:
        breakeven = None
    interest_current = p1 * n1 - current_balance
    interest_new = p2 * n2 - current_balance
    return {"current_payment": p1, "new_payment": p2, "monthly_savings": savings, "breakeven_months": breakeven,
            "remaining_interest_current": interest_current, "total_interest_new": interest_new,
            "net_lifetime_savings": interest_current - interest_new - closing_costs}
