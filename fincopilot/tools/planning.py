"""Personal financial planning: savings goals, retirement, budgeting."""
from __future__ import annotations

from typing import Optional

from .base import Number, ToolArgs, ToolDomainError, arg, register
from .fmt import money, num, pct, years_phrase

CATEGORY = "Personal planning"


# --- savings_goal_monthly ---------------------------------------------------------

class SavingsGoalArgs(ToolArgs):
    target_amount: Number = arg("money", "your savings goal", "Amount you want to have, in dollars.", gt=0, le=1e13)
    years: Number = arg("years", "how many years you have", "Years until the goal.", gt=0, le=100)
    annual_rate: Number = arg("percent", "the expected annual return", "Expected annual return on savings in percent (0 for cash).", ge=0, le=100)
    current_savings: Number = arg("money", "what you've already saved", "Amount already saved toward the goal, in dollars.", 0, ge=0, le=1e13)


def _explain_savings_goal(a, r):
    if r["monthly_contribution"] == 0:
        return (f"Good news: your current {money(a.current_savings)} would grow to "
                f"{money(r['current_savings_future_value'])} in {years_phrase(a.years)} at {pct(a.annual_rate)}, "
                f"which already reaches your {money(a.target_amount)} goal without adding anything.")
    start = f", starting from {money(a.current_savings)}," if a.current_savings else ""
    text = (f"To reach {money(a.target_amount)} in {years_phrase(a.years)}{start} at a {pct(a.annual_rate)} annual "
            f"return, you'd need to save {money(r['monthly_contribution'])} a month. Over {r['number_of_months']} "
            f"months you'd contribute {money(r['total_contributions'])}")
    if r["investment_growth"] > 0:
        text += f", and investment growth would cover the remaining {money(r['investment_growth'])}."
    else:
        text += "."
    return text


@register("savings_goal_monthly", category=CATEGORY, args=SavingsGoalArgs, explain=_explain_savings_goal,
          task="work out how much to save each month",
          description="Monthly saving needed to reach a target amount by a deadline.")
def savings_goal_monthly(target_amount, years, annual_rate, current_savings=0):
    n = round(years * 12)
    if n < 1:
        raise ToolDomainError("The deadline needs to be at least a month away.")
    i = annual_rate / 100 / 12
    growth = (1 + i) ** n
    future_current = current_savings * growth
    remaining = target_amount - future_current
    if remaining <= 0:
        contribution = 0.0
    else:
        contribution = remaining * i / (growth - 1) if i else remaining / n
    total = contribution * n
    return {"monthly_contribution": contribution, "total_contributions": total,
            "current_savings_future_value": future_current,
            "investment_growth": max(target_amount - current_savings - total, 0.0) if contribution else 0.0,
            "number_of_months": n}


# --- retirement_nest_egg -------------------------------------------------------------

class NestEggArgs(ToolArgs):
    annual_expenses: Optional[Number] = arg("money", "your expected spending in retirement", "Spending per year in retirement, in dollars. Use this OR monthly_expenses.", None, gt=0, le=1e10)
    monthly_expenses: Optional[Number] = arg("money", "your expected spending in retirement", "Spending per month in retirement, in dollars. Use this OR annual_expenses.", None, gt=0, le=1e9)
    withdrawal_rate: Number = arg("percent", "the withdrawal rate", "Safe withdrawal rate in percent (the 4% rule by default).", 4, gt=0, le=20)

    one_of = ((("annual_expenses", "monthly_expenses"), True),)


def _explain_nest_egg(a, r):
    lead = ""
    if a.monthly_expenses is not None:
        lead = f"{money(a.monthly_expenses)} a month is {money(r['annual_expenses'])} a year. "
    return (lead + f"To cover {money(r['annual_expenses'])} a year with a {pct(a.withdrawal_rate)} withdrawal rate, "
            f"you'd need about {money(r['nest_egg'])} saved, which is {num(r['multiple_of_spending'])} times your "
            f"annual spending. A {pct(a.withdrawal_rate)} withdrawal rate is a rule of thumb based on historical "
            "returns, not a guarantee.")


@register("retirement_nest_egg", category=CATEGORY, args=NestEggArgs, explain=_explain_nest_egg,
          task="work out how much you need to retire",
          description="Savings needed to fund retirement spending at a withdrawal rate (the 4% rule / FIRE number).")
def retirement_nest_egg(annual_expenses=None, monthly_expenses=None, withdrawal_rate=4):
    annual = annual_expenses if annual_expenses is not None else monthly_expenses * 12
    return {"annual_expenses": annual, "nest_egg": annual / (withdrawal_rate / 100),
            "multiple_of_spending": 100 / withdrawal_rate}


# --- retirement_drawdown -----------------------------------------------------------------

class DrawdownArgs(ToolArgs):
    balance: Number = arg("money", "your savings balance", "Savings at the start of retirement, in dollars.", gt=0, le=1e13)
    annual_withdrawal: Optional[Number] = arg("money", "how much you'll withdraw", "Amount withdrawn per year, in dollars. Use this OR monthly_withdrawal.", None, gt=0)
    monthly_withdrawal: Optional[Number] = arg("money", "how much you'll withdraw", "Amount withdrawn per month, in dollars. Use this OR annual_withdrawal.", None, gt=0)
    annual_return: Number = arg("percent", "the expected annual return", "Expected annual return on the remaining balance, in percent.", ge=-50, le=100)

    one_of = ((("annual_withdrawal", "monthly_withdrawal"), True),)


def _explain_drawdown(a, r):
    lead = ""
    if a.monthly_withdrawal is not None:
        lead = f"{money(a.monthly_withdrawal)} a month is {money(r['annual_withdrawal'])} a year. "
    if r["lasts_indefinitely"]:
        return (lead + f"Withdrawing {money(r['annual_withdrawal'])} a year from {money(a.balance)} earning "
                f"{pct(a.annual_return)} a year, your savings would last more than 100 years, because the returns keep "
                "up with the withdrawals.")
    return (lead + f"Withdrawing {money(r['annual_withdrawal'])} at the start of each year from {money(a.balance)} "
            f"earning {pct(a.annual_return)} a year, your savings would last about {num(r['years_lasting'])} years and "
            f"pay out {money(r['total_withdrawn'])} in total.")


@register("retirement_drawdown", category=CATEGORY, args=DrawdownArgs, explain=_explain_drawdown, constants=(100,),
          task="work out how long your savings will last",
          description="How many years savings last with fixed withdrawals while the balance keeps earning a return.")
def retirement_drawdown(balance, annual_withdrawal=None, monthly_withdrawal=None, annual_return=0):
    w = annual_withdrawal if annual_withdrawal is not None else monthly_withdrawal * 12
    r = annual_return / 100
    b, years = balance, 0.0
    for _ in range(100):
        if b >= w:
            b = (b - w) * (1 + r)
            years += 1
        else:
            years += b / w
            return {"annual_withdrawal": w, "years_lasting": years, "lasts_indefinitely": False,
                    "total_withdrawn": w * years}
    return {"annual_withdrawal": w, "years_lasting": None, "lasts_indefinitely": True, "total_withdrawn": None}


# --- emergency_fund ----------------------------------------------------------------------------

class EmergencyFundArgs(ToolArgs):
    monthly_expenses: Number = arg("money", "your monthly expenses", "Essential spending per month, in dollars.", gt=0, le=1e9)
    months: Number = arg("count", "how many months to cover", "Months of expenses to keep in reserve (usually 3 to 6).", 6, ge=1, le=36)
    current_savings: Number = arg("money", "what you have saved", "Money already set aside, in dollars.", 0, ge=0, le=1e12)


def _explain_emergency(a, r):
    text = (f"Covering {num(a.months)} months of {money(a.monthly_expenses)} in expenses means an emergency fund of "
            f"{money(r['target'])}.")
    if a.current_savings:
        if r["shortfall"] > 0:
            text += (f" With {money(a.current_savings)} saved (about {num(r['months_covered'])} months), you're "
                     f"{money(r['shortfall'])} short.")
        else:
            text += f" With {money(a.current_savings)} saved (about {num(r['months_covered'])} months), you're fully covered."
    return text


@register("emergency_fund", category=CATEGORY, args=EmergencyFundArgs, explain=_explain_emergency,
          task="size your emergency fund",
          description="Emergency fund target from monthly expenses, and the shortfall versus current savings.")
def emergency_fund(monthly_expenses, months=6, current_savings=0):
    target = monthly_expenses * months
    return {"target": target, "shortfall": max(target - current_savings, 0.0),
            "months_covered": current_savings / monthly_expenses}


# --- debt_to_income ---------------------------------------------------------------------------------

class DebtToIncomeArgs(ToolArgs):
    monthly_debt_payments: Number = arg("money", "your monthly debt payments", "Total monthly debt payments, in dollars.", ge=0, le=1e9)
    gross_monthly_income: Optional[Number] = arg("money", "your gross income", "Pre-tax income per month. Use this OR gross_annual_income.", None, gt=0, le=1e9)
    gross_annual_income: Optional[Number] = arg("money", "your gross income", "Pre-tax income per year. Use this OR gross_monthly_income.", None, gt=0, le=1e10)

    one_of = ((("gross_monthly_income", "gross_annual_income"), True),)


def _explain_dti(a, r):
    lead = ""
    if a.gross_annual_income is not None:
        lead = f"A gross income of {money(a.gross_annual_income)} a year is {money(r['gross_monthly_income'])} a month. "
    text = (lead + f"Monthly debt payments of {money(a.monthly_debt_payments)} against a gross monthly income of "
            f"{money(r['gross_monthly_income'])} give a debt-to-income ratio of {pct(r['dti_percent'])}. ")
    status = r["status"]
    if status == "healthy":
        text += "That's in the range most lenders consider healthy (36% or less)."
    elif status == "manageable":
        text += "That's above the 36% comfort zone but within the 43% limit many mortgage lenders use."
    else:
        text += "That's above the 43% level many mortgage lenders treat as a ceiling, so new credit may be hard to get."
    return text


@register("debt_to_income", category=CATEGORY, args=DebtToIncomeArgs, explain=_explain_dti, constants=(36, 43),
          task="work out your debt-to-income ratio",
          description="Debt-to-income ratio (DTI) from monthly debt payments and gross income, with lender thresholds.")
def debt_to_income(monthly_debt_payments, gross_monthly_income=None, gross_annual_income=None):
    income = gross_monthly_income if gross_monthly_income is not None else gross_annual_income / 12
    dti = monthly_debt_payments / income * 100
    status = "healthy" if dti <= 36 else "manageable" if dti <= 43 else "high"
    return {"gross_monthly_income": income, "dti_percent": dti, "status": status}


# --- budget_50_30_20 -----------------------------------------------------------------------------------

class BudgetArgs(ToolArgs):
    monthly_income: Optional[Number] = arg("money", "your take-home pay", "After-tax income per month. Use this OR annual_income.", None, gt=0, le=1e9)
    annual_income: Optional[Number] = arg("money", "your take-home pay", "After-tax income per year. Use this OR monthly_income.", None, gt=0, le=1e10)

    one_of = ((("monthly_income", "annual_income"), True),)


def _explain_budget(a, r):
    lead = ""
    if a.annual_income is not None:
        lead = f"{money(a.annual_income)} a year is {money(r['monthly_income'])} a month. "
    return (lead + f"Applying the 50/30/20 rule to {money(r['monthly_income'])} of monthly take-home pay: "
            f"{money(r['needs'])} for needs (50%), {money(r['wants'])} for wants (30%), and {money(r['savings'])} for "
            "savings and extra debt payments (20%).")


@register("budget_50_30_20", category=CATEGORY, args=BudgetArgs, explain=_explain_budget,
          task="build your 50/30/20 budget",
          description="Split take-home pay into needs (50%), wants (30%) and savings (20%).")
def budget_50_30_20(monthly_income=None, annual_income=None):
    m = monthly_income if monthly_income is not None else annual_income / 12
    return {"monthly_income": m, "needs": m * 0.5, "wants": m * 0.3, "savings": m * 0.2,
            "needs_percent": 50, "wants_percent": 30, "savings_percent": 20}
