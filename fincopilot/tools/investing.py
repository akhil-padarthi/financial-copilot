"""Investment returns and valuation."""
from __future__ import annotations

from typing import List, Optional

from pydantic import Field, field_validator

from .base import Integer, Number, NumberList, ToolArgs, ToolDomainError, arg, register
from .fmt import money, money_list, num, pct, years_phrase

CATEGORY = "Investing"

_FLOWS_DESC = "Cash flow for each year after the initial investment, in order, in dollars, e.g. [15000, 20000, 25000]."


def _flows_field(required: bool = True):
    default = ... if required else None
    return Field(default, min_length=1, max_length=50, description=_FLOWS_DESC,
                 json_schema_extra={"kind": "money_list", "label": "the yearly cash flows"})


def _npv(rate: float, initial: float, flows: List[float]) -> float:
    return -initial + sum(cf / (1 + rate) ** t for t, cf in enumerate(flows, start=1))


# --- roi --------------------------------------------------------------------------

class ROIArgs(ToolArgs):
    initial_investment: Number = arg("money", "how much you invested", "Total amount invested, in dollars.", gt=0, le=1e13)
    final_value: Number = arg("money", "what it's worth now", "Current or sale value, in dollars.", ge=0, le=1e14)


def _explain_roi(a, r):
    if r["gain"] >= 0:
        return (f"Your {money(a.initial_investment)} investment is now worth {money(a.final_value)}, a gain of "
                f"{money(r['gain'])}. That's a return on investment (ROI) of {pct(r['roi_percent'])}.")
    return (f"Your {money(a.initial_investment)} investment is now worth {money(a.final_value)}, a loss of "
            f"{money(-r['gain'])}. That's a return on investment (ROI) of {pct(r['roi_percent'])}.")


@register("roi", category=CATEGORY, args=ROIArgs, explain=_explain_roi, task="work out your return on investment",
          description="Return on investment: total gain or loss as a percent of the amount invested.")
def roi(initial_investment, final_value):
    gain = final_value - initial_investment
    return {"gain": gain, "roi_percent": gain / initial_investment * 100}


# --- cagr ---------------------------------------------------------------------------

class CAGRArgs(ToolArgs):
    beginning_value: Number = arg("money", "the starting value", "Value at the start, in dollars.", gt=0, le=1e14)
    ending_value: Number = arg("money", "the ending value", "Value at the end, in dollars.", gt=0, le=1e15)
    years: Number = arg("years", "the number of years", "Years between the two values.", gt=0, le=200)


def _explain_cagr(a, r):
    return (f"Going from {money(a.beginning_value)} to {money(a.ending_value)} over {years_phrase(a.years)} is a total "
            f"return of {pct(r['total_return_percent'])}, which works out to a compound annual growth rate (CAGR) of "
            f"{pct(r['cagr_percent'])} per year.")


@register("cagr", category=CATEGORY, args=CAGRArgs, explain=_explain_cagr, task="work out the annual growth rate",
          description="Compound annual growth rate between a starting and ending value (annualized return).")
def cagr(beginning_value, ending_value, years):
    ratio = ending_value / beginning_value
    return {"cagr_percent": (ratio ** (1 / years) - 1) * 100, "total_return_percent": (ratio - 1) * 100}


# --- npv ------------------------------------------------------------------------------

class NPVArgs(ToolArgs):
    discount_rate: Number = arg("percent", "the discount rate", "Discount rate / cost of capital in percent.", ge=0, le=100)
    initial_investment: Number = arg("money", "the upfront cost", "Upfront cost paid today, in dollars (positive number).", ge=0, le=1e13)
    cash_flows: NumberList = _flows_field()


def _explain_npv(a, r):
    if r["npv"] > 0:
        verdict = "so at that rate the investment is expected to add value"
    elif r["npv"] < 0:
        verdict = "so at that rate the investment would lose value"
    else:
        verdict = "so it exactly breaks even at that rate"
    return (f"At a {pct(a.discount_rate)} discount rate, the {r['number_of_years']} years of cash flows are worth "
            f"{money(r['present_value_of_cash_flows'])} today. Subtracting the {money(a.initial_investment)} upfront "
            f"cost gives a net present value (NPV) of {money(r['npv'])}, {verdict}.")


@register("npv", category=CATEGORY, args=NPVArgs, explain=_explain_npv, task="work out the net present value",
          description="Net present value of an upfront cost followed by yearly cash flows.")
def npv(discount_rate, initial_investment, cash_flows):
    pv = _npv(discount_rate / 100, 0.0, cash_flows)
    return {"npv": pv - initial_investment, "present_value_of_cash_flows": pv, "number_of_years": len(cash_flows)}


# --- irr ------------------------------------------------------------------------------

class IRRArgs(ToolArgs):
    initial_investment: Number = arg("money", "the upfront cost", "Upfront cost paid today, in dollars (positive number).", gt=0, le=1e13)
    cash_flows: NumberList = _flows_field()


def _explain_irr(a, r):
    return (f"Paying {money(a.initial_investment)} upfront for cash flows of {money_list(a.cash_flows)} gives an "
            f"internal rate of return (IRR) of {pct(r['irr_percent'])} per year. That's the discount rate at which the "
            f"investment's net present value is exactly zero.")


@register("irr", category=CATEGORY, args=IRRArgs, explain=_explain_irr, task="work out the internal rate of return",
          description="Internal rate of return: the annual rate at which NPV equals zero.")
def irr(initial_investment, cash_flows):
    lo, hi = -0.9999, 10.0
    f_lo, f_hi = _npv(lo, initial_investment, cash_flows), _npv(hi, initial_investment, cash_flows)
    if f_lo * f_hi > 0:
        raise ToolDomainError("These cash flows don't have an internal rate of return between -99.99% and 1000% per year.")
    for _ in range(200):  # bisection: slow but guaranteed to converge
        mid = (lo + hi) / 2
        f_mid = _npv(mid, initial_investment, cash_flows)
        if f_lo * f_mid <= 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return {"irr_percent": (lo + hi) / 2 * 100, "number_of_years": len(cash_flows)}


# --- payback_period ---------------------------------------------------------------------

class PaybackArgs(ToolArgs):
    initial_investment: Number = arg("money", "the upfront cost", "Upfront cost, in dollars.", gt=0, le=1e13)
    cash_flows: Optional[NumberList] = _flows_field(required=False)
    annual_cash_flow: Optional[Number] = arg("money", "the yearly cash flows", "Same cash flow (or savings) every year, in dollars. Use this OR cash_flows.", None, gt=0)

    one_of = ((("cash_flows", "annual_cash_flow"), True),)


def _explain_payback(a, r):
    if a.annual_cash_flow is not None:
        return (f"Recovering a {money(a.initial_investment)} investment at {money(a.annual_cash_flow)} a year takes "
                f"{num(r['payback_years'])} years.")
    return (f"With cash flows of {money_list(a.cash_flows)}, recovering the {money(a.initial_investment)} investment "
            f"takes {num(r['payback_years'])} years.")


@register("payback_period", category=CATEGORY, args=PaybackArgs, explain=_explain_payback,
          task="work out the payback period",
          description="Years until cumulative cash flows (or savings) repay an upfront cost.")
def payback_period(initial_investment, cash_flows=None, annual_cash_flow=None):
    if annual_cash_flow is not None:
        return {"payback_years": initial_investment / annual_cash_flow}
    cumulative = 0.0
    for year, cf in enumerate(cash_flows, start=1):
        before = cumulative
        cumulative += cf
        if cumulative >= initial_investment and cf > 0:
            return {"payback_years": year - 1 + (initial_investment - before) / cf}
    raise ToolDomainError(f"Those cash flows add up to {money(cumulative)}, which never recovers the "
                          f"{money(initial_investment)} investment.")


# --- dividend_yield ----------------------------------------------------------------------

class DividendYieldArgs(ToolArgs):
    dividend_per_share: Number = arg("money", "the dividend per share", "Dividend paid per share each time, in dollars.", gt=0, le=1e6)
    share_price: Number = arg("money", "the share price", "Current share price, in dollars.", gt=0, le=1e7)
    payments_per_year: Integer = arg("freq", "how often the dividend is paid", "Dividend payments per year (1=annual, 4=quarterly, 12=monthly).", 1, ge=1, le=52)


def _explain_dividend_yield(a, r):
    lead = ""
    if a.payments_per_year > 1:
        word = {2: "twice a year", 4: "quarterly", 12: "monthly"}.get(a.payments_per_year, f"{a.payments_per_year} times a year")
        lead = f"A {money(a.dividend_per_share)} dividend paid {word} adds up to {money(r['annual_dividend'])} per share per year. "
        return lead + f"At a share price of {money(a.share_price)}, that's a dividend yield of {pct(r['dividend_yield_percent'])}."
    return (f"An annual dividend of {money(r['annual_dividend'])} on a {money(a.share_price)} share price is a "
            f"dividend yield of {pct(r['dividend_yield_percent'])}.")


@register("dividend_yield", category=CATEGORY, args=DividendYieldArgs, explain=_explain_dividend_yield,
          task="work out the dividend yield",
          description="Annual dividend income as a percent of the share price.")
def dividend_yield(dividend_per_share, share_price, payments_per_year=1):
    annual = dividend_per_share * payments_per_year
    return {"annual_dividend": annual, "dividend_yield_percent": annual / share_price * 100}


# --- pe_ratio --------------------------------------------------------------------------------

class PERatioArgs(ToolArgs):
    share_price: Number = arg("money", "the share price", "Current share price, in dollars.", gt=0, le=1e7)
    earnings_per_share: Number = arg("money", "the earnings per share (EPS)", "Annual earnings per share, in dollars.", ge=-1e6, le=1e6)

    @field_validator("earnings_per_share")
    @classmethod
    def _nonzero(cls, v: float) -> float:
        if v == 0:
            raise ValueError("earnings per share cannot be zero")
        return v


def _explain_pe(a, r):
    if a.earnings_per_share > 0:
        return (f"At {money(a.share_price)} a share with earnings of {money(a.earnings_per_share)} per share, the stock "
                f"trades at a P/E ratio of {num(r['pe_ratio'])}, meaning investors pay {num(r['pe_ratio'])} times its "
                f"annual earnings. The earnings yield is {pct(r['earnings_yield_percent'])}.")
    return (f"With negative earnings of {money(a.earnings_per_share)} per share, the P/E ratio comes out to "
            f"{num(r['pe_ratio'])}. A negative P/E isn't meaningful for valuation; it just signals the company is losing money.")


@register("pe_ratio", category=CATEGORY, args=PERatioArgs, explain=_explain_pe, task="work out the P/E ratio",
          description="Price-to-earnings ratio and earnings yield of a stock.")
def pe_ratio(share_price, earnings_per_share):
    return {"pe_ratio": share_price / earnings_per_share, "earnings_yield_percent": earnings_per_share / share_price * 100}


# --- sharpe_ratio ----------------------------------------------------------------------------

class SharpeArgs(ToolArgs):
    portfolio_return: Number = arg("percent", "the portfolio's return", "Annual portfolio return in percent.", ge=-100, le=1000)
    risk_free_rate: Number = arg("percent", "the risk-free rate", "Risk-free rate (e.g. Treasury bills) in percent.", ge=-10, le=100)
    volatility: Number = arg("percent", "the volatility (standard deviation)", "Annual standard deviation of returns in percent.", gt=0, le=1000)


def _explain_sharpe(a, r):
    s = r["sharpe_ratio"]
    if s < 0:
        quality = "A negative Sharpe ratio means the portfolio did worse than the risk-free rate."
    elif s < 1:
        quality = "Below 1 is generally considered modest risk-adjusted performance."
    elif s < 2:
        quality = "Between 1 and 2 is generally considered good risk-adjusted performance."
    else:
        quality = "Above 2 is generally considered very good risk-adjusted performance."
    return (f"A {pct(a.portfolio_return)} return against a {pct(a.risk_free_rate)} risk-free rate is "
            f"{num(r['excess_return_percent'])} percentage points of excess return. Dividing by {pct(a.volatility)} "
            f"volatility gives a Sharpe ratio of {num(s)}. {quality}")


@register("sharpe_ratio", category=CATEGORY, args=SharpeArgs, explain=_explain_sharpe,
          task="work out the Sharpe ratio",
          description="Risk-adjusted return: excess return over the risk-free rate divided by volatility.")
def sharpe_ratio(portfolio_return, risk_free_rate, volatility):
    excess = portfolio_return - risk_free_rate
    return {"sharpe_ratio": excess / volatility, "excess_return_percent": excess}


# --- average_cost_per_share ------------------------------------------------------------------

class AverageCostArgs(ToolArgs):
    amount_per_purchase: Number = arg("money", "how much you invest each time", "Fixed dollar amount invested at each purchase.", gt=0, le=1e10)
    share_prices: NumberList = Field(..., min_length=1, max_length=120,
                                     description="Share price at each purchase, in dollars, e.g. [50, 45, 55].",
                                     json_schema_extra={"kind": "price_list", "label": "the share price at each purchase"})

    @field_validator("share_prices")
    @classmethod
    def _positive(cls, v: List[float]) -> List[float]:
        if any(p <= 0 for p in v):
            raise ValueError("every share price must be greater than 0")
        return v


def _explain_average_cost(a, r):
    text = (f"Investing {money(a.amount_per_purchase)} at each of {r['number_of_purchases']} prices buys "
            f"{num(r['total_shares'])} shares for {money(r['total_invested'])}, an average cost of "
            f"{money(r['average_cost_per_share'])} per share.")
    if r["average_cost_per_share"] < r["average_price"]:
        text += (f" That's lower than the simple average price of {money(r['average_price'])}, because a fixed dollar "
                 "amount buys more shares when the price is low.")
    return text


@register("average_cost_per_share", category=CATEGORY, args=AverageCostArgs, explain=_explain_average_cost,
          task="work out your average cost per share",
          description="Dollar-cost averaging: shares bought and average cost when investing a fixed amount at several prices.")
def average_cost_per_share(amount_per_purchase, share_prices):
    shares = sum(amount_per_purchase / p for p in share_prices)
    total = amount_per_purchase * len(share_prices)
    return {"total_invested": total, "total_shares": shares, "average_cost_per_share": total / shares,
            "average_price": sum(share_prices) / len(share_prices), "number_of_purchases": len(share_prices)}


# --- investment_fee_impact ------------------------------------------------------------------------

class FeeImpactArgs(ToolArgs):
    initial_investment: Number = arg("money", "how much you have invested", "Amount invested, in dollars.", gt=0, le=1e13)
    annual_return: Number = arg("percent", "the expected annual return (before fees)", "Annual return before fees, in percent.", ge=-50, le=100)
    annual_fee: Number = arg("percent", "the annual fee", "Annual fee or expense ratio in percent (1 means 1%).", ge=0, le=20)
    years: Number = arg("years", "the number of years", "Investment horizon in years.", gt=0, le=100)


def _explain_fee_impact(a, r):
    return (f"Over {years_phrase(a.years)}, {money(a.initial_investment)} growing at {pct(a.annual_return)} a year "
            f"would reach {money(r['value_without_fees'])} with no fees, but only {money(r['value_with_fees'])} after a "
            f"{pct(a.annual_fee)} annual fee. The fee costs you {money(r['fee_cost'])}, about "
            f"{pct(r['percent_of_value_lost'])} of what you'd otherwise have.")


@register("investment_fee_impact", category=CATEGORY, args=FeeImpactArgs, explain=_explain_fee_impact,
          task="work out what that fee costs you",
          description="How much an annual fee or expense ratio reduces an investment's ending value.")
def investment_fee_impact(initial_investment, annual_return, annual_fee, years):
    gross = initial_investment * (1 + annual_return / 100) ** years
    net = initial_investment * (1 + (annual_return - annual_fee) / 100) ** years
    return {"value_without_fees": gross, "value_with_fees": net, "fee_cost": gross - net,
            "percent_of_value_lost": (gross - net) / gross * 100}
