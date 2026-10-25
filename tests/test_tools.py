"""Tests for the calculation layer.

Three kinds of checks:
1. Textbook values that are published widely (e.g. $200k at 6% for 30 years = $1,199.10/month).
2. Independent re-computation: closed-form formulas are compared with brute-force
   month-by-month simulations written separately here.
3. Round trips: inverse tools must undo each other (PV <-> FV, payment <-> max loan, ...).

Runs with `python -m unittest` or `pytest`.
"""
import math
import random
import unittest

from pydantic import ValidationError

from fincopilot.tools import REGISTRY, ToolDomainError, get_tool


def run(name, **kwargs):
    return get_tool(name).run(kwargs).result


def raw(name, **kwargs):
    """Unrounded result, for precise comparisons."""
    tool = get_tool(name)
    args = tool.validate(kwargs)
    return tool.func(**args.model_dump())


# Independent reference implementations (deliberately naive) -----------------

def sim_fv_contributions(contribution, rate, years, initial=0.0, m=12):
    b = initial
    for _ in range(round(m * years)):
        b = b * (1 + rate / 100 / m) + contribution
    return b


def sim_balance(principal, rate, pmt, months):
    b = principal
    for _ in range(months):
        b = b * (1 + rate / 100 / 12) - pmt
    return b


def bisect(f, lo, hi, iters=300):
    flo = f(lo)
    for _ in range(iters):
        mid = (lo + hi) / 2
        fm = f(mid)
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return (lo + hi) / 2


class TestTextbookValues(unittest.TestCase):
    def test_compound_interest_readme_example(self):
        r = run("compound_interest", principal=10_000, annual_rate=7, years=10)
        self.assertEqual(r["future_value"], 19671.51)

    def test_compound_monthly(self):
        r = run("compound_interest", principal=1000, annual_rate=5, years=10, compounds_per_year=12)
        self.assertEqual(r["future_value"], 1647.01)

    def test_mortgage_textbook_payment(self):
        self.assertEqual(run("loan_payment", principal=200_000, annual_rate=6, years=30)["payment"], 1199.10)
        self.assertEqual(run("loan_payment", principal=100_000, annual_rate=4.5, years=15)["payment"], 764.99)

    def test_zero_rate_loan(self):
        self.assertEqual(run("loan_payment", principal=12_000, annual_rate=0, term_months=24)["payment"], 500.0)

    def test_simple_interest(self):
        self.assertEqual(run("simple_interest", principal=1000, annual_rate=5, years=3)["interest"], 150.0)

    def test_present_value(self):
        self.assertEqual(run("present_value", future_value=10_000, annual_rate=5, years=10)["present_value"], 6139.13)

    def test_effective_annual_rate(self):
        self.assertEqual(run("effective_annual_rate", nominal_rate=12, compounds_per_year=12)["effective_annual_rate"], 12.68)

    def test_rule_of_72(self):
        r = run("rule_of_72", annual_rate=8)
        self.assertEqual(r["estimated_years"], 9.0)
        self.assertEqual(r["exact_years"], 9.01)

    def test_continuous(self):
        self.assertEqual(run("continuous_compounding", principal=1000, annual_rate=5, years=10)["future_value"], 1648.72)

    def test_npv_classic(self):
        self.assertEqual(run("npv", discount_rate=10, initial_investment=100, cash_flows=[60, 60])["npv"], 4.13)

    def test_irr_classic(self):
        self.assertEqual(run("irr", initial_investment=100, cash_flows=[60, 60])["irr_percent"], 13.07)

    def test_payback(self):
        self.assertEqual(run("payback_period", initial_investment=1000, cash_flows=[300, 400, 500])["payback_years"], 2.6)
        self.assertEqual(run("payback_period", initial_investment=18000, annual_cash_flow=2400)["payback_years"], 7.5)

    def test_cagr(self):
        self.assertEqual(run("cagr", beginning_value=100, ending_value=200, years=5)["cagr_percent"], 14.87)

    def test_break_even(self):
        r = run("break_even_units", fixed_costs=10_000, price_per_unit=25, variable_cost_per_unit=15)
        self.assertEqual(r["break_even_units"], 1000.0)
        self.assertEqual(r["units_to_sell"], 1000)

    def test_budget_and_dti(self):
        r = run("budget_50_30_20", annual_income=60_000)
        self.assertEqual((r["needs"], r["wants"], r["savings"]), (2500.0, 1500.0, 1000.0))
        self.assertEqual(run("debt_to_income", monthly_debt_payments=1500, gross_monthly_income=5000)["dti_percent"], 30.0)

    def test_nest_egg(self):
        self.assertEqual(run("retirement_nest_egg", annual_expenses=40_000)["nest_egg"], 1_000_000.0)
        self.assertEqual(run("retirement_nest_egg", monthly_expenses=5_000)["nest_egg"], 1_500_000.0)

    def test_markup_margin(self):
        r = run("markup_percentage", cost=60, selling_price=100)
        self.assertEqual((r["markup_percent"], r["margin_percent"]), (66.67, 40.0))

    def test_dollar_cost_averaging_is_harmonic_mean(self):
        r = run("average_cost_per_share", amount_per_purchase=100, share_prices=[10, 20])
        self.assertEqual(r["average_cost_per_share"], 13.33)
        self.assertEqual(r["average_price"], 15.0)


class TestAgainstSimulation(unittest.TestCase):
    """Closed-form formulas vs. brute-force simulation on random inputs."""

    def setUp(self):
        self.rng = random.Random(7)

    def test_loan_payment_amortizes_to_zero(self):
        for _ in range(200):
            p, r, y = self.rng.uniform(1e3, 1e6), self.rng.uniform(0, 20), self.rng.randint(1, 40)
            pmt = raw("loan_payment", principal=p, annual_rate=r, years=y)["payment"]
            self.assertAlmostEqual(sim_balance(p, r, pmt, y * 12), 0, delta=1e-6 * p)

    def test_fv_contributions(self):
        for _ in range(200):
            c, r, y, b = self.rng.uniform(10, 5000), self.rng.uniform(0, 15), self.rng.randint(1, 40), self.rng.uniform(0, 1e5)
            m = self.rng.choice([1, 4, 12, 26, 52])
            got = raw("future_value_with_contributions", contribution=c, annual_rate=r, years=y,
                      initial_balance=b, contributions_per_year=m)["future_value"]
            self.assertAlmostEqual(got, sim_fv_contributions(c, r, y, b, m), delta=1e-7 * got)

    def test_remaining_balance(self):
        for _ in range(200):
            p, r, y = self.rng.uniform(1e4, 1e6), self.rng.uniform(0, 12), self.rng.randint(5, 30)
            k = self.rng.randint(0, y * 12)
            res = raw("remaining_loan_balance", principal=p, annual_rate=r, years=y, elapsed_months=k)
            expected = max(sim_balance(p, r, res["monthly_payment"], k), 0)
            self.assertAlmostEqual(res["remaining_balance"], expected, delta=1e-6 * p)

    def test_payoff_time_matches_payment(self):
        # Paying exactly the amortizing payment must take exactly n months.
        for _ in range(100):
            p, r, y = self.rng.uniform(1e3, 1e5), self.rng.uniform(1, 25), self.rng.randint(1, 10)
            pmt = raw("loan_payment", principal=p, annual_rate=r, years=y)["payment"]
            res = raw("loan_payoff_time", balance=p, annual_rate=r, monthly_payment=pmt)
            self.assertEqual(res["months"], y * 12)
            self.assertAlmostEqual(res["total_interest"], pmt * y * 12 - p, delta=0.01)

    def test_extra_payments_save_money(self):
        for _ in range(100):
            p, r, y, e = self.rng.uniform(1e4, 5e5), self.rng.uniform(2, 10), self.rng.randint(5, 30), self.rng.uniform(25, 1000)
            res = raw("extra_payment_savings", principal=p, annual_rate=r, years=y, extra_monthly_payment=e)
            self.assertGreater(res["interest_saved"], 0)
            self.assertGreaterEqual(res["months_saved"], 0)

    def test_drawdown_matches_simulation(self):
        res = raw("retirement_drawdown", balance=100_000, annual_withdrawal=10_000, annual_return=0)
        self.assertAlmostEqual(res["years_lasting"], 10.0)
        res = raw("retirement_drawdown", balance=100_000, annual_withdrawal=3_000, annual_return=5)
        self.assertTrue(res["lasts_indefinitely"])


class TestRoundTrips(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(11)

    def test_pv_fv_inverse(self):
        for _ in range(100):
            fv, r, y, n = self.rng.uniform(1e3, 1e6), self.rng.uniform(0.5, 15), self.rng.uniform(1, 40), self.rng.choice([1, 4, 12, 365])
            pv = raw("present_value", future_value=fv, annual_rate=r, years=y, compounds_per_year=n)["present_value"]
            back = raw("compound_interest", principal=pv, annual_rate=r, years=y, compounds_per_year=n)["future_value"]
            self.assertAlmostEqual(back, fv, delta=1e-8 * fv)

    def test_affordable_loan_inverse_of_payment(self):
        for _ in range(100):
            pmt, r, y = self.rng.uniform(100, 10_000), self.rng.uniform(0, 12), self.rng.randint(1, 30)
            loan = raw("max_affordable_loan", monthly_payment=pmt, annual_rate=r, years=y)["max_loan_amount"]
            back = raw("loan_payment", principal=loan, annual_rate=r, years=y)["payment"]
            self.assertAlmostEqual(back, pmt, delta=1e-8 * pmt)

    def test_savings_goal_inverse_of_fv(self):
        for _ in range(100):
            target, y, r, cur = self.rng.uniform(1e4, 2e6), self.rng.randint(1, 40), self.rng.uniform(0, 12), self.rng.uniform(0, 5e3)
            c = raw("savings_goal_monthly", target_amount=target, years=y, annual_rate=r, current_savings=cur)["monthly_contribution"]
            if c == 0:
                continue
            back = raw("future_value_with_contributions", contribution=c, annual_rate=r, years=y, initial_balance=cur)["future_value"]
            self.assertAlmostEqual(back, target, delta=1e-7 * target)

    def test_years_to_goal_inverse(self):
        for _ in range(100):
            cur, r, y = self.rng.uniform(1e3, 1e5), self.rng.uniform(1, 15), self.rng.uniform(1, 40)
            target = raw("compound_interest", principal=cur, annual_rate=r, years=y)["future_value"]
            self.assertAlmostEqual(raw("years_to_goal", current_amount=cur, target_amount=target, annual_rate=r)["years"], y, places=6)

    def test_cagr_inverse(self):
        for _ in range(100):
            b, r, y = self.rng.uniform(1e3, 1e6), self.rng.uniform(-20, 40), self.rng.uniform(1, 30)
            e = b * (1 + r / 100) ** y
            self.assertAlmostEqual(raw("cagr", beginning_value=b, ending_value=e, years=y)["cagr_percent"], r, places=6)

    def test_irr_zeroes_npv(self):
        for _ in range(100):
            initial = self.rng.uniform(1e3, 1e5)
            flows = [self.rng.uniform(0, initial) for _ in range(self.rng.randint(2, 8))]
            try:
                irr = raw("irr", initial_investment=initial, cash_flows=flows)["irr_percent"]
            except ToolDomainError:
                continue
            npv = raw("npv", discount_rate=max(irr, 0), initial_investment=initial, cash_flows=flows)["npv"] if irr >= 0 else None
            if npv is not None:
                self.assertAlmostEqual(npv, 0, delta=1e-6 * initial)
            # Independent root-finder (scipy if available, else our own bisection).
            f = lambda rate: -initial + sum(cf / (1 + rate) ** t for t, cf in enumerate(flows, 1))
            try:
                from scipy.optimize import brentq
                ref = brentq(f, -0.9999, 10, xtol=1e-14) * 100
            except ImportError:
                ref = bisect(f, -0.9999, 10) * 100
            self.assertAlmostEqual(irr, ref, places=6)


class TestValidation(unittest.TestCase):
    """The argument forms are the safety net between the model and the math."""

    def test_friendly_numbers_are_coerced(self):
        t = get_tool("compound_interest")
        args = t.validate({"principal": "$10,000", "annual_rate": "7%", "years": "10"})
        self.assertEqual((args.principal, args.annual_rate, args.years), (10_000, 7, 10))
        self.assertEqual(t.validate({"principal": "10k", "annual_rate": 7, "years": 1}).principal, 10_000)
        self.assertEqual(t.validate({"principal": "1.2 million", "annual_rate": 7, "years": 1}).principal, 1_200_000)

    def test_cash_flow_string_is_coerced(self):
        args = get_tool("npv").validate({"discount_rate": 8, "initial_investment": 100, "cash_flows": "15000, 20000 and 25000"})
        self.assertEqual(args.cash_flows, [15000, 20000, 25000])

    def test_missing_field_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            get_tool("compound_interest").validate({"principal": 100, "annual_rate": 7})
        self.assertEqual(ctx.exception.errors()[0]["type"], "missing")

    def test_invented_parameter_rejected(self):
        with self.assertRaises(ValidationError):
            get_tool("compound_interest").validate({"principal": 100, "rate": 7, "years": 3})

    def test_out_of_range_rejected(self):
        for bad in ({"principal": -5, "annual_rate": 7, "years": 3},
                    {"principal": 100, "annual_rate": 700, "years": 3},
                    {"principal": 100, "annual_rate": 7, "years": 0},
                    {"principal": float("nan"), "annual_rate": 7, "years": 3},
                    {"principal": True, "annual_rate": 7, "years": 3},
                    {"principal": "lots", "annual_rate": 7, "years": 3}):
            with self.assertRaises(ValidationError, msg=str(bad)):
                get_tool("compound_interest").validate(bad)

    def test_one_of_groups(self):
        loan = get_tool("loan_payment")
        with self.assertRaises(ValidationError) as ctx:
            loan.validate({"principal": 1000, "annual_rate": 5})
        self.assertEqual(ctx.exception.errors()[0]["type"], "missing_one_of")
        with self.assertRaises(ValidationError) as ctx:
            loan.validate({"principal": 1000, "annual_rate": 5, "years": 2, "term_months": 24})
        self.assertEqual(ctx.exception.errors()[0]["type"], "conflicting_fields")
        a = loan.run({"principal": 35000, "annual_rate": 7.9, "term_months": 72}).result
        b = loan.run({"principal": 35000, "annual_rate": 7.9, "years": 6}).result
        self.assertEqual(a, b)

    def test_mortgage_down_payment_forms_agree(self):
        m = get_tool("mortgage_payment")
        a = m.run({"home_price": 300000, "annual_rate": 6.5, "years": 30, "down_payment": 60000}).result
        b = m.run({"home_price": 300000, "annual_rate": 6.5, "years": 30, "down_payment_percent": 20}).result
        self.assertEqual(a, b)

    def test_domain_errors_are_friendly(self):
        with self.assertRaises(ToolDomainError) as ctx:
            run("loan_payoff_time", balance=10_000, annual_rate=24, monthly_payment=150)
        self.assertIn("$200", str(ctx.exception))  # 10,000 * 2% monthly interest
        with self.assertRaises(ToolDomainError):
            run("break_even_units", fixed_costs=100, price_per_unit=5, variable_cost_per_unit=6)
        with self.assertRaises(ToolDomainError):
            run("payback_period", initial_investment=1000, cash_flows=[100, 100])
        with self.assertRaises(ValidationError):
            run("pe_ratio", share_price=10, earnings_per_share=0)


class TestRegistry(unittest.TestCase):
    def test_at_least_32_tools(self):
        self.assertGreaterEqual(len(REGISTRY), 32)

    def test_every_tool_documented(self):
        for tool in REGISTRY.values():
            self.assertTrue(tool.description and tool.task, tool.name)
            for name, info in tool.fields.items():
                self.assertTrue(info.description, f"{tool.name}.{name}")
                self.assertIn("label", tool.field_meta(name), f"{tool.name}.{name}")
                self.assertIn("kind", tool.field_meta(name), f"{tool.name}.{name}")

    def test_results_are_rounded_and_finite(self):
        r = run("compound_interest", principal=1234.567, annual_rate=3.33, years=7.7, compounds_per_year=365)
        for v in r.values():
            self.assertTrue(math.isfinite(v))
            self.assertEqual(round(v, 2), v)


if __name__ == "__main__":
    unittest.main()
