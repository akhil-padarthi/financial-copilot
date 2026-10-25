"""Tests for the evaluation harness: the scorer must be right before its numbers mean anything."""
import json
import os
import tempfile
import unittest

from fincopilot.backends import ScriptedBackend
from fincopilot.datagen.generate import Generator, build_realworld
from fincopilot.evaluation import load_sets, markdown_report, run, score, summarize
from fincopilot.tools import to_json


def oracle_reply(item):
    exp = item["expected"]
    if exp["action"] == "tool":
        return to_json({"name": exp["name"], "parameters": exp["parameters"]})
    if exp["action"] == "clarify":
        return "Could you tell me the interest rate?"
    return "Here's a short explanation."


class TestScoring(unittest.TestCase):
    item = {"id": "x", "messages": [{"role": "user", "content": "35k car loan, 72 months at 7.9%?"}],
            "expected": {"action": "tool", "name": "loan_payment",
                         "parameters": {"principal": 35000, "annual_rate": 7.9, "term_months": 72}}}

    def test_exact(self):
        r = score(self.item, '{"name": "loan_payment", "parameters": {"principal": 35000, "annual_rate": 7.9, "term_months": 72}}')
        self.assertTrue(r["answer_correct"] and r["args_exact"] and r["correct"])

    def test_equivalent_form_counts_as_correct_answer(self):
        r = score(self.item, '{"name": "loan_payment", "parameters": {"principal": 35000, "annual_rate": 7.9, "years": 6}}')
        self.assertTrue(r["answer_correct"])
        self.assertFalse(r["args_exact"])

    def test_explicit_default_is_exact(self):
        r = score(self.item, '{"name": "loan_payment", "parameters": {"principal": 35000, "annual_rate": 7.9, "term_months": 72, "payments_per_year": 12}}')
        self.assertTrue(r["args_exact"])

    def test_wrong_value(self):
        r = score(self.item, '{"name": "loan_payment", "parameters": {"principal": 35000, "annual_rate": 0.079, "term_months": 72}}')
        self.assertFalse(r["answer_correct"])
        self.assertTrue(r["args_valid"])

    def test_wrong_tool(self):
        r = score(self.item, '{"name": "max_affordable_loan", "parameters": {"monthly_payment": 35000, "annual_rate": 7.9, "term_months": 72}}')
        self.assertFalse(r["correct"])
        self.assertEqual(r["predicted_tool"], "max_affordable_loan")

    def test_text_instead_of_tool(self):
        r = score(self.item, "Your payment would be about $611 a month.")
        self.assertEqual(r["predicted_action"], "text")
        self.assertFalse(r["correct"])

    def test_unparseable(self):
        r = score(self.item, '{"name": "loan_payment", "parameters": {')
        self.assertFalse(r["valid_json"])

    def test_clarify_and_text_items(self):
        clarify = {"id": "c", "messages": [{"role": "user", "content": "loan payment?"}], "expected": {"action": "clarify", "name": None}}
        self.assertTrue(score(clarify, "What's the rate and term?")["correct"])
        self.assertFalse(score(clarify, '{"name": "loan_payment", "parameters": {"principal": 1}}')["correct"])


class TestHarness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items = Generator(5).build_test()[:300] + build_realworld()
        for it in cls.items:
            it["_set"] = "realworld" if it["id"].startswith("realworld") else "test"
        cls.lookup = {json.dumps(it["messages"]): it for it in cls.items}

    def _backend(self, fn):
        return ScriptedBackend(lambda msgs: fn(self.lookup[json.dumps([m for m in msgs if m["role"] != "system"])]))

    def test_oracle_scores_100(self):
        results = run(self._backend(oracle_reply), self.items, verbose=False)
        s = summarize(results)
        for key in ("overall_accuracy", "answer_accuracy", "routing_accuracy", "argument_match", "valid_call_rate",
                    "asked_when_info_missing", "no_tool_when_not_needed"):
            self.assertEqual(s["all"][key], 1.0, key)
        self.assertEqual(s["all"]["false_tool_call_rate"], 0.0)
        self.assertEqual(s["realworld"]["items"], len(build_realworld()))

    def test_never_calls_tools(self):
        s = summarize(run(self._backend(lambda it: "I'm not sure."), self.items, verbose=False))
        self.assertEqual(s["all"]["answer_accuracy"], 0.0)
        self.assertEqual(s["all"]["no_tool_when_not_needed"], 1.0)

    def test_explanations_checked(self):
        def fn(msgs):
            core = [m for m in msgs if m["role"] != "system"]
            if core[-1]["role"] == "tool":
                return "Here you go: the answer is $123,456,789.12."  # ungrounded on purpose
            return oracle_reply(self.lookup[json.dumps(core)])
        results = run(ScriptedBackend(fn), self.items[:40], explain=True, verbose=False)
        s = summarize(results)
        self.assertEqual(s["all"]["explanation_grounded_rate"], 0.0)

    def test_report_and_files(self):
        results = run(self._backend(oracle_reply), self.items, verbose=False)
        runs = {"base": {"summary": summarize(results), "results": results, "meta": {"description": "x"}},
                "finetuned": {"summary": summarize(results), "results": results, "meta": {"description": "y"}}}
        md = markdown_report(runs)
        self.assertIn("| Answer accuracy (calculation questions) | 100.0% | 100.0% |", md)
        self.assertIn("Hand-written real-world questions", md)

    def test_load_sets_limit_is_stratified(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for it in Generator(9).build_test():
                    f.write(json.dumps(it) + "\n")
            items = load_sets([path], limit=100)
        self.assertLessEqual(abs(len(items) - 100), 8)
        self.assertGreaterEqual(len({it["type"] for it in items}), 6)


if __name__ == "__main__":
    unittest.main()
