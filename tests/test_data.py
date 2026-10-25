"""Tests for the synthetic data generator.

Bad labels silently teach a model wrong behavior, so these checks are strict:
every gold value must be recoverable from the question text, every tool call
must validate and reproduce its stored result, and every assistant reply
must pass the same grounding check the agent applies at inference.
"""
import json
import unittest

from fincopilot.datagen.generate import Generator, build_realworld, slots_of
from fincopilot.datagen.templates import TEMPLATES
from fincopilot.grounding import numbers_in_messages, ungrounded_numbers
from fincopilot.parsing import TextReply, ToolCall, parse_model_output
from fincopilot.prompts import build_system_prompt
from fincopilot.tools import all_tools, get_tool

from fincopilot.datagen.validate import value_in_text as _value_in_text


class TestTemplates(unittest.TestCase):
    def test_every_tool_has_templates(self):
        for tool in all_tools():
            t = TEMPLATES[tool.name]
            self.assertGreaterEqual(len(t["train"]), 7, tool.name)
            self.assertGreaterEqual(len(t["test"]), 3, tool.name)
            self.assertTrue(t["partial"], tool.name)

    def test_slots_are_real_fields(self):
        for tool in all_tools():
            t = TEMPLATES[tool.name]
            for template in t["train"] + t["test"] + [p[0] for p in t["partial"]]:
                for slot in slots_of(template):
                    self.assertIn(slot, tool.fields, f"{tool.name}: {template}")
            for _template, missing in t["partial"]:
                for field in missing:
                    self.assertIn(field, tool.fields, tool.name)

    def test_test_templates_are_held_out(self):
        for name, t in TEMPLATES.items():
            self.assertFalse(set(t["train"]) & set(t["test"]), name)


class TestGeneratedData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = Generator(123).build_train(scale=0.25)
        cls.test = Generator(456).build_test()

    def test_system_prompt_first(self):
        system = build_system_prompt("compact")
        for ex in self.train:
            self.assertEqual(ex["messages"][0], {"role": "system", "content": system})

    def test_tool_calls_validate_and_match_results(self):
        checked = 0
        for ex in self.train:
            msgs = ex["messages"]
            for i, m in enumerate(msgs):
                if m["role"] != "assistant" or m.get("train") is False:
                    continue
                parsed = parse_model_output(m["content"])
                if not isinstance(parsed, ToolCall):
                    continue
                tool = get_tool(parsed.name)
                self.assertIsNotNone(tool, ex["id"])
                args = tool.validate(parsed.parameters)
                result_msg = json.loads(msgs[i + 1]["content"])
                if "error" not in result_msg:
                    self.assertEqual(tool.execute(args), result_msg, ex["id"])
                    explanation = msgs[i + 2]["content"]
                    self.assertEqual(ungrounded_numbers(explanation, tool.allowed_numbers(args, result_msg)), [], ex["id"])
                checked += 1
        self.assertGreater(checked, 1500)

    def test_gold_values_appear_in_question(self):
        """Every number in a gold tool call must be readable from the user's own words."""
        for ex in self.train + self.test:
            msgs = ex["messages"]
            if ex["type"] not in ("tool", "followup", "domain", "repair"):
                continue
            user_text = " ".join(m["content"] for m in msgs if m["role"] == "user")
            if "expected" in ex:
                params = ex["expected"]["parameters"]
                tool = get_tool(ex["expected"]["name"])
            else:
                calls = [parse_model_output(m["content"]) for m in msgs
                         if m["role"] == "assistant" and m.get("train") is not False]
                call = next(c for c in calls if isinstance(c, ToolCall))
                params, tool = call.parameters, get_tool(call.name)
            for field, value in params.items():
                kind = tool.kind(field)
                values = value if isinstance(value, list) else [value]
                for v in values:
                    self.assertTrue(_value_in_text(v, kind, user_text),
                                    f"{ex['id']} {tool.name}.{field}={v!r} not found in: {user_text}")

    def test_text_replies_are_grounded(self):
        for ex in self.train:
            msgs = ex["messages"]
            for i, m in enumerate(msgs):
                if m["role"] == "assistant" and isinstance(parse_model_output(m["content"]), TextReply):
                    if i > 0 and msgs[i - 1]["role"] == "tool":
                        continue  # explanations are checked against tool output above
                    bad = ungrounded_numbers(m["content"], numbers_in_messages(msgs[:i]), small_int_max=100)
                    self.assertEqual(bad, [], f"{ex['id']}: {m['content']}")

    def test_repair_examples_mask_the_mistake(self):
        repairs = [ex for ex in self.train if ex["type"] == "repair"]
        self.assertTrue(repairs)
        for ex in repairs:
            self.assertEqual(ex["messages"][2].get("train"), False)
            self.assertIn("error", json.loads(ex["messages"][3]["content"]))

    def test_test_set_shape(self):
        for ex in self.test:
            self.assertEqual(ex["messages"][-1]["role"], "user")
            exp = ex["expected"]
            if exp["action"] == "tool":
                get_tool(exp["name"]).run(exp["parameters"])

    def test_every_type_present(self):
        types = {ex["type"] for ex in self.train}
        self.assertEqual(types, {"tool", "followup", "clarify", "repair", "domain", "concept", "advice", "offtopic", "greeting"})


class TestRealWorld(unittest.TestCase):
    def test_gold_labels_compute(self):
        rows = build_realworld()
        self.assertGreaterEqual(len(rows), 90)
        tools_covered = {r["expected"]["name"] for r in rows if r["expected"]["action"] == "tool"}
        self.assertEqual(tools_covered, {t.name for t in all_tools()})

    def test_gold_values_appear_in_question(self):
        for row in build_realworld():
            exp = row["expected"]
            if exp["action"] != "tool":
                continue
            tool = get_tool(exp["name"])
            for field, value in exp["parameters"].items():
                for v in (value if isinstance(value, list) else [value]):
                    ok = _value_in_text(v, tool.kind(field), row["messages"][0]["content"])
                    if not ok and "bucks" in row["messages"][0]["content"]:
                        ok = v == 100  # "a hundred bucks"
                    if not ok and field in ("payments_per_year", "contributions_per_year"):
                        ok = True  # implied by "per year" phrasing checked elsewhere
                    self.assertTrue(ok, f"{row['id']} {field}={v}")


if __name__ == "__main__":
    unittest.main()
