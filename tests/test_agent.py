"""Tests for parsing, grounding, prompt formatting and the agent loop.

The agent tests use ScriptedBackend, which plays the role of the language model
with canned outputs, so every branch (happy path, self-repair, clarification,
grounding fallback...) is exercised without a GPU.
"""
import json
import unittest

from fincopilot.agent import Agent
from fincopilot.backends import ScriptedBackend
from fincopilot.grounding import extract_numbers, ungrounded_numbers
from fincopilot.parsing import ParseError, TextReply, ToolCall, parse_model_output
from fincopilot.prompts import EOT, build_system_prompt, build_training_features, format_llama31
from fincopilot.tools import all_tools

CALL = '{"name": "compound_interest", "parameters": {"principal": 10000, "annual_rate": 7, "years": 10}}'
QUESTION = "If I invest $10,000 at 7% for 10 years, how much will I have?"


class TestParsing(unittest.TestCase):
    def test_plain_json(self):
        p = parse_model_output(CALL)
        self.assertIsInstance(p, ToolCall)
        self.assertEqual(p.name, "compound_interest")
        self.assertEqual(p.parameters["principal"], 10000)

    def test_variants(self):
        for text in (f"```json\n{CALL}\n```", f"<|python_tag|>{CALL}", f"Sure! {CALL} Let me know.",
                     CALL.replace('"parameters"', '"arguments"'),
                     "{'name': 'compound_interest', 'parameters': {'principal': 10000, 'annual_rate': 7, 'years': 10}}",
                     '{"name": "compound_interest", "parameters": "{\\"principal\\": 10000}"}'):
            self.assertIsInstance(parse_model_output(text), ToolCall, text)

    def test_other_llama_and_openai_formats(self):
        for text in ('<function=compound_interest>{"principal": 10000, "annual_rate": 7, "years": 10}</function>',
                     'I will call a tool. <function=compound_interest>{"principal": 10000, "annual_rate": 7, "years": 10}</function>',
                     '{"type": "function", "name": "compound_interest", "parameters": {"principal": 10000, "annual_rate": 7, "years": 10}}',
                     'Sure: {"type": "function", "name": "compound_interest", "parameters": {"principal": 10000, "annual_rate": 7, "years": 10}}',
                     '{"function": {"name": "compound_interest", "arguments": "{\\"principal\\": 10000, \\"annual_rate\\": 7, \\"years\\": 10}"}}',
                     '{"tool_calls": [{"type": "function", "function": {"name": "compound_interest", "arguments": {"principal": 10000, "annual_rate": 7, "years": 10}}}]}',
                     '[{"name": "compound_interest", "parameters": {"principal": 10000, "annual_rate": 7, "years": 10}}]'):
            p = parse_model_output(text)
            self.assertIsInstance(p, ToolCall, text)
            self.assertEqual((p.name, p.parameters["annual_rate"]), ("compound_interest", 7), text)

    def test_braces_inside_strings(self):
        p = parse_model_output('{"name": "x", "parameters": {"note": "a } brace"}}')
        self.assertEqual(p.parameters["note"], "a } brace")

    def test_text_reply(self):
        self.assertIsInstance(parse_model_output("What interest rate should I use?"), TextReply)

    def test_malformed(self):
        with self.assertRaises(ParseError):
            parse_model_output('{"name": "compound_interest", "parameters": {"principal": 10000')
        with self.assertRaises(ParseError):
            parse_model_output('{"parameters": {}}')


class TestGrounding(unittest.TestCase):
    def test_extract(self):
        values = [v for _, v, _ in extract_numbers("Pay $1,199.10 for 30 years, about 19.7k, -$5, 1.2 million")]
        self.assertEqual(values, [1199.10, 30, 19700, -5, 1_200_000])

    def test_exact_and_rounded_numbers_pass(self):
        allowed = [10000, 7, 10, 19671.51, 9671.51]
        for text in ("It grows to $19,671.51.", "That's about $19,672.", "Roughly $19.7k.", "About $20,000 in 10 years."):
            self.assertEqual(ungrounded_numbers(text, allowed), [], text)

    def test_wrong_numbers_fail(self):
        allowed = [10000, 7, 10, 19671.51, 9671.51]
        self.assertEqual(ungrounded_numbers("It grows to $19,500.", allowed), ["$19,500"])
        self.assertEqual(ungrounded_numbers("You'll have it by 2036.", allowed), ["2036"])
        self.assertEqual(ungrounded_numbers("About $25,000.", allowed), ["$25,000"])

    def test_product_names_ignored(self):
        self.assertEqual(ungrounded_numbers("Put it in your 401(k) or an S&P 500 fund.", []), [])

    def test_every_template_explanation_is_grounded(self):
        """The fallback explanations must themselves pass the check, for every tool."""
        from fincopilot.datagen.samplers import SAMPLERS
        import random
        rng = random.Random(0)
        for tool in all_tools():
            for _ in range(25):
                values = SAMPLERS[tool.name](rng)
                try:
                    run = tool.run(_drop_alternatives(tool, values, rng))
                except Exception:
                    continue
                bad = ungrounded_numbers(run.explanation, tool.allowed_numbers(run.args, run.result))
                self.assertEqual(bad, [], f"{tool.name}: {run.explanation}")


def _drop_alternatives(tool, values, rng):
    out = dict(values)
    for fields, _required in tool.one_of_groups():
        keep = rng.choice(fields)
        for f in fields:
            if f != keep:
                out.pop(f, None)
    return out


class TestPrompts(unittest.TestCase):
    def test_system_prompt_lists_every_tool(self):
        compact, full = build_system_prompt("compact"), build_system_prompt("full")
        for tool in all_tools():
            self.assertIn(tool.signature(), compact)
            self.assertIn(tool.name + ":", full)

    def test_llama31_format(self):
        text = format_llama31([{"role": "system", "content": "S"}, {"role": "user", "content": "Hi"},
                               {"role": "tool", "content": "{}"}])
        self.assertEqual(text, "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nS<|eot_id|>"
                               "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>"
                               "<|start_header_id|>ipython<|end_header_id|>\n\n{}<|eot_id|>"
                               "<|start_header_id|>assistant<|end_header_id|>\n\n")

    def test_loss_mask_covers_only_assistant_text(self):
        # A character-level fake tokenizer makes the mask easy to inspect.
        encode = lambda s: [ord(c) for c in s]
        msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"},
                {"role": "assistant", "content": "BAD CALL", "train": False},
                {"role": "tool", "content": "ERROR"},
                {"role": "assistant", "content": "GOOD CALL"}, {"role": "tool", "content": "RESULT"},
                {"role": "assistant", "content": "EXPLANATION"}]
        feats = build_training_features(msgs, encode)
        learned = "".join(chr(t) if t != -100 else "\x00" for t in feats["labels"])
        pieces = [p for p in learned.split("\x00") if p]
        self.assertEqual(pieces, ["GOOD CALL" + EOT, "EXPLANATION" + EOT])
        # The full input (not just labels) must equal the inference-time prompt format.
        full = "".join(chr(t) for t in feats["input_ids"])
        self.assertEqual(full, format_llama31([{k: v for k, v in m.items() if k != "train"} for m in msgs],
                                              add_generation_prompt=False))
        self.assertIsNone(build_training_features(msgs, encode, max_length=10))


class TestAgent(unittest.TestCase):
    def test_happy_path_with_model_explanation(self):
        backend = ScriptedBackend([CALL, "You'd have $19,671.51 after 10 years, earning $9,671.51 in interest."])
        reply = Agent(backend).chat([], QUESTION)
        self.assertEqual(reply.kind, "answer")
        self.assertEqual(reply.tool, "compound_interest")
        self.assertEqual(reply.result["future_value"], 19671.51)
        self.assertEqual(reply.explanation_source, "model")
        self.assertEqual([m["role"] for m in reply.history], ["user", "assistant", "tool", "assistant"])
        # The explanation prompt must contain the tool result, as in training.
        self.assertIn("19671.51", backend.calls[1][-1]["content"])

    def test_hallucinated_explanation_is_replaced(self):
        backend = ScriptedBackend([CALL, "You'd have about $21,000 by 2036."])
        reply = Agent(backend).chat([], QUESTION)
        self.assertEqual(reply.explanation_source, "template")
        self.assertIn("$19,671.51", reply.text)
        self.assertEqual(reply.steps[-1].status, "fixed")

    def test_template_mode_uses_one_model_call(self):
        reply = Agent(ScriptedBackend([CALL]), explain_with="template").chat([], QUESTION)
        self.assertEqual(reply.model_calls, 1)
        self.assertIn("$19,671.51", reply.text)

    def test_self_repair_of_bad_parameter_name(self):
        bad = '{"name": "compound_interest", "parameters": {"principal": 10000, "rate": 7, "years": 10}}'
        backend = ScriptedBackend([bad, CALL, "It grows to $19,671.51."])
        reply = Agent(backend).chat([], QUESTION)
        self.assertEqual(reply.kind, "answer")
        feedback = json.loads(backend.calls[1][-1]["content"])
        self.assertEqual(feedback["error"], "invalid_arguments")
        self.assertTrue(any("rate" in d for d in feedback["details"]))
        self.assertEqual([s.status for s in reply.steps if s.stage == "validate"], ["failed", "fixed"])

    def test_unknown_tool_then_repair(self):
        backend = ScriptedBackend(['{"name": "investment_calculator", "parameters": {}}', CALL, "It grows to $19,671.51."])
        self.assertEqual(Agent(backend).chat([], QUESTION).kind, "answer")

    def test_missing_value_becomes_a_question(self):
        partial = '{"name": "compound_interest", "parameters": {"principal": 10000, "annual_rate": 7}}'
        reply = Agent(ScriptedBackend([partial, partial])).chat([], "If I invest $10,000 at 7%, how much will I have?")
        self.assertEqual(reply.kind, "clarification")
        self.assertIn("the number of years", reply.text)
        self.assertEqual([m["role"] for m in reply.history], ["user", "assistant"])

    def test_clarification_follow_up_uses_history(self):
        agent = Agent(ScriptedBackend(["What interest rate and time period should I use?", CALL, "It grows to $19,671.51."]))
        first = agent.chat([], "If I invest $10,000, how much will I have?")
        self.assertEqual(first.kind, "text")
        second = agent.chat(first.history, "7% for 10 years")
        self.assertEqual(second.result["future_value"], 19671.51)
        self.assertEqual(len(second.history), 6)

    def test_mental_math_in_text_is_blocked(self):
        backend = ScriptedBackend(["You'll have about $19,600.", "Still $19,600."])
        reply = Agent(backend).chat([], QUESTION)
        self.assertEqual(reply.kind, "error")
        self.assertEqual(json.loads(backend.calls[1][-1]["content"])["error"], "calculation_required")

    def test_mental_math_then_repair(self):
        backend = ScriptedBackend(["You'll have about $19,600.", CALL, "It grows to $19,671.51."])
        self.assertEqual(Agent(backend).chat([], QUESTION).kind, "answer")

    def test_concept_answer_passes(self):
        text = "APR is the yearly rate before compounding; APY includes the effect of compounding."
        reply = Agent(ScriptedBackend([text])).chat([], "What's the difference between APR and APY?")
        self.assertEqual((reply.kind, reply.text), ("text", text))

    def test_domain_error(self):
        call = '{"name": "loan_payoff_time", "parameters": {"balance": 10000, "annual_rate": 24, "monthly_payment": 150}}'
        reply = Agent(ScriptedBackend([call])).chat([], "How long to pay off $10,000 at 24% paying $150 a month?")
        self.assertEqual(reply.kind, "cannot_compute")
        self.assertIn("never go down", reply.text)

    def test_parse_error_then_give_up(self):
        reply = Agent(ScriptedBackend(['{"name": "roi", "param', '{"name": "roi", "param'])).chat([], "roi?")
        self.assertEqual(reply.kind, "error")


if __name__ == "__main__":
    unittest.main()
