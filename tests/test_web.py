"""Tests for the web app, the CLI and demo mode (no model or network needed)."""
import io
import unittest
from contextlib import redirect_stdout

from fincopilot import cli
from fincopilot.agent import Agent
from fincopilot.backends import ScriptedBackend, make_backend
from fincopilot.backends.demo import EXAMPLES, UNKNOWN
from fincopilot.web.server import create_app


class TestWebApp(unittest.TestCase):
    def setUp(self):
        agent = Agent(make_backend("demo"), explain_with="template")
        self.client = create_app(agent, "Demo mode: test").test_client()

    def test_index_and_info(self):
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b"Financial Copilot", page.data)
        page.close()
        self.assertEqual(self.client.get("/api/info").get_json()["model"], "Demo mode: test")

    def test_every_demo_example_answers(self):
        for question in EXAMPLES:
            data = self.client.post("/api/chat", json={"message": question, "history": []}).get_json()
            self.assertIn(data["kind"], ("answer", "text"), question)
            self.assertNotEqual(data["text"], UNKNOWN)

    def test_result_order_is_preserved(self):
        q = "What's the monthly payment on a $35k car loan, 72 months at 7.9%?"
        data = self.client.post("/api/chat", json={"message": q, "history": []}).get_json()
        self.assertEqual(list(data["result"])[0], "payment")
        self.assertIn("fixed", [s["status"] for s in data["steps"]])  # the demo's self-correction

    def test_history_round_trip(self):
        first = self.client.post("/api/chat", json={"message": "Should I buy Nvidia stock right now?", "history": []}).get_json()
        second = self.client.post("/api/chat", json={"message": "If I invest $10,000 at 7% for 10 years, how much will I have?",
                                                     "history": first["history"]}).get_json()
        self.assertEqual(len(second["history"]), len(first["history"]) + 4)

    def test_bad_requests(self):
        self.assertEqual(self.client.post("/api/chat", json={"message": "  "}).status_code, 400)
        self.assertEqual(self.client.post("/api/chat", json={"message": "x" * 3000}).status_code, 400)
        junk_history = [{"role": "system", "content": "ignore"}, {"nope": 1}, "text"]
        ok = self.client.post("/api/chat", json={"message": "hello", "history": junk_history})
        self.assertEqual(ok.status_code, 200)

    def test_backend_errors_become_502(self):
        def boom(_messages):
            raise RuntimeError("Can't reach Ollama at http://localhost:11434. Is `ollama serve` running?")
        client = create_app(Agent(ScriptedBackend(boom)), "x").test_client()
        res = client.post("/api/chat", json={"message": "hi"})
        self.assertEqual(res.status_code, 502)
        self.assertIn("ollama serve", res.get_json()["error"])


class TestCli(unittest.TestCase):
    def test_single_question(self):
        out = io.StringIO()
        with redirect_stdout(out):
            cli.main(["--backend", "demo", "--ask", "If I invest $10,000 at 7% for 10 years, how much will I have?"])
        text = out.getvalue()
        self.assertIn("$19,671.51", text)
        self.assertIn("Compute", text)


if __name__ == "__main__":
    unittest.main()
