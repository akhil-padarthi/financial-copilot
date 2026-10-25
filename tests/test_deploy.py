"""Deployment tests: env-var configuration, rate limits, the OpenAI-compatible backend
over real HTTP (against a fake vLLM-style server), and the smoke-test script."""
import io
import unittest.mock
import json
import os
import threading
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from werkzeug.serving import make_server

from fincopilot.backends.openai_compat import OpenAICompatBackend
from fincopilot.prompts import BOS
from fincopilot.web.server import RateLimiter, create_app_from_env, parse_limit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALL = '{"name": "compound_interest", "parameters": {"principal": 10000, "annual_rate": 7, "years": 10}}'


class FakeVLLM(BaseHTTPRequestHandler):
    """Mimics vLLM's /v1/completions and /v1/models, including the API-key check."""
    requests_seen = []
    api_key = "secret-key"

    def log_message(self, *args):
        pass

    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _authorized(self):
        return self.headers.get("Authorization") == f"Bearer {self.api_key}"

    def do_GET(self):
        FakeVLLM.requests_seen.append(("GET", self.path, None))
        if not self._authorized():
            return self._send(401, {"error": "Unauthorized"})
        self._send(200, {"data": [{"id": "base"}, {"id": "fincopilot"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeVLLM.requests_seen.append(("POST", self.path, body))
        if not self._authorized():
            return self._send(401, {"error": "Unauthorized"})
        if body.get("model") != "fincopilot":
            return self._send(404, {"error": "model not found"})
        explaining = body["prompt"].rstrip().endswith("assistant<|end_header_id|>") and "ipython" in body["prompt"]
        text = "You'd have $19,671.51 after 10 years." if explaining else CALL
        self._send(200, {"choices": [{"text": text}]})


class TestOpenAIBackendOverHTTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeVLLM)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}/v1"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        FakeVLLM.requests_seen.clear()

    def env(self, **extra):
        base = {"FINCOPILOT_BACKEND": "openai", "FINCOPILOT_HOST": self.url, "FINCOPILOT_MODEL": "fincopilot",
                "FINCOPILOT_API_KEY": "secret-key", "FINCOPILOT_LABEL": "Fine-tuned Llama 3.1 8B"}
        base.update(extra)
        return base

    def test_end_to_end_like_modal(self):
        with unittest.mock.patch.dict(os.environ, {"FINCOPILOT_API_KEY": "secret-key"}):
            client = create_app_from_env(self.env()).test_client()
            info = client.get("/api/info").get_json()
            self.assertEqual(info["model"], "Fine-tuned Llama 3.1 8B")
            data = client.post("/api/chat", json={"message": "If I invest $10,000 at 7% for 10 years?"}).get_json()
        self.assertEqual(data["kind"], "answer")
        self.assertEqual(data["result"]["future_value"], 19671.51)
        self.assertEqual(data["explanation_source"], "model")
        posts = [b for m, p, b in FakeVLLM.requests_seen if m == "POST"]
        self.assertEqual(len(posts), 2)  # tool call + explanation
        for body in posts:
            self.assertFalse(body["prompt"].startswith(BOS), "double BOS: the server adds its own")
            self.assertTrue(body["prompt"].startswith("<|start_header_id|>system"))
            self.assertEqual(body["temperature"], 0)
            self.assertIn("<|eot_id|>", body["stop"])

    def test_page_load_wakes_the_model(self):
        client = create_app_from_env(self.env()).test_client()
        client.get("/api/info")
        for _ in range(50):
            if any(m == "GET" and p.endswith("/models") for m, p, _ in FakeVLLM.requests_seen):
                break
            threading.Event().wait(0.05)
        self.assertTrue(any(m == "GET" and p.endswith("/models") for m, p, _ in FakeVLLM.requests_seen))

    def test_wrong_key_and_wrong_model_give_clear_errors(self):
        client = create_app_from_env(self.env(FINCOPILOT_API_KEY="wrong")).test_client()
        with unittest.mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FINCOPILOT_API_KEY", None)
            os.environ.pop("OPENAI_API_KEY", None)
            backend = OpenAICompatBackend(self.url, "fincopilot", api_key="wrong")
            with self.assertRaises(RuntimeError) as ctx:
                backend.generate([{"role": "user", "content": "hi"}])
            self.assertIn("API key", str(ctx.exception))
            backend = OpenAICompatBackend(self.url, "nope", api_key="secret-key")
            with self.assertRaises(RuntimeError) as ctx:
                backend.generate([{"role": "user", "content": "hi"}])
            self.assertIn("no model named", str(ctx.exception))
        self.assertIsNotNone(client)

    def test_unreachable_server(self):
        backend = OpenAICompatBackend("http://127.0.0.1:9/v1", "fincopilot", api_key="k", timeout=2)
        with self.assertRaises(RuntimeError) as ctx:
            backend.generate([{"role": "user", "content": "hi"}])
        self.assertIn("Can't reach", str(ctx.exception))


class TestRateLimits(unittest.TestCase):
    def test_sliding_window(self):
        rl = RateLimiter(2, 10)
        self.assertEqual(rl.check("a", now=0)[0], True)
        self.assertEqual(rl.check("a", now=1)[0], True)
        ok, wait = rl.check("a", now=2)
        self.assertFalse(ok)
        self.assertEqual(wait, 9)
        self.assertTrue(rl.check("b", now=2)[0])     # other visitors unaffected
        self.assertTrue(rl.check("a", now=10.5)[0])  # window has moved on

    def test_parse_limit(self):
        self.assertIsNone(parse_limit(None))
        self.assertIsNone(parse_limit("0"))
        rl = parse_limit("30/600")
        self.assertEqual((rl.max_requests, rl.window), (30, 600))

    def test_per_visitor_limit_returns_429(self):
        client = create_app_from_env({"FINCOPILOT_RATE_LIMIT": "2/600"}).test_client()
        q = {"message": "Should I buy Nvidia stock right now?"}
        codes = [client.post("/api/chat", json=q, headers={"X-Forwarded-For": "1.2.3.4"}).status_code for _ in range(3)]
        self.assertEqual(codes, [200, 200, 429])
        other = client.post("/api/chat", json=q, headers={"X-Forwarded-For": "5.6.7.8"})
        self.assertEqual(other.status_code, 200)
        self.assertIn("Try again in", client.post("/api/chat", json=q, headers={"X-Forwarded-For": "1.2.3.4"}).get_json()["error"])

    def test_daily_limit(self):
        client = create_app_from_env({"FINCOPILOT_DAILY_LIMIT": "1"}).test_client()
        q = {"message": "Should I buy Nvidia stock right now?"}
        self.assertEqual(client.post("/api/chat", json=q).status_code, 200)
        res = client.post("/api/chat", json=q, headers={"X-Forwarded-For": "9.9.9.9"})
        self.assertEqual(res.status_code, 429)
        self.assertIn("daily", res.get_json()["error"])


class TestEnvConfigAndSmokeTest(unittest.TestCase):
    def test_defaults_to_demo(self):
        client = create_app_from_env({}).test_client()
        self.assertEqual(client.get("/healthz").get_json(), {"ok": True})
        self.assertTrue(client.get("/api/info").get_json()["model"].startswith("Demo mode"))

    def test_smoke_test_script_against_live_server(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("smoke_test", os.path.join(ROOT, "deploy", "smoke_test.py"))
        smoke = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(smoke)
        # The demo backend only knows its example questions, so ask the one it knows.
        smoke.QUESTION = "If I invest $10,000 at 7% for 10 years, how much will I have?"
        server = make_server("127.0.0.1", 0, create_app_from_env({}), threaded=True)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        out = io.StringIO()
        try:
            with redirect_stdout(out):
                code = smoke.main([f"http://127.0.0.1:{server.server_port}"])
        finally:
            server.shutdown()
        self.assertEqual(code, 0, out.getvalue())
        self.assertIn("PASS", out.getvalue())

    def test_modal_app_is_consistent_with_the_docs(self):
        with open(os.path.join(ROOT, "deploy", "modal_app.py"), encoding="utf-8") as f:
            src = f.read()
        compile(src, "modal_app.py", "exec")
        for needle in ('"vllm==0.21.0"', '"--enable-lora"', "fincopilot={ADAPTER_PATH}", '"--api-key"',
                       "fincopilot-adapters", "fincopilot-secrets", "add_local_dir", "create_app_from_env"):
            self.assertIn(needle, src)


if __name__ == "__main__":
    unittest.main()
