"""Check a running Financial Copilot web app end to end.

    python deploy/smoke_test.py https://<your-workspace>--fincopilot-web.modal.run
    python deploy/smoke_test.py http://localhost:7860

Exits with status 1 if anything fails. The first question can take a couple of minutes
if the model server is waking up.
"""
from __future__ import annotations

import sys
import time

import requests

QUESTION = "If I invest $10,000 at 7% for 10 years, how much will I have?"
EXPECTED = "19,671.51"


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    base = args[0].rstrip("/")
    try:
        health = requests.get(f"{base}/healthz", timeout=30)
        health.raise_for_status()
        info = requests.get(f"{base}/api/info", timeout=30).json()
        print(f"OK  web app is up ({info['model']})")
        start = time.time()
        r = requests.post(f"{base}/api/chat", json={"message": QUESTION, "history": []}, timeout=600)
        data = r.json()
    except requests.RequestException as exc:
        print(f"FAIL  could not reach {base}: {exc}")
        return 1
    if r.status_code != 200:
        print(f"FAIL  {r.status_code}: {data.get('error')}")
        return 1
    print(f"OK  answered in {time.time() - start:.0f}s using {data.get('tool')}")
    for step in data["steps"]:
        print(f"    {step['stage']:<10} {step['status']:<7} {step['detail']}")
    print(f"    reply: {data['text']}")
    if EXPECTED not in data["text"]:
        print(f"FAIL  expected the reply to contain {EXPECTED}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
