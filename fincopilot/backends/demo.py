"""Demo mode: replays recorded model replies for the example questions so you can try
the app before installing a model. Calculations still run for real in Python;
only the language-model step is canned. Any other question gets a note explaining that.
"""
from __future__ import annotations

from typing import Sequence

from ..responses import clarification_text
from ..tools import get_tool, to_json
from .base import Backend, Message

EXAMPLES = {
    "If I invest $10,000 at 7% for 10 years, how much will I have?":
        to_json({"name": "compound_interest", "parameters": {"principal": 10000, "annual_rate": 7, "years": 10}}),
    "What's the monthly payment on a $35k car loan, 72 months at 7.9%?":
        to_json({"name": "loan_payment", "parameters": {"principal": 35000, "annual_rate": 7.9, "term_months": 72}}),
    "I have $8,750 on a card at 27.24% APR. If I pay $300 a month, how long until it's paid off?":
        to_json({"name": "loan_payoff_time", "parameters": {"balance": 8750, "annual_rate": 27.24, "monthly_payment": 300}}),
    "How much do I need to save each month to have $1 million in 35 years at 7%?":
        to_json({"name": "savings_goal_monthly", "parameters": {"target_amount": 1000000, "years": 35, "annual_rate": 7}}),
    "What would my monthly mortgage payment be on a $400,000 house?":
        clarification_text(get_tool("mortgage_payment"), ["annual_rate", "years"]),
    "Should I buy Nvidia stock right now?":
        "I can't predict where markets or individual investments are headed, and I don't recommend specific "
        "investments. What I can do is run the numbers for you: projecting growth at different return rates, "
        "comparing fees, or working out what you'd need to save for a goal.",
}

# One example first uses a wrong field name, to show the validator rejecting it and the
# model correcting itself after seeing the real error message.
FIRST_TRY = {
    "What's the monthly payment on a $35k car loan, 72 months at 7.9%?":
        to_json({"name": "loan_payment", "parameters": {"principal": 35000, "rate": 7.9, "term_months": 72}}),
}

UNKNOWN = ("Demo mode only knows the example questions, because no language model is connected. "
           "Start the app with a real model (for example --backend ollama) to ask anything.")


class DemoBackend(Backend):
    name = "demo"

    def generate(self, messages: Sequence[Message], max_new_tokens: int = 384) -> str:
        last_user = next(m["content"] for m in reversed(messages) if m["role"] == "user").strip()
        got_feedback = messages[-1]["role"] == "tool"
        if last_user in FIRST_TRY and not got_feedback:
            return FIRST_TRY[last_user]
        return EXAMPLES.get(last_user, UNKNOWN)
