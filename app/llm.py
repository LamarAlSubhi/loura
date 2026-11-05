"""The model interface, and a fake that misbehaves the way real models do."""

import asyncio
import json
import random
from typing import Protocol


class LLMError(Exception):
    pass


class LLMClient(Protocol):
    async def complete(self, system: str, user: str) -> str: ...


def _valid(category: str, priority: str, summary: str) -> str:
    return json.dumps({"category": category, "priority": priority, "summary": summary})


BROKEN_RESPONSES: dict[str, str | Exception] = {
    "prose": "Sure! This ticket looks like a billing issue with low priority.",
    "empty": "",
    "truncated_json": '{"category": "billing", "priority": "lo',
    "code_fence": "```json\n" + _valid("billing", "low", "Customer was charged twice.") + "\n```",
    "not_an_object": '["billing", "low"]',
    "category_out_of_range": _valid("sales", "low", "Customer wants to buy more seats."),
    "priority_out_of_range": _valid("billing", "urgent", "Customer was charged twice."),
    "wrong_case": _valid("Billing", "low", "Customer was charged twice."),
    "missing_field": json.dumps({"category": "billing", "priority": "low"}),
    "extra_field": json.dumps(
        {"category": "billing", "priority": "low", "summary": "A charge.", "refund_approved": True}
    ),
    "summary_not_a_string": json.dumps({"category": "billing", "priority": "low", "summary": 42}),
    "blank_summary": _valid("billing", "low", "   "),
    "two_sentences": _valid("billing", "low", "Customer was charged twice. They want a refund."),
    "multiline_summary": _valid("billing", "low", "Customer was charged twice\nand wants a refund"),
    "overlong_summary": _valid("billing", "low", "x" * 301),
    "timeout": TimeoutError(),
    "provider_error": LLMError("503 from provider"),
}

# First match wins.
CATEGORY_KEYWORDS = {
    "technical": ("error", "http 500", "500s", "api", "export", "upload", "timeout"),
    "billing": ("charge", "refund", "invoice"),
    "account": ("password", "log in", "account"),
}
LOW_PRIORITY_KEYWORDS = ("not urgent", "nice to have")
HIGH_PRIORITY_KEYWORDS = ("production", "blocking", "not fixed")
SUMMARIES = {
    "technical": "Customer is reporting a technical problem.",
    "billing": "Customer has a question about billing.",
    "account": "Customer needs help with their account.",
    "other": "Customer sent a request that fits no specific category.",
}


class FakeLLMClient:
    """Keyword-based answers, some broken. Says nothing about real-model accuracy."""

    def __init__(self, failure_rate: float = 0.2, seed: int | None = None, delay: float = 0.05):
        self._failure_rate = failure_rate
        self._random = random.Random(seed)
        self._delay = delay

    async def complete(self, system: str, user: str) -> str:
        await asyncio.sleep(self._delay)
        if self._random.random() < self._failure_rate:
            broken = self._random.choice(list(BROKEN_RESPONSES.values()))
            if isinstance(broken, Exception):
                raise broken
            return broken
        return self._answer(user)

    @staticmethod
    def _answer(user: str) -> str:
        ticket = json.loads(user[user.index("{") : user.rindex("}") + 1])
        text = f"{ticket['subject']} {ticket['body']}".lower()
        category = next(
            (name for name, words in CATEGORY_KEYWORDS.items() if any(w in text for w in words)),
            "other",
        )
        if category == "other" or any(word in text for word in LOW_PRIORITY_KEYWORDS):
            priority = "low"
        elif any(word in text for word in HIGH_PRIORITY_KEYWORDS):
            priority = "high"
        else:
            priority = "medium"
        return _valid(category, priority, SUMMARIES[category])
