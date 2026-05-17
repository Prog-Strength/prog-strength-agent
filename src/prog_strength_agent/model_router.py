"""Lightweight tier classifier for incoming chat requests.

Calls Haiku with a tiny system prompt to decide whether the user's
latest message can be served by the cheap simple-tier model
(CRUD-shaped requests: log this workout, list my workouts, look up
an exercise) or whether it needs the more expensive complex-tier
model (analysis, progression tracking, planning).

Cost: one Haiku classification call per /chat request, max_tokens=10.
Roughly $0.0001 per call — basically free relative to the savings.
Latency: ~300–700ms; that's the price of the tier decision.

Failure mode: any exception in classification falls back to the
simple tier (Haiku-default per the project's routing policy). The
user gets a possibly-degraded response rather than a 500.
"""

import logging
from typing import Any

from anthropic import AsyncAnthropic

log = logging.getLogger(__name__)

ROUTER_SYSTEM_PROMPT = """\
You are a routing classifier for the Prog Strength training assistant.
Given the user's most recent message, decide whether the request needs a
big reasoning model or whether a small fast model can handle it.

Respond with EXACTLY one word: "simple" or "complex". No punctuation.

simple — straightforward CRUD or lookup tasks. Examples:
  - "log my workout from this morning: bench 5x5 at 185"
  - "what chest exercises are in the catalog?"
  - "show me my workouts from last week"
  - "did I do squats on Monday?"

complex — multi-step analysis, trend reasoning, planning, or
  recommendations that need to compose information across multiple
  workouts. Examples:
  - "how has my bench progressed over the last 3 months?"
  - "am I training enough volume for legs?"
  - "what should I program next month?"
  - "compare my squat sets this week vs last week"

When uncertain, prefer "simple". Cost matters.
"""


class ModelRouter:
    """Classifies chat requests into a simple/complex tier.

    Caller maps the tier string to a ModelHarness; the router itself
    doesn't know about harnesses.
    """

    def __init__(
        self,
        client: AsyncAnthropic,
        router_model: str,
    ):
        self.client = client
        self.router_model = router_model

    async def route(self, messages: list[dict[str, Any]]) -> str:
        """Return "simple" or "complex" for the conversation.

        Looks at the latest user turn only — keeps the prompt short
        and the decision focused on what the user just asked. The
        full conversation context is left to the chosen harness to
        handle on the actual response call.
        """
        text = _last_user_text(messages)
        if not text:
            # No content to classify (empty message or no user turns).
            # Default to simple — nothing to spend Sonnet tokens on.
            return "simple"

        try:
            resp = await self.client.messages.create(
                model=self.router_model,
                max_tokens=10,
                system=ROUTER_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": text}],
            )
        except Exception:
            log.exception("router classification call failed")
            return "simple"

        # The classifier should answer with one word; tolerate trailing
        # punctuation/whitespace and case differences. Anything that
        # contains "complex" routes complex; otherwise simple. Defaults
        # to simple matches the Haiku-default routing policy.
        decision = ""
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                decision = block.text.strip().lower()
                break

        if "complex" in decision:
            log.info("router: complex (text=%r)", text[:80])
            return "complex"
        log.info("router: simple (text=%r)", text[:80])
        return "simple"


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    """Pull the most recent user message's text content.

    Messages can have content as a plain string (typical user turns)
    or as a list of typed blocks (assistant turns / tool_result
    follow-ups). The router only ever wants text — anything else is
    flattened out.
    """
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Reassemble any text blocks; skip tool_result and other
            # non-text shapes. Tool results in the user role come from
            # the agent's own loop, not the human user.
            parts: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
            return " ".join(parts).strip()
        return ""
    return ""
