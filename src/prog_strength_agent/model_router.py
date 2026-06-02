"""Structured-output classifier for incoming chat requests.

Calls Haiku with a forced tool_use that returns
{tier: simple|complex, intent: <known intent>}. Same cost as the
previous one-word-output router (~$0.0001/call, ~500ms p50).

Failure mode: any exception or malformed output falls back to
RouterDecision(tier="simple", intent="general") — the cheapest+safest
default. The user gets a possibly-degraded response rather than a 500.

See prog-strength-docs/sows/intent-driven-context-enrichment.md.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from anthropic import AsyncAnthropic

from prog_strength_agent.intents import KNOWN_INTENTS
from prog_strength_agent.telemetry import TurnInstrumentation, now_ms

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RouterDecision:
    tier: str
    intent: str


_ROUTER_TOOL = {
    "name": "classify_request",
    "description": "Classify the user's request by intent and required model tier.",
    "input_schema": {
        "type": "object",
        "properties": {
            "tier": {"type": "string", "enum": ["simple", "complex"]},
            "intent": {"type": "string", "enum": list(KNOWN_INTENTS)},
        },
        "required": ["tier", "intent"],
    },
}


ROUTER_SYSTEM_PROMPT = """\
You are a routing classifier for the Prog Strength training assistant.
Output BOTH a tier and an intent for the user's latest request via
the classify_request tool.

tier:
- simple — CRUD or lookup (logging a workout, listing exercises).
- complex — multi-step analysis, planning, trend reasoning.

intent:
- log_nutrition — the user wants to record food/drink they consumed.
- log_workout — the user is reporting a completed workout.
- log_bodyweight — the user is logging a bodyweight reading.
- analyze_progress — the user wants insight, trends, or planning advice.
- general — anything else (greeting, lookup, off-topic).

When uncertain on intent, pick "general". When uncertain on tier,
pick "simple". Both err on the side of cheap and safe.
"""


_FALLBACK = RouterDecision(tier="simple", intent="general")


class ModelRouter:
    def __init__(self, client: AsyncAnthropic, router_model: str):
        self.client = client
        self.router_model = router_model

    async def route(
        self,
        messages: list[dict[str, Any]],
        telemetry: TurnInstrumentation | None = None,
        prior_intent: str | None = None,
    ) -> RouterDecision:
        started = now_ms()
        user_text = _last_user_text(messages)
        if not user_text:
            _record(telemetry, self.router_model, now_ms() - started, _FALLBACK)
            return _FALLBACK

        hint = ""
        if prior_intent:
            hint = (
                f"\n\nPrevious classified intent for this conversation: "
                f"{prior_intent}. Use it as a hint when the latest message "
                f"is a short follow-up; switch when the user clearly pivots."
            )

        try:
            resp = await self.client.messages.create(
                model=self.router_model,
                max_tokens=200,
                system=ROUTER_SYSTEM_PROMPT,
                tools=[_ROUTER_TOOL],
                tool_choice={"type": "tool", "name": "classify_request"},
                messages=[
                    {"role": "user", "content": user_text + hint},
                ],
            )
        except Exception:
            log.exception("router classification call failed")
            _record(telemetry, self.router_model, now_ms() - started, _FALLBACK)
            return _FALLBACK

        decision = _parse_decision(resp)
        log.info(
            "router: tier=%s intent=%s (chars=%d, hint=%s)",
            decision.tier, decision.intent, len(user_text),
            "y" if prior_intent else "n",
        )
        _record(telemetry, self.router_model, now_ms() - started, decision)
        return decision


def _parse_decision(resp: Any) -> RouterDecision:
    for block in getattr(resp, "content", []) or []:
        if getattr(block, "type", None) != "tool_use":
            continue
        if getattr(block, "name", None) != "classify_request":
            continue
        payload = getattr(block, "input", None) or {}
        tier = payload.get("tier") if isinstance(payload, dict) else None
        intent = payload.get("intent") if isinstance(payload, dict) else None
        if tier not in ("simple", "complex"):
            return _FALLBACK
        if intent not in KNOWN_INTENTS:
            return _FALLBACK
        return RouterDecision(tier=tier, intent=intent)
    return _FALLBACK


def _record(
    telemetry: TurnInstrumentation | None,
    router_model: str,
    latency_ms: int,
    decision: RouterDecision,
) -> None:
    if telemetry is None:
        return
    telemetry.router_model = router_model
    telemetry.router_latency_ms = latency_ms
    telemetry.routed_tier = decision.tier
    telemetry.intent = decision.intent


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
            return " ".join(parts).strip()
        return ""
    return ""
