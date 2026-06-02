"""Tests for TurnInstrumentation field additions + telemetry payload shape."""

from __future__ import annotations

from prog_strength_agent.telemetry import TurnInstrumentation, _build_turn_payload


def test_payload_includes_intent_fields():
    t = TurnInstrumentation.new(user_id="u-1", session_id="s-1")
    t.routed_tier = "simple"
    t.router_model = "claude-haiku-4-5-20251001"
    t.model = "claude-haiku-4-5-20251001"
    t.completion_reason = "end_turn"
    t.intent = "log_nutrition"
    t.intent_prefetch_duration_ms = 87
    t.intent_prefetch_failed = False

    body = _build_turn_payload(t)
    assert body["intent"] == "log_nutrition"
    assert body["intent_prefetch_duration_ms"] == 87
    assert body["intent_prefetch_failed"] is False
