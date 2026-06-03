"""Tests for TurnInstrumentation field additions + telemetry payload shape."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import respx
from httpx import Response

from prog_strength_agent.telemetry import (
    MessageRecord,
    TelemetryClient,
    ToolCallRecord,
    TurnInstrumentation,
    _build_turn_payload,
)


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


async def test_record_turn_serializes_turn_before_dependents():
    """Regression: child telemetry POSTs (/tool-calls, /messages) must
    not race the parent /turns POST. agent_tool_calls and agent_messages
    FK to agent_turns.id; firing all three concurrently let children
    arrive at the API before the parent committed and 500'd with
    FOREIGN KEY constraint failed, silently dropping the child rows.
    """
    client = TelemetryClient("http://api:8080")

    t = TurnInstrumentation.new(user_id="u-1", session_id="s-1")
    t.routed_tier = "simple"
    t.router_model = "claude-haiku-4-5-20251001"
    t.model = "claude-haiku-4-5-20251001"
    t.completion_reason = "end_turn"
    t.messages.append(MessageRecord(role="user", content="hello"))
    t.tool_calls.append(
        ToolCallRecord(
            tool_name="get_user",
            arguments_json="{}",
            result_summary="{}",
            latency_ms=10,
            error=None,
            started_at=datetime.now(timezone.utc),
            ended_at=datetime.now(timezone.utc),
        )
    )

    call_order: list[str] = []

    def record(request):
        call_order.append(request.url.path)
        return Response(201, json={})

    with respx.mock(base_url="http://api:8080") as mock:
        mock.post("/internal/telemetry/turns").mock(side_effect=record)
        mock.post("/internal/telemetry/tool-calls").mock(side_effect=record)
        mock.post("/internal/telemetry/messages").mock(side_effect=record)

        client.record_turn(t)
        # record_turn is fire-and-forget; wait for the scheduled task
        # to drain before asserting on call_order.
        await asyncio.sleep(0.05)

    await client.aclose()

    assert call_order, "no telemetry POSTs were made"
    assert call_order[0] == "/internal/telemetry/turns", (
        "/turns must be POSTed first to commit the parent row before "
        f"any FK-referencing child POST; got: {call_order}"
    )
    assert set(call_order[1:]) == {
        "/internal/telemetry/tool-calls",
        "/internal/telemetry/messages",
    }, f"all three child POSTs must follow; got: {call_order}"
