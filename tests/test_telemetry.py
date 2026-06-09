"""Tests for TurnInstrumentation field additions + telemetry payload shape."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import respx
from httpx import Response

from prog_strength_agent.telemetry import (
    MessageRecord,
    SpeakCallRecord,
    TelemetryClient,
    ToolCallRecord,
    TurnInstrumentation,
    _build_speak_payload,
    _build_turn_payload,
)


def _speak_record() -> SpeakCallRecord:
    now = datetime.now(UTC)
    return SpeakCallRecord(
        id="speak-1",
        user_id="u-1",
        session_id="s-1",
        model="gpt-4o-mini-tts",
        chars=184,
        voice="cedar",
        started_at=now,
        ended_at=now,
        error=None,
    )


def test_speak_payload_shape():
    body = _build_speak_payload(_speak_record())
    assert body["id"] == "speak-1"
    assert body["user_id"] == "u-1"
    assert body["session_id"] == "s-1"
    assert body["model"] == "gpt-4o-mini-tts"
    assert body["chars"] == 184
    assert body["voice"] == "cedar"
    assert body["error"] is None
    # Timestamps serialized as RFC3339 with a trailing Z.
    assert body["started_at"].endswith("Z")
    assert body["ended_at"].endswith("Z")


async def test_record_speak_posts_once():
    client = TelemetryClient("http://api:8080")
    paths: list[str] = []
    bodies: list[dict] = []

    def record(request):
        paths.append(request.url.path)
        import json as _json

        bodies.append(_json.loads(request.content))
        return Response(204)

    with respx.mock(base_url="http://api:8080") as mock:
        mock.post("/internal/telemetry/speak").mock(side_effect=record)
        client.record_speak(_speak_record())
        await asyncio.sleep(0.05)

    await client.aclose()
    assert paths == ["/internal/telemetry/speak"]
    assert bodies[0]["chars"] == 184


async def test_record_speak_swallows_failure():
    client = TelemetryClient("http://api:8080")
    with respx.mock(base_url="http://api:8080") as mock:
        mock.post("/internal/telemetry/speak").mock(return_value=Response(500))
        # Fire-and-forget; the failure must not propagate.
        client.record_speak(_speak_record())
        await asyncio.sleep(0.05)
    await client.aclose()


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
            started_at=datetime.now(UTC),
            ended_at=datetime.now(UTC),
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
