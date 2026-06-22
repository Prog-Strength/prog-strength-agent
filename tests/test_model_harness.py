"""Tests for the intent-aware additions to ModelHarness.

The full SSE tool-use loop is covered by integration tests in CI; this
file pins the small surface that intent wiring touches: that
stream_chat composes the system prompt from base + intent rules + data
before invoking the model.
"""

from __future__ import annotations

import pytest

from prog_strength_agent.prompt import compose_system_prompt


def test_compose_system_prompt_orders_sections():
    out = compose_system_prompt(base="BASE", rules="RULES", data="DATA")
    assert out.index("BASE") < out.index("RULES") < out.index("DATA")


@pytest.mark.asyncio
async def test_harness_uses_intent_registry_to_compose_prompt(monkeypatch):
    """Stub IntentRegistry.run to return known blocks and assert the
    final system prompt the harness would have sent.
    """
    from prog_strength_agent import model_harness as mh
    from prog_strength_agent.intents import IntentRegistry

    async def fake_run(cls, intent, session, client_timezone=None):
        return "RULES_BLOCK", "DATA_BLOCK", False

    monkeypatch.setattr(IntentRegistry, "run", classmethod(fake_run))

    composed, failed = await mh.build_intent_aware_prompt(
        base="BASE",
        intent="log_nutrition",
        session=None,
    )
    assert "BASE" in composed
    assert "RULES_BLOCK" in composed
    assert "DATA_BLOCK" in composed
    assert failed is False


# --- timezone auto-injection ------------------------------------------
#
# The injection logic is factored into the pure helper
# `_maybe_inject_timezone` so it's testable without standing up a full
# fake Anthropic stream loop (the existing test infra doesn't drive the
# real loop, and faking the streaming context manager + get_final_message
# + a tool-use turn followed by a terminating turn would be far heavier
# than the behavior under test). The harness call site uses this same
# helper, so testing it directly exercises the production path.


def test_inject_timezone_into_list_nutrition_log():
    from prog_strength_agent.model_harness import _maybe_inject_timezone

    out = _maybe_inject_timezone(
        "list_nutrition_log", {"date": "2026-06-03"}, "America/Denver"
    )
    assert out["timezone"] == "America/Denver"
    assert out["date"] == "2026-06-03"


def test_inject_timezone_into_get_daily_macros():
    from prog_strength_agent.model_harness import _maybe_inject_timezone

    out = _maybe_inject_timezone(
        "get_daily_macros", {"date": "2026-06-03"}, "America/Denver"
    )
    assert out["timezone"] == "America/Denver"


def test_non_nutrition_tool_passes_through_unchanged():
    from prog_strength_agent.model_harness import _maybe_inject_timezone

    out = _maybe_inject_timezone(
        "list_pantry_items", {"foo": "bar"}, "America/Denver"
    )
    assert out == {"foo": "bar"}
    assert "timezone" not in out


def test_no_injection_when_client_timezone_is_none():
    from prog_strength_agent.model_harness import _maybe_inject_timezone

    out = _maybe_inject_timezone(
        "list_nutrition_log", {"date": "2026-06-03"}, None
    )
    assert "timezone" not in out


def test_model_supplied_timezone_not_overwritten():
    from prog_strength_agent.model_harness import _maybe_inject_timezone

    out = _maybe_inject_timezone(
        "get_daily_macros",
        {"date": "2026-06-03", "timezone": "Europe/London"},
        "America/Denver",
    )
    assert out["timezone"] == "Europe/London"


def test_inject_does_not_mutate_input():
    from prog_strength_agent.model_harness import _maybe_inject_timezone

    original = {"date": "2026-06-03"}
    _maybe_inject_timezone("list_nutrition_log", original, "America/Denver")
    assert original == {"date": "2026-06-03"}


# --- batch item_count on tool_use_start -------------------------------
#
# `_batch_item_count` is the pure helper the harness uses to stamp the
# web chip's count onto the tool_use_start SSE event. It's tested
# directly (same rationale as `_maybe_inject_timezone` above — the suite
# intentionally does not fake the streaming loop).


def test_batch_item_count_for_batch_tool():
    from prog_strength_agent.model_harness import _batch_item_count

    assert _batch_item_count("log_consumption_batch", {"items": [1, 2, 3]}) == 3
    assert _batch_item_count("log_consumption_batch", {"items": []}) == 0


def test_batch_item_count_none_for_other_tools_and_missing_input():
    from prog_strength_agent.model_harness import _batch_item_count

    assert _batch_item_count("list_pantry_items", {"items": [1]}) is None
    assert _batch_item_count("log_consumption_batch", None) is None
    assert _batch_item_count("log_consumption_batch", {}) is None


# --- vision routing (_has_image + _route_and_stream short-circuit) -----
#
# Image turns must skip the intent classifier entirely and force the
# vision-capable (complex) harness. _has_image is a pure helper (same
# pattern as _maybe_inject_timezone) so the detection is unit-testable
# without driving the routing path; the routing-branch tests below drive
# _route_and_stream with a fake harness so no real Anthropic/MCP session
# is needed.


def _image_block() -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": "ZmFrZQ==",
        },
    }


def test_has_image_detects_image_block():
    from prog_strength_agent.server import _has_image

    messages = [
        {
            "role": "user",
            "content": [_image_block(), {"type": "text", "text": "log this"}],
        }
    ]
    assert _has_image(messages) is True


def test_has_image_false_for_string_content():
    from prog_strength_agent.server import _has_image

    messages = [{"role": "user", "content": "just text, no image"}]
    assert _has_image(messages) is False


def test_has_image_false_for_text_only_list():
    from prog_strength_agent.server import _has_image

    messages = [
        {"role": "user", "content": [{"type": "text", "text": "still no image"}]}
    ]
    assert _has_image(messages) is False


def test_has_image_ignores_assistant_image():
    from prog_strength_agent.server import _has_image

    # Only user turns count — an image carried by an assistant-role
    # message is not a vision request.
    messages = [
        {"role": "assistant", "content": [_image_block()]},
        {"role": "user", "content": "what's in that picture?"},
    ]
    assert _has_image(messages) is False


class _FakeHarness:
    """Stand-in for a ModelHarness that records the args stream_chat was
    called with and yields a single model_chosen SSE event carrying its
    own .model — mirroring the real harness's first event.
    """

    def __init__(self, model: str):
        self.model = model
        self.called = False
        self.received_messages = None
        self.received_intent = None
        self.received_memories = None

    async def stream_chat(
        self,
        messages,
        user_token,
        telemetry=None,
        system_prompt=None,
        intent="general",
        client_timezone=None,
        memories=None,
    ):
        self.called = True
        self.received_messages = messages
        self.received_intent = intent
        self.received_memories = memories
        yield _sse_model_chosen(self.model)


def _sse_model_chosen(model: str) -> bytes:
    import json

    return f"data: {json.dumps({'type': 'model_chosen', 'model': model})}\n\n".encode()


def _parse_model_chosen(chunks: list[bytes]) -> dict:
    import json

    payload = chunks[0].decode().removeprefix("data: ").strip()
    return json.loads(payload)


async def test_vision_turn_skips_intent_classifier_and_routes_to_sonnet(monkeypatch):
    """An image turn must skip the prior-intent lookup + router and route
    straight to the complex (vision-capable) harness, stamping had_image.
    """
    from prog_strength_agent import server
    from prog_strength_agent.telemetry import TurnInstrumentation

    # No api_client → prior-intent lookup is skipped regardless of branch.
    monkeypatch.setattr(server, "api_client", None)

    # router_obj.route must NOT be called on a vision turn.
    def _fail_route(*args, **kwargs):
        raise AssertionError("router_obj.route was called on a vision turn")

    monkeypatch.setattr(server.router_obj, "route", _fail_route)

    fake = _FakeHarness(model=server.VISION_MODEL)
    monkeypatch.setitem(server.HARNESSES, "complex", fake)

    messages = [
        {
            "role": "user",
            "content": [_image_block(), {"type": "text", "text": "log this"}],
        }
    ]
    telemetry = TurnInstrumentation.new(user_id="u-1", session_id="s-1")

    chunks = [
        chunk
        async for chunk in server._route_and_stream(
            messages, "dummy-token", telemetry, "SYSTEM_PROMPT"
        )
    ]

    assert fake.called is True
    assert fake.received_intent == "general"
    event = _parse_model_chosen(chunks)
    assert event["model"] == server.VISION_MODEL
    assert telemetry.had_image is True
    assert telemetry.routed_tier == "complex"
    assert telemetry.intent == "general"


async def test_text_only_turn_unchanged(monkeypatch):
    """Regression guard: a text-only turn still calls router_obj.route and
    selects the matching harness; had_image stays False.
    """
    from prog_strength_agent import server
    from prog_strength_agent.model_router import RouterDecision
    from prog_strength_agent.telemetry import TurnInstrumentation

    monkeypatch.setattr(server, "api_client", None)

    route_called = {"n": 0}

    async def _fake_route(messages, telemetry=None, prior_intent=None):
        route_called["n"] += 1
        telemetry.routed_tier = "simple"
        telemetry.intent = "general"
        return RouterDecision(tier="simple", intent="general")

    monkeypatch.setattr(server.router_obj, "route", _fake_route)

    simple = _FakeHarness(model="claude-haiku-4-5-20251001")
    complex_ = _FakeHarness(model=server.VISION_MODEL)
    monkeypatch.setitem(server.HARNESSES, "simple", simple)
    monkeypatch.setitem(server.HARNESSES, "complex", complex_)

    messages = [{"role": "user", "content": "how many calories today?"}]
    telemetry = TurnInstrumentation.new(user_id="u-1", session_id="s-1")

    chunks = [
        chunk
        async for chunk in server._route_and_stream(
            messages, "dummy-token", telemetry, "SYSTEM_PROMPT"
        )
    ]

    assert route_called["n"] == 1
    assert simple.called is True
    assert complex_.called is False
    assert telemetry.had_image is False
    event = _parse_model_chosen(chunks)
    assert event["model"] == "claude-haiku-4-5-20251001"


async def test_vision_turn_forwards_multimodal_content_unchanged(monkeypatch):
    """Proposes-before-logging seam.

    Driving the full Anthropic streaming context manager + MCP tool-use
    loop to assert "no log_custom_meal on the first image turn, fires
    after a 'yes'" is heavier than the existing test infra supports (see
    the timezone-helper rationale block above — the harness loop is not
    faked anywhere in this suite). The behavior the *routing* layer is
    responsible for, and the only seam testable without that machinery,
    is that it forwards the user's multimodal content to the harness
    UNCHANGED (it does not rewrite or strip the image block) and invokes
    no tool itself — the model proposing vs. logging is the harness's
    job, exercised end-to-end in CI integration tests. The prompt
    paragraph (test_prompt.py / prompt.py) carries the "propose first,
    log on yes" instruction the model follows.
    """
    from prog_strength_agent import server
    from prog_strength_agent.telemetry import TurnInstrumentation

    monkeypatch.setattr(server, "api_client", None)
    monkeypatch.setattr(
        server.router_obj,
        "route",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("router_obj.route called on a vision turn")
        ),
    )

    fake = _FakeHarness(model=server.VISION_MODEL)
    monkeypatch.setitem(server.HARNESSES, "complex", fake)

    user_content = [_image_block(), {"type": "text", "text": "log this"}]
    messages = [{"role": "user", "content": user_content}]
    telemetry = TurnInstrumentation.new(user_id="u-1", session_id="s-1")

    async for _ in server._route_and_stream(
        messages, "dummy-token", telemetry, "SYSTEM_PROMPT"
    ):
        pass

    # The multimodal content reaches the harness byte-for-byte — the
    # image block is neither rewritten nor dropped by the routing layer.
    forwarded = fake.received_messages[0]["content"]
    assert forwarded == user_content
    assert forwarded[0]["type"] == "image"
    # The routing path invokes no tool; telemetry recorded none.
    assert telemetry.tool_calls == []


# --- prompt caching -------------------------------------------------------


class _FakeTool:
    def __init__(self, name: str, schema: dict | None = None):
        self.name = name
        self.description = f"{name} tool"
        self.inputSchema = schema or {"type": "object"}


def test_tool_schemas_cache_breakpoint_on_last_tool_only():
    """The tools array is the largest fully-stable prefix block in every
    request; one breakpoint on the LAST tool caches the whole array.
    Earlier tools must stay unmarked — each cache_control is a separate
    breakpoint, and Anthropic caps them at four per request."""
    from prog_strength_agent.model_harness import _build_tool_schemas

    tools = _build_tool_schemas([_FakeTool("a"), _FakeTool("b"), _FakeTool("c")])
    assert tools[-1]["cache_control"] == {"type": "ephemeral"}
    assert all("cache_control" not in t for t in tools[:-1])
    # Schema conversion is otherwise unchanged.
    assert [t["name"] for t in tools] == ["a", "b", "c"]


def test_tool_schemas_empty_list_unmarked():
    from prog_strength_agent.model_harness import _build_tool_schemas

    assert _build_tool_schemas([]) == []


def test_system_blocks_shape():
    """System goes to the API as a single text block carrying the
    second cache breakpoint, so follow-up calls re-read it at the
    cached-input price. The text itself is passed through verbatim —
    caching must never change what the model sees."""
    from prog_strength_agent.model_harness import _system_blocks

    blocks = _system_blocks("PROMPT TEXT")
    assert blocks == [
        {
            "type": "text",
            "text": "PROMPT TEXT",
            "cache_control": {"type": "ephemeral"},
        }
    ]


# --- tool_result request-id surfacing --------------------------------------


def test_request_id_extracted_from_json_object_result():
    from prog_strength_agent.model_harness import _request_id_from_result

    text = '{"matches": [], "quantity": 1, "request_id": "req-abc-123"}'
    assert _request_id_from_result(text) == "req-abc-123"
    # Error-shaped results carry it too (failed lookups stay traceable).
    assert (
        _request_id_from_result('{"error": "lookup_failed", "request_id": "req-x"}')
        == "req-x"
    )


def test_request_id_absent_or_unparseable_yields_none():
    from prog_strength_agent.model_harness import _request_id_from_result

    assert _request_id_from_result('{"matches": []}') is None  # no id
    assert _request_id_from_result('{"request_id": 42}') is None  # wrong type
    assert _request_id_from_result('{"request_id": ""}') is None  # empty
    assert _request_id_from_result("[1, 2, 3]") is None  # not an object
    assert _request_id_from_result("plain text tool result") is None
    assert _request_id_from_result("") is None


# --- shared tool-event helpers --------------------------------------------
#
# The model loop and the prefetch path both surface tool activity using
# the same two SSE event shapes, factored into these helpers so the two
# code paths can't drift.


def test_tool_start_event_stamps_batch_item_count():
    from prog_strength_agent.model_harness import _tool_start_event

    assert _tool_start_event("log_consumption_batch", {"items": [1, 2]}) == {
        "type": "tool_use_start",
        "name": "log_consumption_batch",
        "item_count": 2,
    }


def test_tool_start_event_omits_item_count_for_non_batch():
    from prog_strength_agent.model_harness import _tool_start_event

    assert _tool_start_event("get_training_snapshot", {"timezone": "UTC"}) == {
        "type": "tool_use_start",
        "name": "get_training_snapshot",
    }


def test_tool_result_event_surfaces_request_id_and_error_flag():
    from prog_strength_agent.model_harness import _tool_result_event

    assert _tool_result_event(
        "lookup_food_nutrition", '{"request_id": "req-9"}', False
    ) == {
        "type": "tool_result",
        "name": "lookup_food_nutrition",
        "is_error": False,
        "request_id": "req-9",
    }
    assert _tool_result_event("get_training_snapshot", "boom", True) == {
        "type": "tool_result",
        "name": "get_training_snapshot",
        "is_error": True,
    }


# --- prefetch tool-usage logging ------------------------------------------
#
# Intent prefetch tool calls (e.g. analyze_training's get_training_snapshot)
# run before the model loop. They used to bypass the only paths that report
# tool usage to the web app — telemetry.tool_calls and the SSE stream — so
# they were invisible in the UI. _PrefetchToolRecorder captures them and
# _prefetch_tool_events replays them onto both paths exactly like a
# model-issued call. These seams are unit-testable without faking the full
# Anthropic stream loop (same convention as the helpers above).


class _FakeContent:
    def __init__(self, text: str):
        self.text = text


class _FakeToolResult:
    def __init__(self, content: list, is_error: bool = False):
        self.content = content
        self.isError = is_error


class _RecordingFakeSession:
    """Minimal MCP-session stand-in: returns a canned result (or raises)
    from call_tool and remembers what it was called with."""

    def __init__(self, result=None, raises: Exception | None = None):
        self.result = result
        self.raises = raises
        self.calls: list = []
        self.extra_attr = "delegated"

    async def call_tool(self, name, arguments=None):
        self.calls.append((name, arguments))
        if self.raises is not None:
            raise self.raises
        return self.result


@pytest.mark.asyncio
async def test_prefetch_recorder_captures_successful_call():
    from prog_strength_agent.model_harness import _PrefetchToolRecorder

    inner = _RecordingFakeSession(
        result=_FakeToolResult([_FakeContent('{"period": {"days": 7}}')])
    )
    rec = _PrefetchToolRecorder(inner)

    out = await rec.call_tool("get_training_snapshot", {"timezone": "UTC"})

    assert out is inner.result  # real result still flows back to the prefetch
    assert inner.calls == [("get_training_snapshot", {"timezone": "UTC"})]
    assert len(rec.calls) == 1
    call = rec.calls[0]
    assert call.record.tool_name == "get_training_snapshot"
    assert call.is_error is False
    assert call.record.error is None
    assert call.arguments == {"timezone": "UTC"}
    assert '"timezone": "UTC"' in call.record.arguments_json
    assert call.record.result_summary is not None


@pytest.mark.asyncio
async def test_prefetch_recorder_marks_is_error_result():
    from prog_strength_agent.model_harness import _PrefetchToolRecorder

    inner = _RecordingFakeSession(
        result=_FakeToolResult([_FakeContent("snapshot failed")], is_error=True)
    )
    rec = _PrefetchToolRecorder(inner)

    await rec.call_tool("get_training_snapshot", {})

    call = rec.calls[0]
    assert call.is_error is True
    assert call.record.error == "snapshot failed"


@pytest.mark.asyncio
async def test_prefetch_recorder_records_then_reraises_on_exception():
    from prog_strength_agent.model_harness import _PrefetchToolRecorder

    inner = _RecordingFakeSession(raises=RuntimeError("mcp down"))
    rec = _PrefetchToolRecorder(inner)

    with pytest.raises(RuntimeError):
        await rec.call_tool("get_training_snapshot", {})

    # The attempt is still recorded so a failing fetch is visible in the UI.
    assert len(rec.calls) == 1
    assert rec.calls[0].is_error is True
    assert "mcp down" in rec.calls[0].record.error


@pytest.mark.asyncio
async def test_prefetch_recorder_delegates_other_session_attrs():
    from prog_strength_agent.model_harness import _PrefetchToolRecorder

    inner = _RecordingFakeSession(result=_FakeToolResult([]))
    rec = _PrefetchToolRecorder(inner)

    # Anything that isn't call_tool passes straight through to the session.
    assert rec.extra_attr == "delegated"


@pytest.mark.asyncio
async def test_prefetch_tool_events_records_telemetry_and_emits_sse():
    import json

    from prog_strength_agent.model_harness import (
        _prefetch_tool_events,
        _PrefetchToolRecorder,
    )
    from prog_strength_agent.telemetry import TurnInstrumentation

    inner = _RecordingFakeSession(
        result=_FakeToolResult([_FakeContent('{"request_id": "req-7"}')])
    )
    rec = _PrefetchToolRecorder(inner)
    await rec.call_tool("get_training_snapshot", {"timezone": "UTC"})

    telemetry = TurnInstrumentation.new(user_id="u-1", session_id="s-1")
    chunks = list(_prefetch_tool_events(rec.calls, telemetry))

    # Persisted path: one ToolCallRecord lands in telemetry.tool_calls.
    assert len(telemetry.tool_calls) == 1
    assert telemetry.tool_calls[0].tool_name == "get_training_snapshot"

    # Live path: a tool_use_start then a tool_result, same shapes the model
    # loop emits.
    events = [
        json.loads(c.decode().removeprefix("data: ").strip()) for c in chunks
    ]
    assert [e["type"] for e in events] == ["tool_use_start", "tool_result"]
    assert events[0]["name"] == "get_training_snapshot"
    assert events[1]["name"] == "get_training_snapshot"
    assert events[1]["is_error"] is False
    assert events[1]["request_id"] == "req-7"


@pytest.mark.asyncio
async def test_prefetch_tool_events_tolerates_no_telemetry():
    from prog_strength_agent.model_harness import (
        _prefetch_tool_events,
        _PrefetchToolRecorder,
    )

    inner = _RecordingFakeSession(result=_FakeToolResult([_FakeContent("{}")]))
    rec = _PrefetchToolRecorder(inner)
    await rec.call_tool("get_training_snapshot", {})

    # telemetry=None (untracked turn) must still emit the SSE events.
    chunks = list(_prefetch_tool_events(rec.calls, None))
    assert len(chunks) == 2


@pytest.mark.asyncio
async def test_analyze_training_prefetch_call_is_recorded():
    """End-to-end on the real intent: running analyze_training's prefetch
    through the recorder captures its get_training_snapshot call — the
    exact tool that was invisible in the web app before this fix."""
    from prog_strength_agent.intents import IntentRegistry
    from prog_strength_agent.model_harness import _PrefetchToolRecorder

    inner = _RecordingFakeSession(
        result=_FakeToolResult([_FakeContent('{"period": {"days": 7}}')])
    )
    rec = _PrefetchToolRecorder(inner)

    _rules, _data, failed = await IntentRegistry.run(
        "analyze_training", rec, "America/Denver"
    )

    assert failed is False
    assert [c.record.tool_name for c in rec.calls] == ["get_training_snapshot"]
