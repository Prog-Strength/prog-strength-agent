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
    from prog_strength_agent.intents import IntentRegistry
    from prog_strength_agent import model_harness as mh

    async def fake_run(cls, intent, session):
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

    async def stream_chat(
        self,
        messages,
        user_token,
        telemetry=None,
        system_prompt=None,
        intent="general",
        client_timezone=None,
    ):
        self.called = True
        self.received_messages = messages
        self.received_intent = intent
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
