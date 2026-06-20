"""Tests for the memory-injection wiring across the prompt composition
and _route_and_stream layers.

Two guarantees matter most here:
  1. Non-empty memories produce a BACKGROUND block in the composed prompt.
  2. Empty/None memories leave the composed prompt byte-for-byte identical
     to today's compose_system_prompt(base=, rules=, data=).
"""

from __future__ import annotations

import pytest

from prog_strength_agent import model_harness as mh
from prog_strength_agent.intents import IntentRegistry
from prog_strength_agent.prompt import compose_system_prompt
from prog_strength_agent.server import _last_user_text

# --- compose_system_prompt byte-for-byte no-op ------------------------


def test_compose_no_sections_equals_base():
    assert compose_system_prompt(base="BASE") == "BASE"


def test_compose_empty_background_equals_no_background():
    with_bg = compose_system_prompt(base="B", rules="R", data="D", background="")
    without = compose_system_prompt(base="B", rules="R", data="D")
    assert with_bg == without


def test_compose_non_empty_background_appended_last():
    out = compose_system_prompt(base="B", rules="R", data="D", background="BG")
    assert out.index("D") < out.index("BG")
    assert out.endswith("BG")


# --- build_intent_aware_prompt memory composition ---------------------


@pytest.mark.asyncio
async def test_build_intent_aware_prompt_includes_background(monkeypatch):
    async def fake_run(cls, intent, session):
        return "RULES", "DATA", False

    monkeypatch.setattr(IntentRegistry, "run", classmethod(fake_run))

    composed, _ = await mh.build_intent_aware_prompt(
        base="BASE",
        intent="log_nutrition",
        session=None,
        memories=["prefers dumbbells"],
    )
    assert "## Background: what you remember about this user" in composed
    assert "- prefers dumbbells" in composed


@pytest.mark.asyncio
async def test_build_intent_aware_prompt_empty_memories_byte_for_byte(monkeypatch):
    async def fake_run(cls, intent, session):
        return "RULES", "DATA", False

    monkeypatch.setattr(IntentRegistry, "run", classmethod(fake_run))

    baseline = compose_system_prompt(base="BASE", rules="RULES", data="DATA")

    composed_none, _ = await mh.build_intent_aware_prompt(
        base="BASE", intent="x", session=None, memories=None
    )
    composed_empty, _ = await mh.build_intent_aware_prompt(
        base="BASE", intent="x", session=None, memories=[]
    )
    assert composed_none == baseline
    assert composed_empty == baseline


# --- _last_user_text --------------------------------------------------


def test_last_user_text_string_content():
    messages = [{"role": "user", "content": "how many calories today?"}]
    assert _last_user_text(messages) == "how many calories today?"


def test_last_user_text_block_list_content():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "log"},
                {"type": "image", "source": {}},
                {"type": "text", "text": "this"},
            ],
        }
    ]
    assert _last_user_text(messages) == "log this"


def test_last_user_text_no_user_returns_empty():
    messages = [{"role": "assistant", "content": "hi"}]
    assert _last_user_text(messages) == ""


def test_last_user_text_picks_most_recent_user():
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "second"},
    ]
    assert _last_user_text(messages) == "second"


# --- _route_and_stream best-effort memory wiring ----------------------


class _FakeHarness:
    def __init__(self, model: str):
        self.model = model
        self.received_memories = "UNSET"

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
        self.received_memories = memories
        yield b"data: {}\n\n"


@pytest.mark.asyncio
async def test_route_injects_retrieved_memories(monkeypatch):
    from prog_strength_agent import server
    from prog_strength_agent.model_router import RouterDecision
    from prog_strength_agent.telemetry import TurnInstrumentation

    class _FakeClient:
        async def get_session_intent(self, session_id):
            return None

        async def retrieve_memories(self, user_id, query):
            return ["remembered fact"]

    monkeypatch.setattr(server, "api_client", _FakeClient())

    async def _fake_route(messages, telemetry=None, prior_intent=None):
        telemetry.routed_tier = "simple"
        telemetry.intent = "general"
        return RouterDecision(tier="simple", intent="general")

    monkeypatch.setattr(server.router_obj, "route", _fake_route)

    fake = _FakeHarness(model="m")
    monkeypatch.setitem(server.HARNESSES, "simple", fake)

    messages = [{"role": "user", "content": "hi"}]
    telemetry = TurnInstrumentation.new(user_id="u-1", session_id="s-1")

    async for _ in server._route_and_stream(
        messages, "tok", telemetry, "SYS"
    ):
        pass

    assert fake.received_memories == ["remembered fact"]


@pytest.mark.asyncio
async def test_route_memory_failure_injects_nothing_and_completes(monkeypatch):
    from prog_strength_agent import server
    from prog_strength_agent.model_router import RouterDecision
    from prog_strength_agent.telemetry import TurnInstrumentation

    class _FakeClient:
        async def get_session_intent(self, session_id):
            return None

        async def retrieve_memories(self, user_id, query):
            raise RuntimeError("memory service down")

    monkeypatch.setattr(server, "api_client", _FakeClient())

    async def _fake_route(messages, telemetry=None, prior_intent=None):
        telemetry.routed_tier = "simple"
        telemetry.intent = "general"
        return RouterDecision(tier="simple", intent="general")

    monkeypatch.setattr(server.router_obj, "route", _fake_route)

    fake = _FakeHarness(model="m")
    monkeypatch.setitem(server.HARNESSES, "simple", fake)

    messages = [{"role": "user", "content": "hi"}]
    telemetry = TurnInstrumentation.new(user_id="u-1", session_id="s-1")

    chunks = [
        chunk
        async for chunk in server._route_and_stream(
            messages, "tok", telemetry, "SYS"
        )
    ]

    # The turn still streamed, and a retrieval exception injected nothing.
    assert chunks  # streamed at least one chunk
    assert fake.received_memories == []
