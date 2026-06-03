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
