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
