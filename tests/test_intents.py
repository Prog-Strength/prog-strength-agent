"""Unit tests for the IntentRegistry — declarative module of per-intent
prefetch + rules + data formatting."""

from __future__ import annotations

import pytest

from prog_strength_agent.intents import IntentRegistry


@pytest.mark.asyncio
async def test_run_general_returns_empty_blocks():
    rules, data = await IntentRegistry.run("general", session=None)
    assert rules == ""
    assert data == ""


@pytest.mark.asyncio
async def test_run_unknown_intent_returns_empty_blocks():
    rules, data = await IntentRegistry.run("definitely_not_an_intent", session=None)
    assert rules == ""
    assert data == ""


def test_known_intents_enum():
    assert set(IntentRegistry.known()) == {
        "log_nutrition",
        "log_workout",
        "log_bodyweight",
        "analyze_progress",
        "general",
    }
