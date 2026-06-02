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


class _FakeMCPResult:
    def __init__(self, text: str):
        # Mirrors mcp.types.TextContent's duck-typed shape.
        self.content = [type("Block", (), {"text": text})()]
        self.isError = False


class _FakeSession:
    def __init__(self, responses: dict[str, str]):
        self._responses = responses

    async def call_tool(self, name: str, args: dict):
        return _FakeMCPResult(self._responses.get(name, "[]"))


@pytest.mark.asyncio
async def test_log_nutrition_prefetch_fans_out_pantry_and_recipes():
    session = _FakeSession({
        "list_pantry_items": '[{"id":"p-1","name":"Whey","calories":120,"protein_g":24,"fat_g":1,"carbs_g":3,"serving_size":1,"serving_unit":"scoop"}]',
        "list_recipes":      '[{"id":"r-1","name":"Standard Breakfast","components":[{"pantry_item_id":"p-1","quantity":1}]}]',
    })
    rules, data = await IntentRegistry.run("log_nutrition", session)

    assert "Assume one serving" in rules
    assert "Whey" in data
    assert "Standard Breakfast" in data
    assert "PANTRY" in data
    assert "RECIPES" in data


@pytest.mark.asyncio
async def test_log_nutrition_prefetch_failure_returns_rules_without_data():
    class _FailingSession:
        async def call_tool(self, name: str, args: dict):
            raise RuntimeError("MCP exploded")
    rules, data = await IntentRegistry.run("log_nutrition", _FailingSession())
    assert "Assume one serving" in rules
    assert data == ""
