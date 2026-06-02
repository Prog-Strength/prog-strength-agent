"""Unit tests for the IntentRegistry — declarative module of per-intent
prefetch + rules + data formatting."""

from __future__ import annotations

import pytest

from prog_strength_agent.intents import IntentRegistry


@pytest.mark.asyncio
async def test_run_general_returns_empty_blocks():
    rules, data, _failed = await IntentRegistry.run("general", session=None)
    assert rules == ""
    assert data == ""


@pytest.mark.asyncio
async def test_run_unknown_intent_returns_empty_blocks():
    rules, data, _failed = await IntentRegistry.run("definitely_not_an_intent", session=None)
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
    rules, data, _failed = await IntentRegistry.run("log_nutrition", session)

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
    rules, data, _failed = await IntentRegistry.run("log_nutrition", _FailingSession())
    assert "Assume one serving" in rules
    assert data == ""


@pytest.mark.asyncio
async def test_log_workout_prefetch_includes_catalog_and_recent_5():
    session = _FakeSession({
        "list_exercises": '[{"id":"barbell-bench-press","name":"Barbell Bench Press","muscle_groups":["chest"]},{"id":"back-squat","name":"Back Squat","muscle_groups":["quads"]}]',
        # 10 recent workouts in production order (newest first, matching
        # the API's `ORDER BY performed_at DESC`). The formatter slices
        # the first 5 — those are the most recent.
        "list_workouts": '[' + ','.join(
            f'{{"id":"w-{i}","performed_at":"2026-05-{i:02d}T18:00:00Z","exercises":[]}}'
            for i in range(10, 0, -1)
        ) + ']',
    })
    rules, data, _failed = await IntentRegistry.run("log_workout", session)
    assert "exercise catalog" in rules.lower() or "look up exercise slugs" in rules.lower()
    assert "barbell-bench-press" in data
    assert "back-squat" in data
    assert "EXERCISE CATALOG" in data
    assert "RECENT WORKOUTS" in data
    assert "w-10" in data
    assert "w-6" in data    # 10..6 = last 5
    assert "w-5" not in data


@pytest.mark.asyncio
async def test_log_bodyweight_prefetch_calls_list_with_14_day_window(monkeypatch):
    from typing import Any
    captured_args: dict[str, Any] = {}

    class _CaptureSession:
        async def call_tool(self, name: str, args: dict):
            captured_args[name] = args
            return _FakeMCPResult(
                '[{"id":"b-1","weight":205.4,"unit":"lb","measured_at":"2026-05-30T07:00:00Z"}]'
            )

    rules, data, _failed = await IntentRegistry.run("log_bodyweight", _CaptureSession())
    assert "bodyweight" in rules.lower()
    assert "205.4" in data
    assert captured_args.get("list_bodyweight", {}).get("since") is not None


@pytest.mark.asyncio
async def test_analyze_progress_prefetch_includes_workouts_and_macros():
    from typing import Any
    captured: dict[str, Any] = {}

    class _CaptureSession:
        async def call_tool(self, name: str, args: dict):
            captured[name] = args
            return _FakeMCPResult({
                "list_workouts": '[' + ','.join(
                    f'{{"id":"w-{i}","performed_at":"2026-05-{(i%28)+1:02d}T18:00:00Z","exercises":[]}}'
                    for i in range(1, 25)
                ) + ']',
                "get_daily_macros": '[{"date":"2026-05-30","calories":2200,"protein_g":180,"fat_g":70,"carbs_g":230}]',
            }[name])

    rules, data, _failed = await IntentRegistry.run("analyze_progress", _CaptureSession())
    assert "favor citing" in rules.lower() or "already have a recent window" in rules.lower()
    # Workouts capped to 20
    assert data.count("w-") <= 20
    assert "2200" in data
    assert captured.get("get_daily_macros", {}).get("since") is not None
    assert captured.get("get_daily_macros", {}).get("until") is not None


@pytest.mark.asyncio
async def test_run_returns_failed_true_when_prefetch_raises():
    class _FailingSession:
        async def call_tool(self, name: str, args: dict):
            raise RuntimeError("MCP exploded")
    rules, data, failed = await IntentRegistry.run("log_nutrition", _FailingSession())
    assert "Assume one serving" in rules
    assert data == ""
    assert failed is True


@pytest.mark.asyncio
async def test_run_returns_failed_false_on_happy_path():
    session = _FakeSession({"list_pantry_items": "[]", "list_recipes": "[]"})
    rules, data, failed = await IntentRegistry.run("log_nutrition", session)
    assert "Assume one serving" in rules
    assert failed is False


@pytest.mark.asyncio
async def test_run_returns_failed_false_for_general_intent():
    rules, data, failed = await IntentRegistry.run("general", session=None)
    assert rules == "" and data == "" and failed is False
