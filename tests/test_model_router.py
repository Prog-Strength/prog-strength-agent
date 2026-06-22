"""Unit tests for ModelRouter — structured-output (tier, intent)
classification plus the prior_intent hint plumbing."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from prog_strength_agent.model_router import (
    ModelRouter,
    RouterDecision,
)


def _tool_use_block(tier: str, intent: str):
    return SimpleNamespace(
        type="tool_use",
        name="classify_request",
        input={"tier": tier, "intent": intent},
    )


def _resp(blocks):
    return SimpleNamespace(content=blocks, usage=None, stop_reason="tool_use")


@pytest.mark.asyncio
async def test_route_parses_tool_use_into_router_decision():
    client = SimpleNamespace(
        messages=SimpleNamespace(
            create=AsyncMock(return_value=_resp([_tool_use_block("simple", "log_nutrition")]))
        )
    )
    router = ModelRouter(client=client, router_model="claude-haiku-4-5-20251001")

    decision = await router.route(
        messages=[{"role": "user", "content": "log a protein shake for dinner"}],
        telemetry=None,
        prior_intent=None,
    )
    assert isinstance(decision, RouterDecision)
    assert decision.tier == "simple"
    assert decision.intent == "log_nutrition"


@pytest.mark.asyncio
async def test_route_falls_back_to_simple_general_on_classifier_error():
    client = SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(side_effect=RuntimeError("boom")))
    )
    router = ModelRouter(client=client, router_model="claude-haiku-4-5-20251001")

    decision = await router.route(messages=[{"role": "user", "content": "hi"}], telemetry=None)
    assert decision == RouterDecision(tier="simple", intent="general")


@pytest.mark.asyncio
async def test_route_falls_back_when_response_has_no_tool_use():
    client = SimpleNamespace(
        messages=SimpleNamespace(
            create=AsyncMock(return_value=_resp([SimpleNamespace(type="text", text="oops")]))
        )
    )
    router = ModelRouter(client=client, router_model="claude-haiku-4-5-20251001")
    decision = await router.route(messages=[{"role": "user", "content": "hi"}], telemetry=None)
    assert decision == RouterDecision(tier="simple", intent="general")


@pytest.mark.asyncio
async def test_route_includes_prior_intent_hint_in_user_message_when_provided():
    create_mock = AsyncMock(return_value=_resp([_tool_use_block("simple", "log_nutrition")]))
    client = SimpleNamespace(messages=SimpleNamespace(create=create_mock))
    router = ModelRouter(client=client, router_model="claude-haiku-4-5-20251001")

    await router.route(
        messages=[{"role": "user", "content": "a protein shake"}],
        telemetry=None,
        prior_intent="log_nutrition",
    )

    sent = create_mock.await_args.kwargs
    # Hint is delivered as user-message context; verify the substring lands somewhere.
    payload = str(sent["messages"])
    assert "log_nutrition" in payload  # hint present


@pytest.mark.asyncio
async def test_route_populates_intent_on_telemetry():
    from prog_strength_agent.telemetry import TurnInstrumentation

    client = SimpleNamespace(
        messages=SimpleNamespace(
            create=AsyncMock(return_value=_resp([_tool_use_block("complex", "analyze_training")]))
        )
    )
    router = ModelRouter(client=client, router_model="claude-haiku-4-5-20251001")
    t = TurnInstrumentation.new(user_id="u-1", session_id=None)

    await router.route(messages=[{"role": "user", "content": "how is my bench progressing?"}], telemetry=t)
    assert t.routed_tier == "complex"
    assert t.intent == "analyze_training"


def test_router_prompt_routes_external_meal_estimation_to_complex():
    """Phase 3 of the macro-accuracy SOW: external-meal logging turns
    classify as complex so no-data estimation runs on the stronger
    model. Pin the prompt rule, not the classifier behavior (that's
    the eval harness's job)."""
    from prog_strength_agent.model_router import ROUTER_SYSTEM_PROMPT

    assert "restaurant" in ROUTER_SYSTEM_PROMPT
    assert "external source" in ROUTER_SYSTEM_PROMPT


def test_router_prompt_mentions_plan_workout_intent():
    """The router must know the plan_workout intent so 'plan my week'
    routes to the planning enrichment (and complex tier)."""
    from prog_strength_agent.model_router import ROUTER_SYSTEM_PROMPT

    assert "plan_workout" in ROUTER_SYSTEM_PROMPT
    assert "plan my week" in ROUTER_SYSTEM_PROMPT
