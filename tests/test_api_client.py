"""Tests for the agent's tiny API client (today: just get_session_intent)."""

from __future__ import annotations

import httpx
import pytest
import respx

from prog_strength_agent.api_client import APIClient


@pytest.mark.asyncio
@respx.mock
async def test_get_session_intent_returns_value_on_200():
    respx.get("http://api:8080/internal/chat-sessions/abc/intent").mock(
        return_value=httpx.Response(
            200,
            json={"data": {"intent": "log_nutrition", "intent_at": "2026-06-02T00:00:00Z"}},
        )
    )
    client = APIClient(base_url="http://api:8080")
    intent = await client.get_session_intent("abc")
    assert intent == "log_nutrition"
    await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_get_session_intent_returns_none_when_null():
    respx.get("http://api:8080/internal/chat-sessions/abc/intent").mock(
        return_value=httpx.Response(200, json={"data": {"intent": None, "intent_at": None}}),
    )
    client = APIClient(base_url="http://api:8080")
    assert await client.get_session_intent("abc") is None
    await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_get_session_intent_returns_none_on_5xx():
    respx.get("http://api:8080/internal/chat-sessions/abc/intent").mock(
        return_value=httpx.Response(500, text="boom"),
    )
    client = APIClient(base_url="http://api:8080")
    assert await client.get_session_intent("abc") is None
    await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_get_session_intent_returns_none_on_network_error():
    respx.get("http://api:8080/internal/chat-sessions/abc/intent").mock(
        side_effect=httpx.ConnectError("nope")
    )
    client = APIClient(base_url="http://api:8080")
    assert await client.get_session_intent("abc") is None
    await client.aclose()
