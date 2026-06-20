"""Tests for APIClient.retrieve_memories — the best-effort vector-memory
lookup the agent runs alongside the router. Mirrors test_api_client.py:
any failure (4xx, 5xx, transport error, malformed payload) collapses to
an empty list, and empty input short-circuits without a request.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from prog_strength_agent.api_client import APIClient


@pytest.mark.asyncio
@respx.mock
async def test_retrieve_memories_returns_texts_on_200():
    respx.post("http://api:8080/internal/memory/retrieve").mock(
        return_value=httpx.Response(
            200,
            json={"data": {"memories": [{"text": "a"}, {"text": "b"}]}},
        )
    )
    client = APIClient(base_url="http://api:8080")
    memories = await client.retrieve_memories("u-1", "bench day")
    assert memories == ["a", "b"]
    await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_retrieve_memories_empty_on_4xx():
    respx.post("http://api:8080/internal/memory/retrieve").mock(
        return_value=httpx.Response(404, text="not found")
    )
    client = APIClient(base_url="http://api:8080")
    assert await client.retrieve_memories("u-1", "q") == []
    await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_retrieve_memories_empty_on_5xx():
    respx.post("http://api:8080/internal/memory/retrieve").mock(
        return_value=httpx.Response(500, text="boom")
    )
    client = APIClient(base_url="http://api:8080")
    assert await client.retrieve_memories("u-1", "q") == []
    await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_retrieve_memories_empty_on_transport_error():
    respx.post("http://api:8080/internal/memory/retrieve").mock(
        side_effect=httpx.ConnectError("nope")
    )
    client = APIClient(base_url="http://api:8080")
    assert await client.retrieve_memories("u-1", "q") == []
    await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_retrieve_memories_empty_on_malformed_payload():
    respx.post("http://api:8080/internal/memory/retrieve").mock(
        return_value=httpx.Response(200, json={"data": {"memories": "nope"}})
    )
    client = APIClient(base_url="http://api:8080")
    assert await client.retrieve_memories("u-1", "q") == []
    await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_retrieve_memories_skips_non_dict_and_non_text_entries():
    respx.post("http://api:8080/internal/memory/retrieve").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "memories": [
                        {"text": "keep"},
                        {"text": 42},
                        "not a dict",
                        {"no_text": "x"},
                    ]
                }
            },
        )
    )
    client = APIClient(base_url="http://api:8080")
    assert await client.retrieve_memories("u-1", "q") == ["keep"]
    await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_retrieve_memories_empty_user_id_makes_no_request():
    route = respx.post("http://api:8080/internal/memory/retrieve").mock(
        return_value=httpx.Response(200, json={"data": {"memories": [{"text": "a"}]}})
    )
    client = APIClient(base_url="http://api:8080")
    assert await client.retrieve_memories("", "q") == []
    assert route.call_count == 0
    await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_retrieve_memories_empty_query_makes_no_request():
    route = respx.post("http://api:8080/internal/memory/retrieve").mock(
        return_value=httpx.Response(200, json={"data": {"memories": [{"text": "a"}]}})
    )
    client = APIClient(base_url="http://api:8080")
    assert await client.retrieve_memories("u-1", "") == []
    assert route.call_count == 0
    await client.aclose()
