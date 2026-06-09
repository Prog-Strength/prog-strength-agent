"""Tests for the per-user UsageGate.

Covers the cache (hit/miss), soft-allow on API failure, the capped →
CapExceeded + invalidate path, the stampede collapse to one HTTP call,
and the disabled no-op.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
import respx

from prog_strength_agent.usage_gate import (
    CAP_EXCEEDED_MESSAGE,
    CapExceeded,
    UsageGate,
)

BASE = "http://api:8080"
TOKEN = "user-jwt-token"


def _usage_response(percent_used: int, capped: bool) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "data": {
                "percent_used": percent_used,
                "capped": capped,
                "resets_at": "2026-06-10T06:00:00Z",
            }
        },
    )


@respx.mock
async def test_cache_miss_then_hit_one_http_call():
    route = respx.get(f"{BASE}/me/usage").mock(return_value=_usage_response(40, False))
    gate = UsageGate(BASE)
    # First call: miss → one fetch.
    await gate.check_or_raise(user_id="u1", token=TOKEN, tz="America/Denver")
    # Second call within TTL: hit → no new fetch.
    await gate.check_or_raise(user_id="u1", token=TOKEN, tz="America/Denver")
    assert route.call_count == 1
    # The user's raw JWT must be forwarded — /me/usage is JWT-gated and
    # identifies the user solely from the bearer token.
    assert route.calls[0].request.headers["Authorization"] == f"Bearer {TOKEN}"
    await gate.aclose()


@respx.mock
async def test_miss_stores_and_passes_tz():
    route = respx.get(f"{BASE}/me/usage").mock(return_value=_usage_response(10, False))
    gate = UsageGate(BASE)
    await gate.check_or_raise(user_id="u1", token=TOKEN, tz="Europe/London")
    assert route.call_count == 1
    assert route.calls[0].request.url.params.get("tz") == "Europe/London"
    assert route.calls[0].request.headers["Authorization"] == f"Bearer {TOKEN}"
    await gate.aclose()


@respx.mock
async def test_api_timeout_soft_allows():
    respx.get(f"{BASE}/me/usage").mock(side_effect=httpx.ConnectTimeout("slow"))
    gate = UsageGate(BASE)
    # Soft-allow: returns None, does not raise.
    assert await gate.check_or_raise(user_id="u1", token=TOKEN, tz=None) is None
    await gate.aclose()


@respx.mock
async def test_api_5xx_soft_allows():
    respx.get(f"{BASE}/me/usage").mock(return_value=httpx.Response(500, text="boom"))
    gate = UsageGate(BASE)
    assert await gate.check_or_raise(user_id="u1", token=TOKEN, tz=None) is None
    await gate.aclose()


@respx.mock
async def test_bad_json_soft_allows():
    respx.get(f"{BASE}/me/usage").mock(
        return_value=httpx.Response(200, json={"data": {"percent_used": "nope"}})
    )
    gate = UsageGate(BASE)
    assert await gate.check_or_raise(user_id="u1", token=TOKEN, tz=None) is None
    await gate.aclose()


@respx.mock
async def test_capped_raises_and_invalidates():
    respx.get(f"{BASE}/me/usage").mock(return_value=_usage_response(100, True))
    gate = UsageGate(BASE)
    with pytest.raises(CapExceeded) as exc:
        await gate.check_or_raise(user_id="u1", token=TOKEN, tz=None, surface="chat")
    assert str(exc.value) == CAP_EXCEEDED_MESSAGE
    # invalidate() ran inside the raise path → no stale entry cached.
    assert "u1" not in gate._cache
    await gate.aclose()


@respx.mock
async def test_stampede_collapses_to_one_call():
    route = respx.get(f"{BASE}/me/usage").mock(return_value=_usage_response(20, False))
    gate = UsageGate(BASE)
    # 10 concurrent first-time checks for the same user → exactly one
    # HTTP fetch; the rest read the freshly stored snapshot.
    await asyncio.gather(
        *(gate.check_or_raise(user_id="u1", token=TOKEN, tz=None) for _ in range(10))
    )
    assert route.call_count == 1
    await gate.aclose()


@respx.mock
async def test_disabled_is_noop_no_http():
    route = respx.get(f"{BASE}/me/usage").mock(return_value=_usage_response(100, True))
    gate = UsageGate(BASE, enabled=False)
    # Even though usage would be capped, disabled gate never checks.
    assert await gate.check_or_raise(user_id="u1", token=TOKEN, tz=None) is None
    assert route.call_count == 0
    await gate.aclose()


@respx.mock
async def test_ttl_expiry_refetches():
    route = respx.get(f"{BASE}/me/usage").mock(return_value=_usage_response(50, False))
    # Zero TTL forces a refetch on the second call.
    gate = UsageGate(BASE, cache_ttl_seconds=0.0)
    await gate.check_or_raise(user_id="u1", token=TOKEN, tz=None)
    await gate.check_or_raise(user_id="u1", token=TOKEN, tz=None)
    assert route.call_count == 2
    await gate.aclose()
