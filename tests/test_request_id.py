"""Tests for the per-request correlation id wiring.

Covers the request_id module's primitives (id format, ContextVar, log
filter) and its end-to-end behavior through the FastAPI app: every
response carries the X-Request-ID header, /health echoes it in the
body, an inbound X-Request-ID is honored, error envelopes carry it, and
the /chat SSE done/error events carry it.
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncGenerator

import jwt
import pytest
from fastapi.testclient import TestClient

from prog_strength_agent import server
from prog_strength_agent.request_id import (
    HEADER_NAME,
    RequestIDLogFilter,
    _request_id,
    current_request_id,
    new_request_id,
)

SIGNING_KEY = "test-signing-key"  # matches conftest's JWT_SIGNING_KEY default

_HEX32 = re.compile(r"\A[0-9a-f]{32}\Z")


def _auth_headers(user_id: str = "u1") -> dict[str, str]:
    token = jwt.encode({"sub": user_id}, SIGNING_KEY, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def client() -> TestClient:
    return TestClient(server.app)


# --- module primitives ----------------------------------------------------


def test_new_request_id_is_32_char_hex():
    """Matches the Go API's id.New shape (16 random bytes hex-encoded),
    not a dashed uuid4."""
    rid = new_request_id()
    assert _HEX32.match(rid), rid


def test_new_request_id_is_unique():
    assert new_request_id() != new_request_id()


def test_current_request_id_defaults_empty_outside_request():
    assert current_request_id() == ""


def test_log_filter_stamps_current_id():
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "msg", None, None)
    token = _request_id.set("abc123")
    try:
        assert RequestIDLogFilter().filter(record) is True
        assert record.request_id == "abc123"
    finally:
        _request_id.reset(token)


def test_log_filter_stamps_dash_when_no_context():
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "msg", None, None)
    assert RequestIDLogFilter().filter(record) is True
    assert record.request_id == "-"


# --- end-to-end through the app -------------------------------------------


def test_health_response_header_present_and_hex(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    rid = resp.headers[HEADER_NAME]
    assert _HEX32.match(rid), rid


def test_health_body_request_id_matches_header(client):
    resp = client.get("/health")
    body = resp.json()
    assert body["service"] == "Prog Strength Agent"
    assert body["message"] == "service is healthy"
    # Body id and header id are the same minted value.
    assert body["request_id"] == resp.headers[HEADER_NAME]
    assert _HEX32.match(body["request_id"])


def test_each_request_gets_a_distinct_id(client):
    first = client.get("/health").json()["request_id"]
    second = client.get("/health").json()["request_id"]
    assert first != second


def test_inbound_request_id_is_honored(client):
    """A caller-supplied X-Request-ID rides through so a trace can span
    the frontend → agent → API hops."""
    supplied = "deadbeefdeadbeefdeadbeefdeadbeef"
    resp = client.get("/health", headers={HEADER_NAME: supplied})
    assert resp.headers[HEADER_NAME] == supplied
    assert resp.json()["request_id"] == supplied


def test_unauthenticated_error_body_carries_request_id(client):
    """A 401 (no auth header) returns the default detail body plus the
    request id, and the header is still set."""
    resp = client.post("/title", json={"messages": []})
    assert resp.status_code == 401
    body = resp.json()
    assert body["detail"]  # unchanged default shape
    assert body["request_id"] == resp.headers[HEADER_NAME]
    assert _HEX32.match(body["request_id"])


def test_validation_error_body_carries_request_id(client):
    """A 422 (malformed body) carries the request id alongside the
    standard detail list."""
    resp = client.post("/title", headers=_auth_headers(), json={"wrong": "shape"})
    assert resp.status_code == 422
    body = resp.json()
    assert isinstance(body["detail"], list)
    assert body["request_id"] == resp.headers[HEADER_NAME]


def test_chat_done_event_carries_request_id(client, monkeypatch):
    """The terminal SSE `done` event surfaces the agent's request id so
    the client can pivot from a finished stream into the agent's logs."""
    import json

    class _AllowGate:
        async def check_or_raise(self, *, user_id, token, tz, surface="chat"):
            return None

    monkeypatch.setattr(server, "usage_gate", _AllowGate())

    # Drive the REAL harness so the done event flows through the harness's
    # _with_request_id path within the request's ContextVar.
    async def _fake_stream(*args, **kwargs) -> AsyncGenerator[bytes, None]:
        from prog_strength_agent.model_harness import _sse, _with_request_id

        yield _sse(_with_request_id({"type": "done", "stop_reason": "end_turn"}))

    monkeypatch.setattr(server, "_route_and_stream", _fake_stream)

    resp = client.post(
        "/chat",
        headers=_auth_headers(),
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    header_id = resp.headers[HEADER_NAME]
    # Parse the SSE payload and confirm the done event's request_id
    # equals the header the middleware stamped on the same response.
    payloads = [
        json.loads(line[len("data: ") :])
        for line in resp.text.splitlines()
        if line.startswith("data: ")
    ]
    done = next(p for p in payloads if p["type"] == "done")
    assert done["request_id"] == header_id
    assert _HEX32.match(done["request_id"])
