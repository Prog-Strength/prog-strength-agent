"""Tests for the usage-gate wiring on /chat and /speak.

Drives the FastAPI app via TestClient with a real (test-key-signed)
JWT, and monkeypatches `server.usage_gate` to control the gate
decision without hitting the API. The /chat happy path stubs the
streaming generator so we don't call Claude; the /speak path stubs
TTSGenerator.generate so we don't call OpenAI.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import jwt
import pytest
from fastapi.testclient import TestClient

from prog_strength_agent import server
from prog_strength_agent.usage_gate import CAP_EXCEEDED_MESSAGE, CapExceeded

# Matches conftest's JWT_SIGNING_KEY default.
SIGNING_KEY = "test-signing-key"


def _token(user_id: str = "u1") -> str:
    return jwt.encode({"sub": user_id}, SIGNING_KEY, algorithm="HS256")


def _auth_headers(user_id: str = "u1") -> dict[str, str]:
    return {"Authorization": f"Bearer {_token(user_id)}"}


@pytest.fixture
def client() -> TestClient:
    return TestClient(server.app)


class _AllowGate:
    """Stand-in usage gate that always allows."""

    async def check_or_raise(self, *, user_id, token, tz, surface="chat"):
        return None


class _BlockGate:
    """Stand-in usage gate that always raises CapExceeded."""

    async def check_or_raise(self, *, user_id, token, tz, surface="chat"):
        raise CapExceeded(CAP_EXCEEDED_MESSAGE)


def test_chat_returns_429_when_capped(client, monkeypatch):
    monkeypatch.setattr(server, "usage_gate", _BlockGate())
    resp = client.post(
        "/chat",
        headers=_auth_headers(),
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 429
    assert resp.text == CAP_EXCEEDED_MESSAGE


def test_speak_returns_429_when_capped(client, monkeypatch):
    monkeypatch.setattr(server, "usage_gate", _BlockGate())
    resp = client.post(
        "/speak",
        headers=_auth_headers(),
        json={"text": "hello"},
    )
    assert resp.status_code == 429
    assert resp.text == CAP_EXCEEDED_MESSAGE


def test_chat_passes_through_when_allowed(client, monkeypatch):
    monkeypatch.setattr(server, "usage_gate", _AllowGate())

    async def _fake_stream(*args, **kwargs) -> AsyncGenerator[bytes, None]:
        yield b'data: {"type": "done"}\n\n'

    monkeypatch.setattr(server, "_route_and_stream", _fake_stream)

    resp = client.post(
        "/chat",
        headers=_auth_headers(),
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    assert b'"type": "done"' in resp.content


def test_speak_passes_through_when_allowed(client, monkeypatch):
    monkeypatch.setattr(server, "usage_gate", _AllowGate())

    async def _fake_generate(*, user_id, text, voice, session_id=None):
        return b"mp3-bytes"

    monkeypatch.setattr(server.tts_generator, "generate", _fake_generate)

    resp = client.post(
        "/speak",
        headers=_auth_headers(),
        json={"text": "hello", "session_id": "sess-1"},
    )
    assert resp.status_code == 200
    assert resp.content == b"mp3-bytes"
    assert resp.headers["content-type"] == "audio/mpeg"


def test_chat_threads_profile_into_system_prompt(client, monkeypatch):
    """A /chat request carrying display_name + height_cm builds a system
    prompt with the identity line, captured at the _route_and_stream
    boundary (mirrors how the gate tests stub the stream)."""
    monkeypatch.setattr(server, "usage_gate", _AllowGate())

    captured: dict[str, str] = {}

    async def _capture_stream(messages, token, telemetry, system_prompt, **kwargs):
        captured["system_prompt"] = system_prompt
        yield b'data: {"type": "done"}\n\n'

    monkeypatch.setattr(server, "_route_and_stream", _capture_stream)

    resp = client.post(
        "/chat",
        headers=_auth_headers(),
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "display_name": "Sam",
            "height_cm": 180.0,
        },
    )
    assert resp.status_code == 200
    assert "You are talking to Sam." in captured["system_prompt"]
    assert "They are 180 cm tall." in captured["system_prompt"]


def test_chat_without_profile_fields_still_works(client, monkeypatch):
    """Older clients that omit display_name/height_cm keep working and
    produce a prompt with no identity line."""
    monkeypatch.setattr(server, "usage_gate", _AllowGate())

    captured: dict[str, str] = {}

    async def _capture_stream(messages, token, telemetry, system_prompt, **kwargs):
        captured["system_prompt"] = system_prompt
        yield b'data: {"type": "done"}\n\n'

    monkeypatch.setattr(server, "_route_and_stream", _capture_stream)

    resp = client.post(
        "/chat",
        headers=_auth_headers(),
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    assert "You are talking to" not in captured["system_prompt"]


def test_disabled_gate_short_circuits_both(client, monkeypatch):
    """A disabled gate's check_or_raise is a no-op even if usage would
    be capped — neither /chat nor /speak should 429.
    """
    from prog_strength_agent.usage_gate import UsageGate

    disabled = UsageGate("http://api:8080", enabled=False)
    monkeypatch.setattr(server, "usage_gate", disabled)

    async def _fake_stream(*args, **kwargs) -> AsyncGenerator[bytes, None]:
        yield b'data: {"type": "done"}\n\n'

    async def _fake_generate(*, user_id, text, voice, session_id=None):
        return b"mp3"

    monkeypatch.setattr(server, "_route_and_stream", _fake_stream)
    monkeypatch.setattr(server.tts_generator, "generate", _fake_generate)

    chat_resp = client.post(
        "/chat",
        headers=_auth_headers(),
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    speak_resp = client.post(
        "/speak",
        headers=_auth_headers(),
        json={"text": "hello"},
    )
    assert chat_resp.status_code == 200
    assert speak_resp.status_code == 200
