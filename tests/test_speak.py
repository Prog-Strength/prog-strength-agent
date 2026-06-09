"""Unit tests for speak.TTSGenerator.

Covers the deterministic bits: input validation (text length + voice
enum), the per-user daily char counter (rollover, atomic charging,
cap enforcement), and the disabled-mode behavior when no OpenAI key
is configured. The actual OpenAI HTTP call is not exercised — that
would require live credentials + cost real money. The validation
and quota gates run before the SDK call so failures there cover
"clients can't misbehave" without touching OpenAI at all.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from prog_strength_agent.speak import (
    SUPPORTED_VOICES,
    QuotaExceeded,
    ServiceDisabled,
    TextRequired,
    TextTooLong,
    TTSGenerator,
    UnknownVoice,
    _DailyCounter,
    _Quota,
)
from prog_strength_agent.telemetry import SpeakCallRecord


class _FakeStreamingResponse:
    """Stand-in for the OpenAI SDK's streaming-response context manager.
    `create(...)` returns this; `__aenter__` yields self; `read()`
    returns canned mp3 bytes.
    """

    def __init__(self, payload: bytes = b"mp3", error: Exception | None = None):
        self._payload = payload
        self._error = error

    async def __aenter__(self):
        if self._error is not None:
            raise self._error
        return self

    async def __aexit__(self, *exc):
        return False

    async def read(self) -> bytes:
        return self._payload


class _FakeSpeech:
    def __init__(self, cm: _FakeStreamingResponse):
        self._cm = cm
        self.with_streaming_response = self

    def create(self, **kwargs):
        return self._cm


class _FakeAudio:
    def __init__(self, cm: _FakeStreamingResponse):
        self.speech = _FakeSpeech(cm)


class _FakeOpenAI:
    def __init__(self, cm: _FakeStreamingResponse):
        self.audio = _FakeAudio(cm)


def _gen_with_fake_client(records: list, *, error: Exception | None = None):
    """Build a TTSGenerator whose OpenAI client is a fake and whose
    on_speak hook appends to `records`.
    """
    gen = TTSGenerator(
        api_key="sk-test",
        model="gpt-4o-mini-tts",
        default_voice="cedar",
        daily_char_cap=10_000,
        on_speak=records.append,
    )
    # Replace the real OpenAI client with the fake context manager.
    cm = _FakeStreamingResponse(error=error) if error else _FakeStreamingResponse()
    gen._client = _FakeOpenAI(cm)
    return gen


@pytest.mark.asyncio
async def test_generate_records_speak_on_success():
    records: list[SpeakCallRecord] = []
    gen = _gen_with_fake_client(records)
    audio = await gen.generate(user_id="u1", text="Hello there", voice="alloy", session_id="sess-1")
    assert audio == b"mp3"
    assert len(records) == 1
    rec = records[0]
    assert rec.user_id == "u1"
    assert rec.chars == len("Hello there")
    assert rec.model == "gpt-4o-mini-tts"
    assert rec.voice == "alloy"
    assert rec.session_id == "sess-1"
    assert rec.error is None


@pytest.mark.asyncio
async def test_generate_records_speak_on_failure():
    records: list[SpeakCallRecord] = []
    gen = _gen_with_fake_client(records, error=RuntimeError("openai exploded"))
    with pytest.raises(RuntimeError):
        await gen.generate(user_id="u1", text="boom text", voice=None)
    # Still recorded one row on the failure path, chars unchanged,
    # error captured, voice defaulted.
    assert len(records) == 1
    rec = records[0]
    assert rec.chars == len("boom text")
    assert rec.voice == "cedar"
    assert rec.error == "openai exploded"


@pytest.mark.asyncio
async def test_generate_does_not_record_on_validation_failure():
    """Rejections before the OpenAI call (no external spend) must not
    produce a telemetry row.
    """
    records: list[SpeakCallRecord] = []
    gen = _gen_with_fake_client(records)
    with pytest.raises(UnknownVoice):
        await gen.generate(user_id="u1", text="hi", voice="elvis")
    assert records == []


@pytest.mark.asyncio
async def test_generate_disabled_when_no_key():
    """No OpenAI key configured → /speak returns ServiceDisabled
    rather than failing to boot. Local dev without OPENAI_API_KEY
    set should still let the rest of the agent run.
    """
    gen = TTSGenerator(
        api_key="",
        model="tts-1",
        default_voice="onyx",
        daily_char_cap=1000,
    )
    assert gen.enabled is False
    with pytest.raises(ServiceDisabled):
        await gen.generate(user_id="u1", text="hi", voice=None)


@pytest.mark.asyncio
async def test_generate_rejects_empty_text():
    gen = TTSGenerator(
        api_key="sk-test",
        model="tts-1",
        default_voice="onyx",
        daily_char_cap=1000,
    )
    with pytest.raises(TextRequired):
        await gen.generate(user_id="u1", text="", voice=None)
    with pytest.raises(TextRequired):
        await gen.generate(user_id="u1", text="    ", voice=None)


@pytest.mark.asyncio
async def test_generate_rejects_over_max_length():
    gen = TTSGenerator(
        api_key="sk-test",
        model="tts-1",
        default_voice="onyx",
        daily_char_cap=10_000,
    )
    with pytest.raises(TextTooLong):
        await gen.generate(user_id="u1", text="x" * 5000, voice=None)


@pytest.mark.asyncio
async def test_generate_rejects_unknown_voice():
    gen = TTSGenerator(
        api_key="sk-test",
        model="tts-1",
        default_voice="onyx",
        daily_char_cap=1000,
    )
    with pytest.raises(UnknownVoice):
        await gen.generate(user_id="u1", text="hi", voice="elvis")


@pytest.mark.asyncio
async def test_default_voice_in_supported_set():
    """Sanity: the configured default voice has to actually be in
    OpenAI's supported set. Catches a typo before it ships. Cedar
    is the current default per config.py (only available on
    gpt-4o-mini-tts, not tts-1).
    """
    assert "cedar" in SUPPORTED_VOICES


def test_quota_rollover_at_new_utc_day():
    """A counter from yesterday should reset to 0 when today's
    request lands — the day field is the rollover marker.
    """
    q = _Quota()
    yesterday = date.today() - timedelta(days=1)
    q.counters["u1"] = _DailyCounter(day=yesterday, chars=999)
    # Today's first request should succeed with chars=10 even
    # though yesterday left u1 at 999/1000.
    q.reserve(user_id="u1", chars=10, daily_cap=1000)
    # The post-reserve count reflects only today's usage.
    assert q.counters["u1"].chars == 10


def test_quota_accumulates_within_day():
    q = _Quota()
    q.reserve(user_id="u1", chars=100, daily_cap=1000)
    q.reserve(user_id="u1", chars=200, daily_cap=1000)
    assert q.counters["u1"].chars == 300


def test_quota_rejects_when_charge_exceeds_cap():
    q = _Quota()
    q.reserve(user_id="u1", chars=900, daily_cap=1000)
    # 900 + 200 = 1100 > 1000 → reject; the counter doesn't move.
    with pytest.raises(QuotaExceeded):
        q.reserve(user_id="u1", chars=200, daily_cap=1000)
    assert q.counters["u1"].chars == 900


def test_quota_is_per_user():
    """One user's quota doesn't affect another's."""
    q = _Quota()
    q.reserve(user_id="u1", chars=950, daily_cap=1000)
    # u2 starts fresh.
    q.reserve(user_id="u2", chars=500, daily_cap=1000)
    assert q.counters["u1"].chars == 950
    assert q.counters["u2"].chars == 500
