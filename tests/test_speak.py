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
    OpenAI's supported set. Catches a typo before it ships.
    """
    assert "onyx" in SUPPORTED_VOICES


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
