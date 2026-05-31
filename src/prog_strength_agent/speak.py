"""Text-to-speech generation via OpenAI.

The web + mobile clients call POST /speak when "voice mode" is on:
after each completed assistant turn, the client sends the agent's
reply text here and plays the returned mp3 bytes. Same shape as
/title — synchronous, single LLM call, no MCP. Costs ~$15 per 1M
input characters with the `tts-1` model; the per-user daily cap
defined in Config bounds runaway-loop blast radius.

Anthropic doesn't have an audio surface yet, so this is the second
LLM-provider SDK the agent talks to. The OpenAI client lives here
to keep that boundary tight; server.py only wires the endpoint.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from openai import AsyncOpenAI

log = logging.getLogger(__name__)

# Closed set of voices we accept. The endpoint rejects anything
# outside this set with a 400 before the OpenAI call so a malformed
# client doesn't burn an API call to learn the constraint. The full
# set listed here matches gpt-4o-mini-tts (our default model); the
# older tts-1/tts-1-hd models support a strict subset (no cedar,
# marin, ballad, verse) — picking a model-incompatible voice will
# 400 from OpenAI itself, which is acceptable: the SOW expects ops
# to set OPENAI_TTS_MODEL and TTS_VOICE_DEFAULT consistently. Keep
# in sync with developers.openai.com/api/docs/guides/text-to-speech.
SUPPORTED_VOICES: frozenset[str] = frozenset(
    {
        "alloy", "ash", "ballad", "cedar", "coral", "echo", "fable",
        "marin", "nova", "onyx", "sage", "shimmer", "verse",
    }
)

# Max characters per individual /speak request. Lower than the per-user
# daily cap so a single misbehaving client can't drain the day's
# budget in one call. Anthropic replies almost never exceed this in
# practice; the client should ideally chunk on its end if it ever
# wants to speak a longer body.
MAX_TEXT_LEN = 4000


class SpeakError(Exception):
    """Base type for /speak-side errors that the handler maps to HTTP
    status codes. Subclasses carry the intended status — kept on the
    exception so the FastAPI handler can stay one tight try/except.
    """

    status: int = 500


class TextTooLong(SpeakError):
    status = 400


class TextRequired(SpeakError):
    status = 400


class UnknownVoice(SpeakError):
    status = 400


class QuotaExceeded(SpeakError):
    status = 429


class ServiceDisabled(SpeakError):
    """Raised when no OpenAI API key is configured. The endpoint
    returns 503 — useful in local dev where the key isn't on hand
    but you still want the agent to boot.
    """

    status = 503


@dataclass
class _DailyCounter:
    """One user's running character count for the current UTC day.
    The day field acts as the rollover marker — when a new request
    arrives on a later UTC date, we reset the count to 0.
    """

    day: date
    chars: int = 0


@dataclass
class _Quota:
    """In-memory per-user quota tracker. Keyed by user_id. Thread-safe
    via a single Lock (uvicorn workers are one process at our scale;
    if we ever fan out across workers we'd need to lift this into
    Redis or the API's SQLite).
    """

    counters: dict[str, _DailyCounter] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def reserve(self, user_id: str, chars: int, daily_cap: int) -> None:
        """Try to charge `chars` against `user_id`'s daily quota.
        Raises QuotaExceeded if the post-charge total would exceed
        the cap. The charge is committed atomically inside the lock.
        """
        today = _utc_today()
        with self.lock:
            current = self.counters.get(user_id)
            if current is None or current.day != today:
                current = _DailyCounter(day=today, chars=0)
                self.counters[user_id] = current
            if current.chars + chars > daily_cap:
                raise QuotaExceeded(
                    f"daily TTS character cap reached ({daily_cap} chars/day)"
                )
            current.chars += chars


def _utc_today() -> date:
    """Anchored on UTC so the rollover is predictable regardless of
    where the user (or the agent's host) lives.
    """
    return datetime.now(UTC).date()


class TTSGenerator:
    """Calls OpenAI TTS with the configured model + voice and a
    per-user daily char cap. Mirrors TitleGenerator's shape so the
    server.py wiring stays consistent: one class wrapping the SDK
    client + the rate-limit state, with one async `generate` method
    the HTTP layer calls.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        default_voice: str,
        daily_char_cap: int,
        instructions: str = "",
    ):
        self._daily_char_cap = daily_char_cap
        self._model = model
        self._default_voice = default_voice
        # Personality / pacing cue passed to gpt-4o-mini-tts. Empty
        # string falls through as "no instructions" — the older
        # tts-1 model doesn't accept the parameter anyway, and the
        # neutral delivery is fine as a fallback.
        self._instructions = instructions
        self._quota = _Quota()
        # Empty key means the endpoint is disabled. We track it
        # rather than failing startup so the agent boots cleanly in
        # local dev without OPENAI_API_KEY on hand.
        self._client: AsyncOpenAI | None = (
            AsyncOpenAI(api_key=api_key) if api_key else None
        )

    @property
    def default_voice(self) -> str:
        return self._default_voice

    @property
    def enabled(self) -> bool:
        return self._client is not None

    async def generate(
        self,
        *,
        user_id: str,
        text: str,
        voice: str | None,
    ) -> bytes:
        """Validate inputs, charge the per-user quota, call OpenAI,
        return the mp3 bytes. Raises a SpeakError subclass on any
        failure — the FastAPI handler maps the subclass to an HTTP
        status. The OpenAI client itself can also raise; those
        propagate as a 500 via the handler's catch-all.
        """
        if self._client is None:
            raise ServiceDisabled("TTS is not configured (OPENAI_API_KEY unset)")
        if not text or not text.strip():
            raise TextRequired("text is required")
        if len(text) > MAX_TEXT_LEN:
            raise TextTooLong(
                f"text exceeds {MAX_TEXT_LEN}-character per-request cap"
            )
        chosen_voice = voice or self._default_voice
        if chosen_voice not in SUPPORTED_VOICES:
            raise UnknownVoice(
                f"voice {chosen_voice!r} not supported; choose from {sorted(SUPPORTED_VOICES)}"
            )
        self._quota.reserve(
            user_id=user_id,
            chars=len(text),
            daily_cap=self._daily_char_cap,
        )

        # The OpenAI SDK's streaming-response context manager hands us
        # the raw bytes for free — we just collect and return them.
        # No streaming back to the client in v1 (the SOW marks
        # streaming TTS as a non-goal).
        #
        # `instructions` is only sent when non-empty: the older
        # tts-1/tts-1-hd models reject the field, so passing a blank
        # string would 400 on those. With gpt-4o-mini-tts (the
        # current default) it carries the personality cue.
        log.info(
            "speak: user=%s voice=%s chars=%d", user_id, chosen_voice, len(text)
        )
        kwargs: dict[str, object] = {
            "model": self._model,
            "voice": chosen_voice,
            "input": text,
            "response_format": "mp3",
        }
        if self._instructions:
            kwargs["instructions"] = self._instructions
        async with self._client.audio.speech.with_streaming_response.create(
            **kwargs,
        ) as response:
            return await response.read()
