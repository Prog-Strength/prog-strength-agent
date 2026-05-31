"""End-to-end test for voice_streamer: confirms text deltas pass
through unchanged, audio_chunk events fire per sentence with the
right index/text/mp3, and the trailing-buffer flush emits the final
sentence's audio.

We don't exercise the OpenAI HTTP call — TTSGenerator gets a fake
that returns deterministic bytes per (user_id, text) pair so we can
assert exact ordering and content.
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

import pytest

from prog_strength_agent.voice_stream import sse_event, voice_streamer


class FakeTTS:
    """Stand-in for TTSGenerator. Returns `text.encode()` as the
    "mp3" bytes so tests can assert exactly which sentence each
    audio_chunk corresponds to without decoding real audio. An
    optional delay map lets a test force out-of-order completion to
    exercise the in-order yield guarantee.
    """

    enabled = True

    def __init__(self, delays: dict[str, float] | None = None) -> None:
        self.calls: list[str] = []
        self.delays = delays or {}

    async def generate(
        self, *, user_id: str, text: str, voice: str | None
    ) -> bytes:
        self.calls.append(text)
        if text in self.delays:
            await asyncio.sleep(self.delays[text])
        return text.encode()


async def _drain(inner_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Helper: feed `inner_events` into voice_streamer as SSE bytes,
    collect the yielded events back as parsed dicts.
    """

    async def inner() -> Any:
        for event in inner_events:
            yield sse_event(event)

    fake = FakeTTS()
    out: list[dict[str, Any]] = []
    async for chunk_bytes in voice_streamer(
        inner(), user_id="u1", tts=fake  # type: ignore[arg-type]
    ):
        text = chunk_bytes.decode()
        assert text.startswith("data: ")
        payload = json.loads(text[len("data: ") :].rstrip("\n"))
        out.append(payload)
    return out


@pytest.mark.asyncio
async def test_passes_text_deltas_through_unchanged():
    """Every text_delta the upstream yielded must appear in the
    output in the same order. Voice streaming is purely additive on
    top of the existing chat events.
    """
    events = [
        {"type": "model_chosen", "model": "claude-haiku-4-5"},
        {"type": "text_delta", "text": "Hello"},
        {"type": "text_delta", "text": " world."},
        {"type": "done", "stop_reason": "end_turn"},
    ]
    out = await _drain(events)
    text_deltas = [e for e in out if e["type"] == "text_delta"]
    assert text_deltas == [
        {"type": "text_delta", "text": "Hello"},
        {"type": "text_delta", "text": " world."},
    ]


@pytest.mark.asyncio
async def test_emits_audio_chunk_per_complete_sentence():
    """One audio_chunk per sentence, in stream order. The fake TTS
    returns the sentence text as 'mp3' bytes so we can verify the
    chunks correspond to the right source text.
    """
    events = [
        {"type": "text_delta", "text": "First sentence. "},
        {"type": "text_delta", "text": "Second one! "},
        {"type": "text_delta", "text": "Third?"},
        {"type": "done", "stop_reason": "end_turn"},
    ]
    out = await _drain(events)
    chunks = [e for e in out if e["type"] == "audio_chunk"]
    assert len(chunks) == 3
    assert [c["index"] for c in chunks] == [0, 1, 2]
    assert chunks[0]["text"] == "First sentence."
    assert chunks[1]["text"] == "Second one!"
    assert chunks[2]["text"] == "Third?"
    for c in chunks:
        decoded = base64.b64decode(c["mp3_base64"]).decode()
        assert decoded == c["text"]


@pytest.mark.asyncio
async def test_flushes_trailing_buffer_at_stream_end():
    """If the upstream stream ends without a trailing whitespace
    after the last sentence's punctuation, that sentence is still in
    the buffer when the upstream loop exits — the streamer must
    flush it as a final audio_chunk so the last sentence's audio
    isn't silently dropped.
    """
    events = [
        {"type": "text_delta", "text": "Just one thing"},
        {"type": "text_delta", "text": " to say."},
        {"type": "done", "stop_reason": "end_turn"},
    ]
    out = await _drain(events)
    chunks = [e for e in out if e["type"] == "audio_chunk"]
    assert len(chunks) == 1
    assert chunks[0]["text"] == "Just one thing to say."


@pytest.mark.asyncio
async def test_strips_markdown_before_tts():
    """The TTS-bound text should have Markdown markers stripped so
    the user doesn't hear "asterisk hello asterisk".
    """
    events = [
        {"type": "text_delta", "text": "You got this **bro**! "},
        {"type": "text_delta", "text": "Try `bench-press` next."},
        {"type": "done", "stop_reason": "end_turn"},
    ]
    out = await _drain(events)
    chunks = [e for e in out if e["type"] == "audio_chunk"]
    assert len(chunks) == 2
    assert chunks[0]["text"] == "You got this bro!"
    assert chunks[1]["text"] == "Try bench-press next."


@pytest.mark.asyncio
async def test_audio_chunks_emit_in_order_despite_out_of_order_completion():
    """Sentence 2's TTS completes before sentence 1's (simulated via
    a delay). Even so, audio_chunk events must yield in source order:
    index 0 before index 1.
    """
    events = [
        {"type": "text_delta", "text": "Slow first. "},
        {"type": "text_delta", "text": "Fast second."},
        {"type": "done", "stop_reason": "end_turn"},
    ]

    async def inner() -> Any:
        for e in events:
            yield sse_event(e)

    fake = FakeTTS(delays={"Slow first.": 0.05})
    chunks: list[dict[str, Any]] = []
    async for raw in voice_streamer(
        inner(), user_id="u1", tts=fake  # type: ignore[arg-type]
    ):
        s = raw.decode()
        if not s.startswith("data: "):
            continue
        payload = json.loads(s[len("data: ") :].rstrip("\n"))
        if payload.get("type") == "audio_chunk":
            chunks.append(payload)

    assert [c["index"] for c in chunks] == [0, 1]
    assert chunks[0]["text"] == "Slow first."
    assert chunks[1]["text"] == "Fast second."


@pytest.mark.asyncio
async def test_disabled_tts_falls_through_to_plain_stream():
    """If TTS is disabled (no OpenAI key in prod), voice_streamer
    should be a no-op pass-through. Voice mode silently doesn't speak
    but the chat keeps working.
    """
    events = [
        {"type": "text_delta", "text": "Hello world."},
        {"type": "done", "stop_reason": "end_turn"},
    ]

    async def inner() -> Any:
        for e in events:
            yield sse_event(e)

    class DisabledTTS:
        enabled = False

        async def generate(self, **_: Any) -> bytes:  # pragma: no cover
            raise AssertionError("should not be called when disabled")

    out: list[dict[str, Any]] = []
    async for raw in voice_streamer(
        inner(), user_id="u1", tts=DisabledTTS()  # type: ignore[arg-type]
    ):
        s = raw.decode()
        if s.startswith("data: "):
            out.append(json.loads(s[len("data: ") :].rstrip("\n")))

    assert out == events


@pytest.mark.asyncio
async def test_tts_failure_for_one_sentence_skips_that_chunk_only():
    """If OpenAI 429s on sentence 3, the user shouldn't lose the rest
    of the turn's audio. The failing sentence is dropped; surrounding
    sentences still get audio_chunks in order.
    """

    async def inner() -> Any:
        for e in [
            {"type": "text_delta", "text": "First. Second. Third."},
            {"type": "done", "stop_reason": "end_turn"},
        ]:
            yield sse_event(e)

    class FlakyTTS:
        enabled = True

        async def generate(self, *, text: str, **_: Any) -> bytes:
            if text == "Second.":
                raise RuntimeError("openai 429")
            return text.encode()

    chunks: list[dict[str, Any]] = []
    async for raw in voice_streamer(
        inner(), user_id="u1", tts=FlakyTTS()  # type: ignore[arg-type]
    ):
        s = raw.decode()
        if not s.startswith("data: "):
            continue
        payload = json.loads(s[len("data: ") :].rstrip("\n"))
        if payload.get("type") == "audio_chunk":
            chunks.append(payload)

    assert [c["text"] for c in chunks] == ["First.", "Third."]
    # Indices stay sequential — the failed sentence's index slot is
    # just skipped, not renumbered, so a debugger can correlate the
    # gap with the log entry.
    assert [c["index"] for c in chunks] == [0, 2]
