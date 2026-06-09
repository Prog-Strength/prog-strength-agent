"""Voice-streaming helpers + orchestrator: split a streaming text
buffer into TTS-able sentences, strip Markdown before TTS, and wrap
an SSE chat stream with per-sentence audio_chunk events.

Pure helpers (pop_complete_sentences, strip_markdown) live here for
unit testing without the FastAPI stack. The voice_streamer
orchestrator also lives here so tests can exercise it directly with
a fake TTSGenerator — server.py just imports it and wires it in.

See prog-strength-docs/sows/streaming-tts.md.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from prog_strength_agent.speak import TTSGenerator

log = logging.getLogger(__name__)

# Abbreviations whose trailing period should NOT end a sentence. Keep
# this list small — sentence boundary detection is heuristic and a
# missed split (one TTS call instead of two) is much less bad than a
# wrong split (TTS reading "Dr" without the rest of the thought).
# Comparison is lowercase; preserve the period in the lookup.
_ABBREVIATIONS = frozenset(
    {
        "dr.",
        "mr.",
        "mrs.",
        "ms.",
        "e.g.",
        "i.e.",
        "vs.",
        "etc.",
        "st.",
        "jr.",
        "sr.",
    }
)

# Matches sentence-ending punctuation followed by whitespace. End-
# of-buffer is intentionally NOT a boundary — the next delta may
# extend the sentence, so we keep "punctuation at end of buffer" in
# the remainder. The caller flushes the trailing remainder
# separately at stream end via end-of-stream code (see SOW).
#
# Group 1 captures the terminator chars so we can preserve them on
# the sentence (TTS keys off them for intonation).
_BOUNDARY = re.compile(r"([.!?]+)(\s+)")


def pop_complete_sentences(buffer: str) -> tuple[list[str], str]:
    """Walk `buffer` left-to-right, splitting off complete sentences.

    A sentence is "complete" when it ends in `.`/`!`/`?` followed by
    whitespace, AND the trailing token doesn't look like an
    abbreviation we recognize. The remainder (text past the last
    boundary) is returned for the next call to consume — the caller
    accumulates more deltas before retrying.

    The final partial sentence at end-of-stream is NOT included in
    the completed list; the caller is expected to flush it via a
    separate end-of-stream code path so the heuristic stays
    monotonic ("did this boundary close the sentence yet?") without
    a special "this is the last delta" flag.

    Examples:
        pop_complete_sentences("Hello world.")
        -> ([], "Hello world.")              # no trailing whitespace yet
        pop_complete_sentences("Hello world. Next")
        -> (["Hello world."], "Next")
        pop_complete_sentences("Visit Dr. Smith. Today")
        -> (["Visit Dr. Smith."], "Today")    # 'Dr.' is in the allowlist
    """
    sentences: list[str] = []
    pos = 0
    for match in _BOUNDARY.finditer(buffer):
        # Look at the word ending at the boundary. If it's an
        # abbreviation we recognize, keep scanning past this match.
        candidate = buffer[pos : match.end(1)]
        word_match = re.search(r"\S+$", candidate)
        last_word = word_match.group(0).lower() if word_match else ""
        if last_word in _ABBREVIATIONS:
            continue
        end_of_sep = match.end(2)
        sentence = buffer[pos:end_of_sep].rstrip()
        if sentence:
            sentences.append(sentence)
        pos = end_of_sep
    return sentences, buffer[pos:]


# Markdown stripping regexes. Order matters: fenced code first (so
# inline-code rules don't half-process a fence), then inline markers.
_FENCED_CODE = re.compile(r"```[\s\S]*?```")
_INLINE_CODE = re.compile(r"`([^`]+)`")
_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_ITALIC_STAR = re.compile(r"\*([^*]+)\*")
_BOLD_UNDER = re.compile(r"__([^_]+)__")
_ITALIC_UNDER = re.compile(r"_([^_]+)_")
_STRIKE = re.compile(r"~~([^~]+)~~")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_LEADING = re.compile(r"^[#>\-*+]\s+", re.MULTILINE)


def strip_markdown(text: str) -> str:
    """Remove Markdown markers from `text`, leaving speakable prose.

    Lossy by design — we're not trying to preserve Markdown
    semantics, we're producing something a TTS engine can read aloud
    without saying "asterisk hello asterisk" or reading a code block
    character-by-character.

    Strategy:
      - Code fences get deleted entirely (don't TTS code)
      - Inline code: keep the content, drop the backticks
      - Bold / italic / strikethrough: keep content, drop markers
      - Links: keep the link text, drop the URL
      - Line-leading list/heading/quote markers stripped
    """
    out = _FENCED_CODE.sub("", text)
    out = _INLINE_CODE.sub(r"\1", out)
    out = _BOLD.sub(r"\1", out)
    out = _BOLD_UNDER.sub(r"\1", out)
    out = _ITALIC_STAR.sub(r"\1", out)
    out = _ITALIC_UNDER.sub(r"\1", out)
    out = _STRIKE.sub(r"\1", out)
    out = _LINK.sub(r"\1", out)
    out = _LEADING.sub("", out)
    return out.strip()


# --- SSE encode/decode --------------------------------------------


def sse_event(payload: dict[str, Any]) -> bytes:
    """Format a payload as a single SSE event. Mirrors the same
    encoding model_harness uses for its own events — we re-encode
    here rather than reach into that module so the two stay
    decoupled and either can change formatting independently.
    """
    return f"data: {json.dumps(payload)}\n\n".encode()


def parse_sse_event(data: bytes) -> dict[str, Any] | None:
    """Parse one SSE event (as yielded by the chat stream) back into
    its JSON payload. Returns None when the bytes don't match the
    expected `data: <json>\\n\\n` shape — comments, multi-event
    chunks, malformed bytes etc. all fall through silently.
    """
    s = data.decode("utf-8", errors="replace")
    if not s.startswith("data: "):
        return None
    json_part = s[len("data: ") :].rstrip("\n")
    try:
        parsed = json.loads(json_part)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


# --- Streaming orchestrator ---------------------------------------


async def voice_streamer(
    inner: AsyncGenerator[bytes, None],
    *,
    user_id: str,
    tts: TTSGenerator,
    session_id: str | None = None,
) -> AsyncGenerator[bytes, None]:
    """Wrap the chat SSE stream with per-sentence TTS audio_chunks.

    Pass-through behavior for every upstream event (text_delta,
    model_chosen, tool_use_*, done, error). The only added behavior
    is: as text_delta events flow by, buffer their text; whenever a
    sentence boundary appears in the buffer, strip the Markdown and
    fire `tts.generate()` as an asyncio.create_task; once the
    upstream stream ends, drain those tasks in order and emit one
    audio_chunk SSE event per sentence's mp3.

    Tasks fire in parallel (so sentence 2's TTS overlaps sentence 1's
    playback) but yields are sequential — sentence N's audio_chunk
    is held until sentence N-1's has been yielded so the client's
    playback queue stays in order even when OpenAI completes the
    calls out of order. See
    prog-strength-docs/sows/streaming-tts.md.

    Degrades cleanly when OpenAI is unconfigured: if `tts.enabled`
    is False, this collapses to a plain pass-through and no
    audio_chunk events get emitted. The chat still works; voice
    mode just silently doesn't speak.
    """
    if not tts.enabled:
        async for chunk in inner:
            yield chunk
        return

    buffer = ""
    audio_tasks: list[tuple[int, str, asyncio.Task[bytes]]] = []
    index = 0

    try:
        async for chunk in inner:
            # Always pass the upstream event through unchanged — the
            # UI's text rendering rides on these events arriving with
            # the same shape and cadence as before.
            yield chunk

            event = parse_sse_event(chunk)
            if event is None or event.get("type") != "text_delta":
                continue
            text = event.get("text", "")
            if not isinstance(text, str) or not text:
                continue
            buffer += text
            sentences, buffer = pop_complete_sentences(buffer)
            for sentence in sentences:
                cleaned = strip_markdown(sentence)
                if not cleaned:
                    continue
                task = asyncio.create_task(
                    tts.generate(
                        user_id=user_id,
                        text=cleaned,
                        voice=None,
                        session_id=session_id,
                    )
                )
                audio_tasks.append((index, cleaned, task))
                index += 1

        # Stream ended — flush any trailing partial sentence as the
        # final TTS call. Without this, a reply that ends mid-thought
        # (or whose last sentence has no trailing whitespace) would
        # silently drop the last sentence's audio.
        if buffer.strip():
            cleaned = strip_markdown(buffer)
            if cleaned:
                task = asyncio.create_task(
                    tts.generate(
                        user_id=user_id,
                        text=cleaned,
                        voice=None,
                        session_id=session_id,
                    )
                )
                audio_tasks.append((index, cleaned, task))
                index += 1

        # Drain in order. The task at audio_tasks[i] may have finished
        # well before audio_tasks[i-1]; awaiting in list order is what
        # forces sequential emission to the client.
        for i, sentence_text, task in audio_tasks:
            try:
                mp3 = await task
            except Exception:
                # One sentence's TTS failure doesn't kill the turn —
                # log it, skip that audio_chunk, and continue with
                # the next sentence. The user hears uninterrupted
                # speech across the gap; the chat UI's text is
                # unaffected.
                log.exception(
                    "voice: TTS failed for sentence %d (user=%s)",
                    i,
                    user_id,
                )
                continue
            yield sse_event(
                {
                    "type": "audio_chunk",
                    "index": i,
                    "text": sentence_text,
                    "mp3_base64": base64.b64encode(mp3).decode("ascii"),
                }
            )
    finally:
        # If the client disconnects mid-stream, cancel any in-flight
        # OpenAI calls so we stop billing for audio nobody will hear.
        # asyncio.Task.cancel() is a no-op on already-completed
        # tasks, so this is safe in the happy path too.
        for _, _, task in audio_tasks:
            if not task.done():
                task.cancel()
