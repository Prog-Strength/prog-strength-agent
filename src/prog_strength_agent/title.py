"""Chat-session title generation via Haiku.

The web + mobile clients call POST /title after the first turn of a
fresh chat session, then PATCH the returned title onto the session
in the Go API. The endpoint is intentionally separate from /chat:
it's synchronous (no streaming), doesn't touch MCP, and runs against
a tighter prompt than the main coach persona. See
prog-strength-docs/sows/persistent-chat-sessions.md.

Cost: one Haiku call, max_tokens≈30. Fractions of a cent per title.
Latency: ~500–1000ms p95.

Failure mode: any exception returns a fallback title derived from
the user's first message. The client also has its own fallback, so
this path is mostly defense-in-depth — the client doesn't need to
know the call failed to ship a usable title.
"""

import logging
import re
from typing import Any

from anthropic import AsyncAnthropic

from prog_strength_agent.prompt import TITLE_SYSTEM_PROMPT

log = logging.getLogger(__name__)

# Matches the chat.MaxTitleLen constant on the API side. Haiku
# occasionally over-runs the requested word count; we truncate
# server-side rather than rejecting so the client always gets a
# usable title.
MAX_TITLE_LEN = 80

# Fallback length when we have to slice the user's first message.
# Shorter than MAX_TITLE_LEN so the truncated message reads as a
# title rather than a sentence fragment.
FALLBACK_LEN = 60


class TitleGenerator:
    """Asks Haiku for a 3–6 word title summarizing a chat turn.

    Mirrors ModelRouter's shape: a small class wrapping the shared
    AsyncAnthropic client + a model id, with one `generate` async
    method the HTTP layer calls.
    """

    def __init__(self, client: AsyncAnthropic, model: str):
        self.client = client
        self.model = model

    async def generate(self, messages: list[dict[str, Any]]) -> str:
        """Return a normalized title for the given conversation.

        Always returns a non-empty string ≤ MAX_TITLE_LEN. Never
        raises; on any failure falls back to the user's first
        message truncated to FALLBACK_LEN, or "New Chat" if there
        isn't even a usable first message.
        """
        fallback = _fallback_title(messages)
        if not messages:
            return fallback

        try:
            resp = await self.client.messages.create(
                model=self.model,
                # Generous cap: 6 words × ~6 tokens/word + tokenizer
                # overhead. We post-truncate to MAX_TITLE_LEN chars so
                # max_tokens just bounds runaway outputs.
                max_tokens=40,
                system=TITLE_SYSTEM_PROMPT,
                messages=_render_for_title(messages),
            )
        except Exception:
            log.exception("title generation call failed")
            return fallback

        raw = _first_text_block(resp.content)
        cleaned = _clean_title(raw)
        if not cleaned:
            return fallback
        return cleaned[:MAX_TITLE_LEN]


def _render_for_title(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reshape the chat-format messages into an Anthropic Messages
    payload for the title prompt.

    Earlier revisions flattened the whole conversation into a single
    labeled "User: …\\nAssistant: …" user turn — Haiku saw it as
    "someone is showing me a transcript" rather than "this is the
    conversation I'm meant to title", and leaned heavily on the
    fallback. Passing the actual user/assistant alternation followed
    by an explicit instruction-shaped user turn produces real topic
    titles instead.

    Tool-use blocks and other non-text content are dropped — they
    aren't useful for titling and aren't worth the role-alternation
    headache they introduce when the agent sent multiple assistant
    blocks in a row inside one turn.
    """
    rendered: list[dict[str, Any]] = []
    last_role: str | None = None
    for msg in messages:
        role = msg.get("role", "")
        if role not in ("user", "assistant"):
            continue
        text = _flatten_text(msg.get("content"))
        if not text:
            continue
        # Anthropic requires strict user/assistant alternation; merge
        # adjacent same-role turns into one block so we don't 400.
        if role == last_role and rendered:
            rendered[-1]["content"] += "\n\n" + text
        else:
            rendered.append({"role": role, "content": text})
            last_role = role

    if not rendered:
        # Shouldn't happen in the live flow (clients only call /title
        # after a real turn), but defend against an empty payload by
        # synthesizing a user turn so the API call has something
        # well-formed to send.
        rendered.append({"role": "user", "content": "(no conversation)"})

    # If the conversation ended on the assistant (the live case — we
    # call /title right after a user/assistant pair), append a final
    # user-role instruction so Haiku knows exactly what to produce.
    # When it already ends on user (rare; only if a malformed payload
    # slips through), merge the instruction into the last turn rather
    # than violating the alternation rule.
    instruction = (
        "Now write a short title (3–6 words, title case, no quotes, no "
        "punctuation) that captures the topic of the conversation above."
    )
    if rendered[-1]["role"] == "assistant":
        rendered.append({"role": "user", "content": instruction})
    else:
        rendered[-1]["content"] += "\n\n" + instruction

    return rendered


def _flatten_text(content: Any) -> str:
    """Pull plain text out of a message's content. Handles the two
    Anthropic shapes (str or list of typed blocks) the same way
    ModelRouter does — tool_result and other non-text blocks are
    dropped since they aren't useful for titling.
    """
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return " ".join(parts).strip()
    return ""


def _first_text_block(blocks: list[Any]) -> str:
    """Extract the first text block from an Anthropic response.
    Returns "" when the response has no text content."""
    for block in blocks:
        if getattr(block, "type", None) == "text":
            return getattr(block, "text", "") or ""
    return ""


# Haiku sometimes wraps the title in quotes or appends a period
# despite the instruction not to. Strip those plus collapse runs of
# whitespace. Anything left after stripping is the title we PATCH.
_TRAILING_PUNCT = re.compile(r"[\s.,;:!?]+$")
_LEADING_PUNCT = re.compile(r"^[\s.,;:!?]+")
_INNER_WHITESPACE = re.compile(r"\s+")


def _clean_title(raw: str) -> str:
    s = raw.strip()
    if not s:
        return ""
    # Strip a wrapping pair of quotes. Both straight ('", ") and curly
    # (“…”, ‘…’) — Haiku occasionally hands back the curly variants
    # when asked to summarize. Done as one shave-from-both-ends pass so
    # mismatched curly pairs ("…") still get caught.
    quote_chars = '"\'“”‘’'
    while len(s) >= 2 and s[0] in quote_chars and s[-1] in quote_chars:
        s = s[1:-1].strip()
    s = _LEADING_PUNCT.sub("", s)
    s = _TRAILING_PUNCT.sub("", s)
    s = _INNER_WHITESPACE.sub(" ", s)
    return s


def _fallback_title(messages: list[dict[str, Any]]) -> str:
    """Derive a title from the first user message when the LLM call
    fails or returns junk. Mirrors the page-side fallback in the
    web/mobile clients so the eventual stored title looks the same
    regardless of which side caught the failure.
    """
    for msg in messages:
        if msg.get("role") != "user":
            continue
        text = _flatten_text(msg.get("content"))
        if text:
            cleaned = _clean_title(text)
            if cleaned:
                return cleaned[:FALLBACK_LEN]
    return "New Chat"
