"""FastAPI app: /health + /chat (SSE streaming, tiered model routing).

Request flow:

  1. Authenticate the JWT and extract the raw token.
  2. ModelRouter classifies the latest user message → "simple" or
     "complex" tier (one Haiku call, ~500ms, ~$0.0001).
  3. The corresponding ModelHarness opens an MCP session with the
     user's JWT in the Authorization header, runs the tool-use loop,
     and streams SSE events to the client.

The first SSE event is always `model_chosen` so the frontend can
label the assistant turn. Subsequent events are unchanged from the
single-model implementation: text_delta, tool_use_start, tool_result,
done, and error.

CORS allows the configured frontend origins (Vercel-hosted in prod).
"""

import logging
from collections.abc import AsyncGenerator
from typing import Any

from anthropic import AsyncAnthropic
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from prog_strength_agent.auth import authenticate
from prog_strength_agent.config import Config
from prog_strength_agent.model_harness import ModelHarness
from prog_strength_agent.model_router import ModelRouter
from prog_strength_agent.prompt import build_chat_system_prompt
from prog_strength_agent.speak import SpeakError, TTSGenerator
from prog_strength_agent.telemetry import (
    AGENT_VOICE_TIME_TO_FIRST_AUDIO_SECONDS,
    TelemetryClient,
    TurnInstrumentation,
    record_prometheus_metrics,
)
from prog_strength_agent.title import TitleGenerator
from prog_strength_agent.version import SERVICE, VERSION
from prog_strength_agent.voice_stream import voice_streamer

log = logging.getLogger(__name__)

# Single shared state. uvicorn's worker process is long-lived; these
# get constructed once at import time and reused for the lifetime of
# the process. The Anthropic client is model-agnostic so a single
# instance backs every harness + the router.
config = Config.from_env()
claude = AsyncAnthropic(api_key=config.anthropic_api_key)

# One ModelHarness per tier. The router returns a tier key; the dict
# lookup picks the harness. Adding a new tier (e.g. "vision",
# "research") is a matter of adding a model id to Config and an entry
# here — no /chat changes.
HARNESSES: dict[str, ModelHarness] = {
    "simple": ModelHarness(
        client=claude,
        model=config.simple_model,
        mcp_url=config.mcp_url,
        max_tokens=config.max_tokens,
    ),
    "complex": ModelHarness(
        client=claude,
        model=config.complex_model,
        mcp_url=config.mcp_url,
        max_tokens=config.max_tokens,
    ),
}
router_obj = ModelRouter(client=claude, router_model=config.router_model)

# TitleGenerator reuses the cheap simple-tier model (Haiku) since
# title summarization is a fixed-cost classification task that
# doesn't benefit from Sonnet. Same shared AsyncAnthropic client as
# the harnesses + router — no extra connection pool.
title_generator = TitleGenerator(client=claude, model=config.simple_model)

# TTSGenerator owns the OpenAI client + the per-user daily char
# counter. Lives as a module-level singleton so the counter survives
# across requests; uvicorn workers are one process at our scale.
# See prog-strength-docs/sows/voice-chat.md.
tts_generator = TTSGenerator(
    api_key=config.openai_api_key,
    model=config.openai_tts_model,
    default_voice=config.tts_voice_default,
    daily_char_cap=config.tts_daily_char_cap_per_user,
    instructions=config.tts_instructions,
)

# Telemetry client: fire-and-forget POSTs to the API's internal
# endpoints. Disabled when api_url is empty (local dev without the
# API container running) — chat keeps working, telemetry just
# doesn't get written.
telemetry_client: TelemetryClient | None = (
    TelemetryClient(api_base_url=config.api_url) if config.api_url else None
)


app = FastAPI(title=SERVICE, version=VERSION)

# Prometheus /metrics: scrape target reachable only from inside the
# Docker network (Caddy refuses to proxy /metrics to the public
# internet). expose() must be called before any routes are added or
# the route registration order will hide /metrics behind the auth
# middleware once we add one — call it eagerly at import.
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# CORS for the frontend on Vercel (or wherever). The /chat endpoint is
# called cross-origin from the browser, so without these headers the
# request never leaves. Bearer auth is in the Authorization header,
# not cookies, so we do NOT enable allow_credentials.
if config.cors_allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.cors_allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Accept"],
    )


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe. Mirrors the API and MCP `/health` envelope shape
    so the same curl muscle memory works across all three services.
    """
    return {"service": SERVICE, "version": VERSION, "message": "service is healthy"}


class ChatMessage(BaseModel):
    """Anthropic-format message. Content can be a string (user turns)
    or a list of typed blocks (assistant turns, tool_result follow-ups).
    """

    role: str
    content: str | list[dict[str, Any]]


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    # Client-generated UUID identifying the chat conversation. When
    # absent the server generates one inside TurnInstrumentation.new
    # so no turn lands in telemetry without a session. The frontend
    # is the canonical generator; absence is a fallback for scripted
    # callers and older clients.
    session_id: str | None = None
    # IANA timezone name (e.g. "America/Denver") detected by the
    # client via Intl.DateTimeFormat().resolvedOptions().timeZone.
    # Used to compute the user's local date for the system-prompt
    # prefix that grounds the model in "what day is it today" —
    # see prompt.build_chat_system_prompt. Falls back to UTC when
    # missing or unrecognized, so older clients keep working.
    client_timezone: str | None = None
    # When true, the server runs the streaming-TTS pipeline alongside
    # the existing text streaming: text deltas pass through unchanged
    # AND new audio_chunk SSE events carry per-sentence mp3 audio for
    # the client to play in order. False (or missing) preserves
    # today's behavior — clients without voice mode see no behavior
    # change. See prog-strength-docs/sows/streaming-tts.md.
    voice_mode: bool = False


@app.post("/chat")
async def chat(req: ChatRequest, request: Request) -> StreamingResponse:
    """SSE-stream a chat turn.

    Auth happens up-front so a 401 surfaces as a normal HTTP status
    before the stream starts. Routing happens lazily inside the
    generator so the model_chosen SSE event can be the first byte
    the client sees — that way the UI shows "via Haiku" without an
    extra HTTP request.
    """
    auth = authenticate(request, config.jwt_signing_key)
    messages: list[dict[str, Any]] = [m.model_dump() for m in req.messages]

    telemetry = TurnInstrumentation.new(
        user_id=auth.user_id, session_id=req.session_id
    )

    # Build the per-request system prompt with the user's local date
    # prefixed. Done here (not inside the harness) so the prompt logic
    # stays out of the model-loop code path — harness takes the prompt
    # as-is and ships it to Anthropic. See
    # prompt.build_chat_system_prompt for the date-prefix rationale.
    system_prompt = build_chat_system_prompt(req.client_timezone)

    inner = _route_and_stream(
        messages, auth.token, telemetry, system_prompt
    )
    # When voice_mode is on, wrap the SSE stream with a voice_streamer
    # that buffers text deltas, detects sentence boundaries, fires
    # TTS for each sentence in parallel, and emits audio_chunk events
    # alongside the original text deltas. Off → no buffering, no TTS,
    # behavior is identical to the pre-streaming-tts /chat.
    stream = (
        voice_streamer(inner, user_id=auth.user_id, tts=tts_generator)
        if req.voice_mode
        else inner
    )

    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={
            # Prevent intermediaries (Caddy, browsers, proxies) from
            # buffering — we want bytes flushed as they're produced so
            # the UI sees tokens arrive live.
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


class TitleRequest(BaseModel):
    """Body for POST /title. Same ChatMessage shape /chat accepts —
    typically the first user message + first assistant reply from a
    fresh chat session, but the endpoint accepts any conversation
    excerpt the client wants summarized.
    """

    messages: list[ChatMessage]


class TitleResponse(BaseModel):
    title: str


@app.post("/title")
async def title(req: TitleRequest, request: Request) -> TitleResponse:
    """Generate a 3–6 word title for the conversation via Haiku.

    Synchronous (non-streaming) by design — the client fires this
    fire-and-forget after the first turn lands, then PATCHes the
    returned title onto the API's chat_sessions row. The endpoint
    never raises: TitleGenerator.generate falls back to a slice of
    the first user message on any error so the client always has
    something to PATCH.

    Auth uses the same JWT middleware /chat does. We don't persist
    the result here — the agent stays stateless; the client owns
    the PATCH against the API.
    """
    # Validate the JWT for parity with /chat. The user id isn't used
    # downstream (TitleGenerator is per-message-text only) but
    # gating the endpoint prevents an anonymous caller from burning
    # Haiku tokens.
    _ = authenticate(request, config.jwt_signing_key)
    messages: list[dict[str, Any]] = [m.model_dump() for m in req.messages]
    generated = await title_generator.generate(messages)
    return TitleResponse(title=generated)


class SpeakRequest(BaseModel):
    """Body for POST /speak. `text` is the assistant turn the client
    wants spoken; `voice` is an optional override of the configured
    default (must be one of the closed enum in speak.SUPPORTED_VOICES,
    or the endpoint returns 400).
    """

    text: str
    voice: str | None = None


@app.post("/speak")
async def speak(req: SpeakRequest, request: Request) -> Response:
    """Generate spoken audio for `text` via OpenAI TTS.

    Synchronous, non-streaming, returns the full mp3 payload after
    OpenAI finishes generating it. Clients call this when "voice
    mode" is on after each completed /chat stream — the assistant's
    text comes in, the audio bytes go back, the client plays them.

    Auth: same JWT middleware /chat + /title use. Per-user daily
    character cap is enforced before the OpenAI call so a malformed
    or runaway client can't drain the API budget.

    Failure modes map directly from SpeakError subclasses to HTTP
    statuses: text length / voice validity → 400, quota → 429,
    missing OpenAI key → 503, OpenAI SDK failure (rate limit,
    network) → 500 with a sanitized message.
    """
    auth = authenticate(request, config.jwt_signing_key)
    try:
        audio = await tts_generator.generate(
            user_id=auth.user_id,
            text=req.text,
            voice=req.voice,
        )
    except SpeakError as e:
        # The exception's status carries the intended HTTP code;
        # FastAPI turns this Response into the actual reply.
        return Response(
            content=str(e),
            status_code=e.status,
            media_type="text/plain; charset=utf-8",
        )
    except Exception as e:
        # OpenAI SDK / network errors. Sanitize the message so any
        # internal hints (org id, request id) the SDK might surface
        # don't leak to the client.
        log.exception("speak: OpenAI call failed")
        return Response(
            content=f"speak failed: {type(e).__name__}",
            status_code=500,
            media_type="text/plain; charset=utf-8",
        )
    return Response(
        content=audio,
        media_type="audio/mpeg",
        # mp3 bytes are immutable for a given (text, voice, model)
        # tuple but we don't expose a stable key the client could
        # cache against, so just disable caching everywhere.
        headers={"Cache-Control": "no-store"},
    )


class VoiceTelemetryRequest(BaseModel):
    """Body for POST /telemetry/voice. The client measures time-to-
    first-audio end-to-end (from "user pressed send" to "first
    audio_chunk's mp3 starts playing") and reports the result here
    once per turn after the first audio plays.
    """

    session_id: str | None = None
    time_to_first_audio_ms: float


@app.post("/telemetry/voice")
async def voice_telemetry(
    req: VoiceTelemetryRequest, request: Request
) -> dict[str, bool]:
    """Record one client-reported time-to-first-audio sample to the
    Prometheus histogram on the agent. Auth-gated with the same JWT
    middleware /chat + /speak use so the metric can't be poisoned
    by anonymous callers; the per-user-id label means a single
    misbehaving client only affects its own bucket.

    Session_id is accepted but not used as a label (high cardinality
    would blow up Prometheus). Useful for future correlation if we
    ever ship per-session debugging.

    See prog-strength-docs/sows/streaming-tts.md.
    """
    auth = authenticate(request, config.jwt_signing_key)
    AGENT_VOICE_TIME_TO_FIRST_AUDIO_SECONDS.labels(
        user_id=auth.user_id,
    ).observe(req.time_to_first_audio_ms / 1000.0)
    return {"ok": True}


async def _route_and_stream(
    messages: list[dict[str, Any]],
    user_token: str,
    telemetry: TurnInstrumentation,
    system_prompt: str,
) -> AsyncGenerator[bytes, None]:
    """Classify the request's tier, dispatch to the matching harness.

    The router's failure mode is "default to simple" (the cheaper
    tier), so even if the classifier call breaks, /chat keeps working
    — the user may just get a Haiku-level answer to a question that
    would have benefitted from Sonnet.

    After the stream completes (success or error), fires fire-and-forget
    telemetry POSTs to the API. Telemetry failures are logged but
    never raised — the chat already returned, and observability
    outages should not affect product behavior.
    """
    try:
        tier = await router_obj.route(messages, telemetry=telemetry)
        harness = HARNESSES.get(tier, HARNESSES["simple"])
        async for chunk in harness.stream_chat(
            messages,
            user_token,
            telemetry,
            system_prompt=system_prompt,
        ):
            yield chunk
    finally:
        # Live Prometheus counters first — synchronous, in-process,
        # no failure mode. The HTTP write to the API is fire-and-
        # forget after; a network glitch must not skip the metrics.
        record_prometheus_metrics(telemetry)
        if telemetry_client is not None:
            telemetry_client.record_turn(telemetry)


