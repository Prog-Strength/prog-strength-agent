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

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any

from anthropic import AsyncAnthropic
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from prog_strength_agent.api_client import APIClient
from prog_strength_agent.auth import authenticate
from prog_strength_agent.config import Config
from prog_strength_agent.model_harness import ModelHarness
from prog_strength_agent.model_router import ModelRouter
from prog_strength_agent.prompt import build_chat_system_prompt
from prog_strength_agent.request_id import (
    RequestIDMiddleware,
    configure_logging,
    current_request_id,
)
from prog_strength_agent.speak import SpeakError, TTSGenerator
from prog_strength_agent.telemetry import (
    AGENT_VOICE_TIME_TO_FIRST_AUDIO_SECONDS,
    TelemetryClient,
    TurnInstrumentation,
    record_prometheus_metrics,
)
from prog_strength_agent.title import TitleGenerator
from prog_strength_agent.usage_gate import CapExceeded, UsageGate
from prog_strength_agent.version import SERVICE, VERSION
from prog_strength_agent.voice_stream import voice_streamer

log = logging.getLogger(__name__)

# Install the request-id-aware log formatter before anything logs, so
# every line the agent emits carries the request's correlation id (or
# "-" when logged outside a request). See request_id.configure_logging.
configure_logging()

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

# Vision turns force the capable tier (Sonnet 4.6 today) from the same
# config slot the `complex` harness pins — this guarantees a turn that
# needs to see never lands on Haiku. We reuse HARNESSES["complex"]
# (already constructed with config.complex_model) as the vision harness
# rather than building a second one.
VISION_MODEL = config.complex_model

# TitleGenerator reuses the cheap simple-tier model (Haiku) since
# title summarization is a fixed-cost classification task that
# doesn't benefit from Sonnet. Same shared AsyncAnthropic client as
# the harnesses + router — no extra connection pool.
title_generator = TitleGenerator(client=claude, model=config.simple_model)

# Telemetry client: fire-and-forget POSTs to the API's internal
# endpoints. Disabled when api_url is empty (local dev without the
# API container running) — chat keeps working, telemetry just
# doesn't get written. Constructed before the TTSGenerator so its
# record_speak can be wired in as the TTS telemetry hook.
telemetry_client: TelemetryClient | None = (
    TelemetryClient(api_base_url=config.api_url) if config.api_url else None
)

# TTSGenerator owns the OpenAI client + the per-user daily char
# counter. Lives as a module-level singleton so the counter survives
# across requests; uvicorn workers are one process at our scale.
# The on_speak hook fires a fire-and-forget POST to
# /internal/telemetry/speak after each OpenAI call (success or
# failure) so the API can bill the characters; None when telemetry is
# disabled. See prog-strength-docs/sows/voice-chat.md and
# prog-strength-docs/sows/per-user-daily-usage-cap.md.
tts_generator = TTSGenerator(
    api_key=config.openai_api_key,
    model=config.openai_tts_model,
    default_voice=config.tts_voice_default,
    daily_char_cap=config.tts_daily_char_cap_per_user,
    instructions=config.tts_instructions,
    on_speak=telemetry_client.record_speak if telemetry_client else None,
)

# api_client: best-effort reader of chat_sessions.last_intent for the
# router hint. Empty api_url disables it (same condition that disables
# telemetry); the route handler treats None as "no hint" and the
# classifier falls back to conversation context alone.
api_client = APIClient(base_url=config.api_url) if config.api_url else None

# UsageGate: pre-call daily-allowance check over the API's
# GET /me/usage. Disabled when api_url is empty (no API to ask) OR
# when USAGE_GATE_ENABLED is false (first deploy lands telemetry
# writes without enforcing). A disabled gate's check_or_raise is a
# no-op. See prog-strength-docs/sows/per-user-daily-usage-cap.md.
usage_gate = UsageGate(
    api_base_url=config.api_url,
    enabled=config.usage_gate_enabled and bool(config.api_url),
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
        # Let browsers read the correlation id off cross-origin responses
        # so the frontend can log it / attach it to support reports.
        expose_headers=["X-Request-ID"],
    )

# Added LAST so it sits OUTERMOST in the middleware stack: the request id
# is minted before any handler (and before CORS short-circuits a
# preflight) and the X-Request-ID header is stamped on every response.
app.add_middleware(RequestIDMiddleware)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Mirror FastAPI's default HTTPException body ({"detail": ...}) but
    add the request id so a client (or its logs) can correlate a 4xx/5xx
    with the agent's logs. Preserves any headers the exception carries
    (e.g. the WWW-Authenticate on a 401)."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "request_id": current_request_id()},
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """422 body mirrors FastAPI's default ({"detail": [...]}) plus the
    request id, for the same correlation reason as above."""
    return JSONResponse(
        status_code=422,
        content={
            "detail": jsonable_encoder(exc.errors()),
            "request_id": current_request_id(),
        },
    )


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe. Mirrors the API and MCP `/health` envelope shape
    (now including request_id) so the same curl muscle memory works
    across all three services.
    """
    return {
        "service": SERVICE,
        "version": VERSION,
        "request_id": current_request_id(),
        "message": "service is healthy",
    }


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
    # Resolved-profile identity threaded into the system prompt, sourced
    # from the same GET /me the web client already holds for the sidebar.
    # display_name is the name the agent should call the user by;
    # height_cm is canonical cm (clients convert at the display edge).
    # Both default to None so older clients that don't send them keep
    # working — a missing name simply omits the identity line. See
    # prompt.build_chat_system_prompt for how they're rendered, and
    # prog-strength-docs/sows/user-profile-and-preferences.md.
    display_name: str | None = None
    height_cm: float | None = None
    # When true, the server runs the streaming-TTS pipeline alongside
    # the existing text streaming: text deltas pass through unchanged
    # AND new audio_chunk SSE events carry per-sentence mp3 audio for
    # the client to play in order. False (or missing) preserves
    # today's behavior — clients without voice mode see no behavior
    # change. See prog-strength-docs/sows/streaming-tts.md.
    voice_mode: bool = False


@app.post("/chat")
async def chat(req: ChatRequest, request: Request) -> Response:
    """SSE-stream a chat turn.

    Auth happens up-front so a 401 surfaces as a normal HTTP status
    before the stream starts. Routing happens lazily inside the
    generator so the model_chosen SSE event can be the first byte
    the client sees — that way the UI shows "via Haiku" without an
    extra HTTP request.
    """
    auth = authenticate(request, config.jwt_signing_key)

    # Pre-call usage gate. A capped user gets a 429 before any Claude
    # tokens are spent. Soft-allow + no-op-when-disabled live inside
    # check_or_raise, so this is a cheap call on the happy path.
    try:
        await usage_gate.check_or_raise(
            user_id=auth.user_id,
            token=auth.token,
            tz=req.client_timezone,
            surface="chat",
        )
    except CapExceeded as e:
        return Response(
            content=str(e),
            status_code=429,
            media_type="text/plain; charset=utf-8",
        )

    messages: list[dict[str, Any]] = [m.model_dump() for m in req.messages]

    telemetry = TurnInstrumentation.new(user_id=auth.user_id, session_id=req.session_id)

    # Build the per-request system prompt with the user's local date
    # prefixed. Done here (not inside the harness) so the prompt logic
    # stays out of the model-loop code path — harness takes the prompt
    # as-is and ships it to Anthropic. See
    # prompt.build_chat_system_prompt for the date-prefix rationale.
    system_prompt = build_chat_system_prompt(
        req.client_timezone,
        display_name=req.display_name,
        height_cm=req.height_cm,
    )

    inner = _route_and_stream(
        messages,
        auth.token,
        telemetry,
        system_prompt,
        client_timezone=req.client_timezone,
    )
    # When voice_mode is on, wrap the SSE stream with a voice_streamer
    # that buffers text deltas, detects sentence boundaries, fires
    # TTS for each sentence in parallel, and emits audio_chunk events
    # alongside the original text deltas. Off → no buffering, no TTS,
    # behavior is identical to the pre-streaming-tts /chat.
    stream = (
        voice_streamer(
            inner,
            user_id=auth.user_id,
            tts=tts_generator,
            session_id=telemetry.session_id,
        )
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
    # Chat session this TTS call belongs to, threaded into the
    # telemetry row so TTS spend can be joined to the conversation
    # that drove it. Nullable — /speak is sometimes called outside a
    # session context.
    session_id: str | None = None


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

    # Pre-call usage gate (first belt). The TTSGenerator._Quota char
    # cap is the second belt and still runs inside generate(). tz is
    # None here — /speak doesn't carry the client's timezone, so the
    # API falls back to UTC for the window.
    try:
        await usage_gate.check_or_raise(
            user_id=auth.user_id,
            token=auth.token,
            tz=None,
            surface="speak",
        )
    except CapExceeded as e:
        return Response(
            content=str(e),
            status_code=429,
            media_type="text/plain; charset=utf-8",
        )

    try:
        audio = await tts_generator.generate(
            user_id=auth.user_id,
            text=req.text,
            voice=req.voice,
            session_id=req.session_id,
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
async def voice_telemetry(req: VoiceTelemetryRequest, request: Request) -> dict[str, bool]:
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


def _has_image(messages: list[dict[str, Any]]) -> bool:
    """True iff any user message carries an image content block.

    Pure + side-effect-free (mirrors _maybe_inject_timezone) so it is
    unit-testable without driving the routing path. Only user turns
    count — an image in an assistant block is not a vision request.
    """
    for m in messages:
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image":
                    return True
    return False


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    """Return the text of the most recent user-authored message, as the
    query for memory retrieval. Content may be a plain string or a list
    of content blocks; for a list, the text of `type=="text"` blocks is
    joined (vision/image blocks are ignored). Returns "" when there is
    no user message or it carries no text.
    """
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = [
                block["text"]
                for block in content
                if isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ]
            return " ".join(texts)
        return ""
    return ""


async def _route_and_stream(
    messages: list[dict[str, Any]],
    user_token: str,
    telemetry: TurnInstrumentation,
    system_prompt: str,
    *,
    client_timezone: str | None = None,
) -> AsyncGenerator[bytes, None]:
    """Classify the request's tier+intent, dispatch to the matching
    harness with the intent in hand, then fire telemetry on the way
    out.

    Vision turns short-circuit before the classifier: an image content
    block forces the capable tier (it would silently degrade on Haiku)
    and skips both the prior-intent lookup and the router call.

    Both router and harness fail soft: a broken classifier collapses
    to (simple, general); a broken prefetch leaves the agent with its
    pre-SOW behavior. Either way /chat keeps streaming bytes.
    """
    try:
        if _has_image(messages):
            # Vision turn: skip the prior-intent lookup and the router
            # entirely, force the complex (vision-capable) harness, and
            # stamp tier+intent so record_prometheus_metrics still counts
            # the turn (it skips turns with an empty routed_tier). "general"
            # intent is a no-op prefetch, so the prompt is the base prompt
            # (plus the image paragraph) with no spurious prefetch.
            telemetry.had_image = True
            telemetry.routed_tier = "complex"
            telemetry.intent = "general"
            harness = HARNESSES["complex"]
            async for chunk in harness.stream_chat(
                messages,
                user_token,
                telemetry,
                system_prompt=system_prompt,
                intent="general",
                client_timezone=client_timezone,
            ):
                yield chunk
            return

        prior_intent: str | None = None
        if api_client is not None and telemetry.session_id:
            prior_intent = await api_client.get_session_intent(telemetry.session_id)

        # Run memory retrieval alongside the router so it never adds to
        # the turn's latency. Best-effort: any retrieval failure (or no
        # client / no user) yields an empty list, which renders to no
        # background block — the prompt is then byte-for-byte unchanged.
        decision_task = router_obj.route(
            messages,
            telemetry=telemetry,
            prior_intent=prior_intent,
        )
        if api_client is not None and telemetry.user_id:
            memory_task = api_client.retrieve_memories(
                telemetry.user_id, _last_user_text(messages)
            )
        else:

            async def _no_memories() -> list[str]:
                return []

            memory_task = _no_memories()
        decision, memories = await asyncio.gather(
            decision_task, memory_task, return_exceptions=True
        )
        if isinstance(memories, Exception) or not isinstance(memories, list):
            memories = []
        if isinstance(decision, Exception):
            raise decision  # preserve today's router-failure behaviour exactly

        harness = HARNESSES.get(decision.tier, HARNESSES["simple"])
        async for chunk in harness.stream_chat(
            messages,
            user_token,
            telemetry,
            system_prompt=system_prompt,
            intent=decision.intent,
            client_timezone=client_timezone,
            memories=memories,
        ):
            yield chunk
    finally:
        record_prometheus_metrics(telemetry)
        if telemetry_client is not None:
            telemetry_client.record_turn(telemetry)
