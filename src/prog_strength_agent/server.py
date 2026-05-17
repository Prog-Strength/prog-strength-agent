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
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from prog_strength_agent.auth import authenticate
from prog_strength_agent.config import Config
from prog_strength_agent.model_harness import ModelHarness
from prog_strength_agent.model_router import ModelRouter
from prog_strength_agent.version import SERVICE, VERSION

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


app = FastAPI(title=SERVICE, version=VERSION)

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

    return StreamingResponse(
        _route_and_stream(messages, auth.token),
        media_type="text/event-stream",
        headers={
            # Prevent intermediaries (Caddy, browsers, proxies) from
            # buffering — we want bytes flushed as they're produced so
            # the UI sees tokens arrive live.
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def _route_and_stream(
    messages: list[dict[str, Any]],
    user_token: str,
) -> AsyncGenerator[bytes, None]:
    """Classify the request's tier, dispatch to the matching harness.

    The router's failure mode is "default to simple" (the cheaper
    tier), so even if the classifier call breaks, /chat keeps working
    — the user may just get a Haiku-level answer to a question that
    would have benefitted from Sonnet.
    """
    tier = await router_obj.route(messages)
    harness = HARNESSES.get(tier, HARNESSES["simple"])
    async for chunk in harness.stream_chat(messages, user_token):
        yield chunk
