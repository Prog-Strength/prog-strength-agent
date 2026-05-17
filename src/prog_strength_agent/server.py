"""FastAPI app: /health + /chat (SSE streaming).

Auth model: each /chat request opens its own MCP session whose HTTP
Authorization header carries the end-user's Bearer JWT. MCP's tool
handlers read the header and forward it to the API — MCP holds no
signing key and cannot impersonate users.

Trade-off: handshake cost is paid per /chat turn (typically <300ms,
dwarfed by Claude latency). In exchange the agent has no
long-lived auth state and MCP becomes a transparent proxy.

Tool-schema rewrite removed (it existed to strip `user_id` from the
schema Claude saw; MCP tools no longer take a `user_id` arg, so
nothing to strip).
"""

import copy
import json
import logging
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack
from typing import Any

from anthropic import AsyncAnthropic
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from pydantic import BaseModel

from prog_strength_agent.auth import authenticate
from prog_strength_agent.config import Config
from prog_strength_agent.prompt import SYSTEM_PROMPT
from prog_strength_agent.version import SERVICE, VERSION

log = logging.getLogger(__name__)

# Single shared state. uvicorn's worker process is long-lived; these get
# constructed once at import time and reused for the lifetime of the
# process.
config = Config.from_env()
claude = AsyncAnthropic(api_key=config.anthropic_api_key)

# Cap the tool-use loop to keep a runaway Claude from hammering MCP in
# an infinite cycle. Eight iterations is enough for any realistic logging
# flow (list_exercises → create_workout is two) with room for follow-ups.
MAX_TOOL_LOOP = 8


app = FastAPI(title=SERVICE, version=VERSION)

# CORS for the frontend on Vercel (or wherever). The /chat endpoint is
# called cross-origin from the browser, so without these headers the
# request never leaves. Bearer auth is in the Authorization header, not
# cookies, so we do NOT enable allow_credentials.
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
    before the stream begins. Once we yield the first byte the status
    is locked to 200 — errors after that point flow through as
    `{"type":"error"}` events inside the stream.

    The MCP session is opened lazily inside the generator (not here)
    because passing it across the StreamingResponse boundary requires
    careful lifetime management; an AsyncExitStack inside the
    generator keeps the open/close logic local and trivially correct.
    """
    auth = authenticate(request, config.jwt_signing_key)
    messages: list[dict[str, Any]] = [m.model_dump() for m in req.messages]

    return StreamingResponse(
        _run_chat_stream(messages, auth.token),
        media_type="text/event-stream",
        headers={
            # Prevent intermediaries (Caddy, browsers, proxies) from
            # buffering — we want bytes flushed as they're produced so
            # the UI sees tokens arrive live.
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def _run_chat_stream(
    messages: list[dict[str, Any]],
    user_token: str,
) -> AsyncGenerator[bytes, None]:
    """Drive the tool-use loop, yielding SSE-formatted bytes.

    Opens a fresh MCP ClientSession scoped to this single /chat request,
    with the user's JWT in the Authorization header. MCP's tool handlers
    read that header via FastMCP's request context and forward it
    verbatim to the API. The session closes when this generator exits.
    """
    stack = AsyncExitStack()
    try:
        # Per-request MCP session. The streamable-HTTP transport carries
        # the Authorization header through the JSON-RPC handshake; FastMCP
        # exposes it to tool handlers via get_http_headers.
        read, write, _ = await stack.enter_async_context(
            streamablehttp_client(
                config.mcp_url,
                headers={"Authorization": f"Bearer {user_token}"},
            )
        )
        session: ClientSession = await stack.enter_async_context(
            ClientSession(read, write)
        )
        await session.initialize()

        mcp_tools = (await session.list_tools()).tools
        tools_for_claude = _build_tool_schemas(mcp_tools)

        for _iteration in range(MAX_TOOL_LOOP):
            assistant_blocks: list[dict[str, Any]] = []
            stop_reason: str | None = None

            async with claude.messages.stream(
                model=config.claude_model,
                max_tokens=config.max_tokens,
                system=SYSTEM_PROMPT,
                tools=tools_for_claude,
                messages=messages,
            ) as stream:
                async for event in stream:
                    if event.type == "content_block_start":
                        block = event.content_block
                        if block.type == "tool_use":
                            yield _sse(
                                {"type": "tool_use_start", "name": block.name}
                            )
                    elif event.type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            yield _sse({"type": "text_delta", "text": delta.text})

                final = await stream.get_final_message()
                # Re-serialize with only INPUT-shape fields. The SDK's
                # response objects carry output-only fields (e.g.
                # `parsed_output` on text blocks) that Anthropic rejects
                # when echoed back unchanged. See _block_for_replay.
                assistant_blocks = [_block_for_replay(b) for b in final.content]
                stop_reason = final.stop_reason

            messages.append({"role": "assistant", "content": assistant_blocks})

            if stop_reason != "tool_use":
                yield _sse({"type": "done", "stop_reason": stop_reason or "unknown"})
                return

            # Execute every tool_use block from this turn before going
            # back to Claude. The MCP session was opened with the user's
            # JWT, so MCP forwards that token to the API on each call;
            # no user_id injection needed here anymore.
            tool_results: list[dict[str, Any]] = []
            for block in final.content:
                if block.type != "tool_use":
                    continue
                try:
                    result = await session.call_tool(
                        block.name, dict(block.input or {})
                    )
                    text = "\n".join(
                        getattr(c, "text", str(c)) for c in result.content
                    )
                    is_error = bool(result.isError)
                except Exception as exc:
                    log.exception("mcp call_tool %s failed", block.name)
                    text = f"tool error: {exc}"
                    is_error = True

                yield _sse(
                    {
                        "type": "tool_result",
                        "name": block.name,
                        "is_error": is_error,
                    }
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": text,
                        "is_error": is_error,
                    }
                )
            messages.append({"role": "user", "content": tool_results})

        yield _sse(
            {
                "type": "error",
                "message": f"exceeded tool-use loop limit ({MAX_TOOL_LOOP})",
            }
        )
    except Exception as exc:
        log.exception("chat stream failed")
        yield _sse({"type": "error", "message": f"agent error: {exc}"})
    finally:
        await stack.aclose()


def _sse(payload: dict[str, Any]) -> bytes:
    """Format a payload as a single SSE event. The blank line at the
    end is the event separator per the SSE spec — clients buffer until
    they see it before dispatching.
    """
    return f"data: {json.dumps(payload)}\n\n".encode()


def _block_for_replay(block: Any) -> dict[str, Any]:
    """Serialize a single Anthropic response content block to the
    input-shape dict that messages.create accepts on a subsequent turn.

    The SDK's response models include output-only fields (e.g. text
    blocks carry `parsed_output` and `citations`) that the Anthropic
    API rejects on input with "Extra inputs are not permitted". This
    whitelist sidesteps that.
    """
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "tool_use":
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
    return block.model_dump(mode="json")


def _build_tool_schemas(mcp_tools: list[Any]) -> list[dict[str, Any]]:
    """Convert MCP tool definitions to Anthropic tool schemas.

    Previously this also stripped `user_id` from each schema so Claude
    couldn't spoof another user. Since MCP now derives identity from
    the inbound Authorization header (no user_id in tool args), there's
    nothing to strip — but we still deep-copy the schema so we don't
    accidentally mutate the SDK's cached objects.
    """
    return [
        {
            "name": t.name,
            "description": t.description or "",
            "input_schema": copy.deepcopy(t.inputSchema) if t.inputSchema else {},
        }
        for t in mcp_tools
    ]
