"""FastAPI app: /health + /chat (SSE streaming).

Lifespan owns the single persistent MCP ClientSession. /chat runs an
Anthropic tool-use loop using the SDK's streaming API and forwards
events to the client as Server-Sent Events:

    data: {"type": "text_delta", "text": "Hello"}

    data: {"type": "tool_use_start", "name": "list_exercises"}

    data: {"type": "tool_result", "name": "list_exercises", "is_error": false}

    data: {"type": "done", "stop_reason": "end_turn"}

The frontend renders text_delta events live, surfaces tool_use_start /
tool_result as ephemeral indicators, and stops the loading state on
`done`. Errors mid-stream emit `{"type": "error", "message": "..."}`.

Tool-schema rewrite remains the security-critical bit: `user_id` is
stripped from every tool's input schema so Claude can't see or influence
it, and the agent injects the authenticated value (from the JWT's sub
claim) when forwarding the call to MCP.
"""

import copy
import json
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from anthropic import AsyncAnthropic
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from prog_strength_agent.auth import authenticated_user_id
from prog_strength_agent.config import Config
from prog_strength_agent.mcp_client import MCPClient
from prog_strength_agent.prompt import SYSTEM_PROMPT
from prog_strength_agent.version import SERVICE, VERSION

log = logging.getLogger(__name__)

# Single shared state. uvicorn's worker process is long-lived; these get
# constructed once at import time and reused for the lifetime of the
# process. MCPClient.connect() and close() are driven by the FastAPI
# lifespan handler below.
config = Config.from_env()
mcp_client = MCPClient(config.mcp_url)
claude = AsyncAnthropic(api_key=config.anthropic_api_key)

# Cap the tool-use loop to keep a runaway Claude from hammering MCP in
# an infinite cycle. Eight iterations is enough for any realistic logging
# flow (list_exercises → create_workout is two) with room for follow-ups.
MAX_TOOL_LOOP = 8


@asynccontextmanager
async def lifespan(_: FastAPI):
    await mcp_client.connect()
    try:
        yield
    finally:
        await mcp_client.close()


app = FastAPI(title=SERVICE, version=VERSION, lifespan=lifespan)

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

    Auth + tool-schema setup happens synchronously so that a 401 or a
    misconfigured MCP shows up as a normal HTTP status before the
    stream starts. Once we begin emitting SSE the status is already
    200 — errors after that point flow through as `{"type":"error"}`
    events within the stream.
    """
    user_id = authenticated_user_id(request, config.jwt_signing_key)

    # Re-fetch tools per request so that adding/removing MCP tools doesn't
    # require an agent restart. The MCP roundtrip is cheap (same host,
    # docker network) so the freshness is worth it.
    mcp_tools = (await mcp_client.list_tools()).tools
    tools_for_claude, tools_taking_user_id = _build_tool_schemas(mcp_tools)

    messages: list[dict[str, Any]] = [m.model_dump() for m in req.messages]

    return StreamingResponse(
        _run_chat_stream(messages, tools_for_claude, tools_taking_user_id, user_id),
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
    tools_for_claude: list[dict[str, Any]],
    tools_taking_user_id: set[str],
    user_id: str,
) -> AsyncGenerator[bytes, None]:
    """Drive the tool-use loop, yielding SSE-formatted bytes.

    Pure generator: no FastAPI coupling, no Request access. Errors are
    converted to `{"type":"error"}` events and the stream is closed
    cleanly rather than raised — once the response status is 200, the
    only way to communicate failure is in-band.
    """
    try:
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
                        # When a tool_use block starts, emit a marker so
                        # the UI can show "Running list_exercises…". The
                        # tool's actual input arrives via input_json_delta
                        # events later; we don't need to surface those to
                        # the UI — the result is what matters.
                        if block.type == "tool_use":
                            yield _sse(
                                {"type": "tool_use_start", "name": block.name}
                            )
                    elif event.type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            yield _sse({"type": "text_delta", "text": delta.text})

                # get_final_message() blocks until the model finishes
                # streaming and returns the assembled message with all
                # content blocks materialized — text + tool_use, fully
                # populated input dicts and all.
                final = await stream.get_final_message()
                assistant_blocks = [b.model_dump(mode="json") for b in final.content]
                stop_reason = final.stop_reason

            messages.append({"role": "assistant", "content": assistant_blocks})

            if stop_reason != "tool_use":
                yield _sse({"type": "done", "stop_reason": stop_reason or "unknown"})
                return

            # Execute every tool_use block from this turn before going
            # back to Claude. Multiple parallel tool_use blocks are
            # common for read-only lookups (e.g. list_exercises +
            # list_workouts in one turn).
            tool_results: list[dict[str, Any]] = []
            for block_dict in assistant_blocks:
                if block_dict.get("type") != "tool_use":
                    continue
                name = block_dict["name"]
                tool_input = dict(block_dict.get("input") or {})
                if name in tools_taking_user_id:
                    # Authoritative injection: whatever Claude sent for
                    # user_id is replaced with the JWT's sub.
                    tool_input["user_id"] = user_id
                try:
                    result = await mcp_client.call_tool(name, tool_input)
                    text = "\n".join(
                        getattr(c, "text", str(c)) for c in result.content
                    )
                    is_error = bool(result.isError)
                except Exception as exc:
                    log.exception("mcp call_tool %s failed", name)
                    text = f"tool error: {exc}"
                    is_error = True

                yield _sse(
                    {"type": "tool_result", "name": name, "is_error": is_error}
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block_dict["id"],
                        "content": text,
                        "is_error": is_error,
                    }
                )
            messages.append({"role": "user", "content": tool_results})

        # Exceeded the loop cap — surface in-band, not as a 500, because
        # we've already sent the 200 status and the response body.
        yield _sse(
            {
                "type": "error",
                "message": f"exceeded tool-use loop limit ({MAX_TOOL_LOOP})",
            }
        )
    except Exception as exc:
        log.exception("chat stream failed")
        yield _sse({"type": "error", "message": f"agent error: {exc}"})


def _sse(payload: dict[str, Any]) -> bytes:
    """Format a payload as a single SSE event. The blank line at the
    end is the event separator per the SSE spec — clients buffer until
    they see it before dispatching.
    """
    return f"data: {json.dumps(payload)}\n\n".encode()


def _build_tool_schemas(
    mcp_tools: list[Any],
) -> tuple[list[dict[str, Any]], set[str]]:
    """Convert MCP tool definitions to Anthropic tool schemas.

    Strips `user_id` from each schema so Claude doesn't see it as a
    parameter — that field is server-side-authoritative, injected from
    the validated JWT. Returns the rewritten schemas plus the set of
    tool names that originally took a `user_id` so the dispatcher knows
    which calls need the injection.
    """
    tools: list[dict[str, Any]] = []
    takes_user_id: set[str] = set()
    for t in mcp_tools:
        schema = copy.deepcopy(t.inputSchema) if t.inputSchema else {}
        props = schema.get("properties") or {}
        if "user_id" in props:
            takes_user_id.add(t.name)
            del props["user_id"]
            schema["properties"] = props
            if "required" in schema:
                schema["required"] = [r for r in schema["required"] if r != "user_id"]
        tools.append(
            {
                "name": t.name,
                "description": t.description or "",
                "input_schema": schema,
            }
        )
    return tools, takes_user_id
