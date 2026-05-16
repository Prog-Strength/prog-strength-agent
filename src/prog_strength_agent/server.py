"""FastAPI app: /health + /chat.

Lifespan owns the single persistent MCP ClientSession. /chat runs a
vanilla Anthropic tool-use loop: ask Claude with the (rewritten) tool
schemas, dispatch any tool_use blocks through MCP, feed results back,
repeat until Claude stops requesting tools.

Tool-schema rewrite is the security-critical bit: `user_id` is stripped
from every tool's input schema so Claude can't see or influence it, and
the agent injects the authenticated value (from the JWT's sub claim)
when forwarding the call to MCP.
"""

import copy
import logging
from contextlib import asynccontextmanager
from typing import Any

from anthropic import AsyncAnthropic
from fastapi import FastAPI, HTTPException, Request
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
async def chat(req: ChatRequest, request: Request) -> dict[str, Any]:
    user_id = authenticated_user_id(request, config.jwt_signing_key)

    # Re-fetch tools per request so that adding/removing MCP tools doesn't
    # require an agent restart. The MCP roundtrip is cheap (same host,
    # docker network) so the freshness is worth it.
    mcp_tools = (await mcp_client.list_tools()).tools
    tools_for_claude, tools_taking_user_id = _build_tool_schemas(mcp_tools)

    messages: list[dict[str, Any]] = [m.model_dump() for m in req.messages]

    for _iteration in range(MAX_TOOL_LOOP):
        resp = await claude.messages.create(
            model=config.claude_model,
            max_tokens=config.max_tokens,
            system=SYSTEM_PROMPT,
            tools=tools_for_claude,
            messages=messages,
        )
        # model_dump(mode="json") gives JSON-serializable dicts (e.g.
        # tool_use blocks become {"type": "tool_use", "id": "...", ...})
        # so we can both echo to the client and feed back to Claude.
        assistant_blocks = [b.model_dump(mode="json") for b in resp.content]
        messages.append({"role": "assistant", "content": assistant_blocks})

        if resp.stop_reason != "tool_use":
            return {
                "stop_reason": resp.stop_reason,
                "content": assistant_blocks,
            }

        # Execute every tool_use block in this turn before going back to
        # Claude. Multiple parallel tool_use blocks are common for read-only
        # lookups (e.g. list_exercises + list_workouts in one shot).
        tool_results: list[dict[str, Any]] = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            tool_input = dict(block.input or {})
            if block.name in tools_taking_user_id:
                # Authoritative injection: whatever Claude sent (or
                # didn't send) for user_id is replaced with the JWT's sub.
                tool_input["user_id"] = user_id
            result = await mcp_client.call_tool(block.name, tool_input)
            text = "\n".join(getattr(c, "text", str(c)) for c in result.content)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": text,
                    "is_error": bool(result.isError),
                }
            )
        messages.append({"role": "user", "content": tool_results})

    # Hit the loop cap — Claude is stuck calling tools. Surface to the
    # client rather than continue forever.
    raise HTTPException(
        status_code=500,
        detail=f"agent exceeded tool-use loop limit ({MAX_TOOL_LOOP})",
    )


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
