"""Single-model harness around the agent's SSE tool-use loop.

Each `ModelHarness` instance pins a Claude model id and exposes
`stream_chat()` — the same loop that used to live inline in
server.py, just factored out so a tiered routing layer can pick
between multiple harnesses (Haiku for simple CRUD, Sonnet for
analysis, …) at request time.

Harnesses are stateless w.r.t. users: instantiate once at startup,
share across requests. The MCP session is opened *per call* inside
`stream_chat` so its lifetime is bound to the generator — the
caller doesn't have to thread an exit stack through.

The first SSE event emitted is always `model_chosen` so the frontend
can label the assistant turn before any text streams.
"""

import copy
import json
import logging
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack
from typing import Any

from anthropic import AsyncAnthropic
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from prog_strength_agent.prompt import SYSTEM_PROMPT

log = logging.getLogger(__name__)

# Cap the tool-use loop to keep a runaway model from hammering MCP in
# an infinite cycle. Eight iterations is enough for any realistic
# workflow (list_exercises → create_workout is two) with room for
# follow-ups.
MAX_TOOL_LOOP = 8


class ModelHarness:
    """One Claude model wired up to the MCP tool layer.

    Construction is cheap; instances are intended to live for the
    lifetime of the process. The Anthropic client is shared across
    harnesses since it's model-agnostic.
    """

    def __init__(
        self,
        client: AsyncAnthropic,
        model: str,
        mcp_url: str,
        max_tokens: int = 2048,
    ):
        self.client = client
        self.model = model
        self.mcp_url = mcp_url
        self.max_tokens = max_tokens

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        user_token: str,
    ) -> AsyncGenerator[bytes, None]:
        """Run the tool-use loop, yielding SSE-formatted bytes.

        Opens a fresh MCP `ClientSession` scoped to this call with the
        user's JWT in the Authorization header; MCP forwards that token
        to the API on each tool call. The session is torn down when the
        generator exits.

        Errors after the first byte is yielded become in-band
        `{"type":"error"}` events rather than HTTP errors — once the
        response is committed to 200, that's the only way to signal
        failure to the client.
        """
        # Emit the chosen model upfront so the client can label the
        # assistant turn while text/tool events arrive.
        yield _sse({"type": "model_chosen", "model": self.model})

        stack = AsyncExitStack()
        try:
            read, write, _ = await stack.enter_async_context(
                streamablehttp_client(
                    self.mcp_url,
                    headers={"Authorization": f"Bearer {user_token}"},
                )
            )
            session: ClientSession = await stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()

            # Re-fetch tools per request so adding/removing MCP tools
            # doesn't require a harness restart.
            mcp_tools = (await session.list_tools()).tools
            tools_for_claude = _build_tool_schemas(mcp_tools)

            for _iteration in range(MAX_TOOL_LOOP):
                assistant_blocks: list[dict[str, Any]] = []
                stop_reason: str | None = None

                async with self.client.messages.stream(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=SYSTEM_PROMPT,
                    tools=tools_for_claude,
                    messages=messages,
                ) as stream:
                    async for event in stream:
                        if event.type == "content_block_start":
                            block = event.content_block
                            if block.type == "tool_use":
                                yield _sse(
                                    {
                                        "type": "tool_use_start",
                                        "name": block.name,
                                    }
                                )
                        elif event.type == "content_block_delta":
                            delta = event.delta
                            if delta.type == "text_delta":
                                yield _sse(
                                    {"type": "text_delta", "text": delta.text}
                                )

                    final = await stream.get_final_message()
                    # Re-serialize to INPUT-shape — see _block_for_replay
                    # for the gory details on why we don't just dump.
                    assistant_blocks = [
                        _block_for_replay(b) for b in final.content
                    ]
                    stop_reason = final.stop_reason

                messages.append(
                    {"role": "assistant", "content": assistant_blocks}
                )

                if stop_reason != "tool_use":
                    yield _sse(
                        {"type": "done", "stop_reason": stop_reason or "unknown"}
                    )
                    return

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
            log.exception("chat stream failed (model=%s)", self.model)
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

    Deep-copy the schema so we don't accidentally mutate the SDK's
    cached objects on subsequent calls.
    """
    return [
        {
            "name": t.name,
            "description": t.description or "",
            "input_schema": copy.deepcopy(t.inputSchema) if t.inputSchema else {},
        }
        for t in mcp_tools
    ]
