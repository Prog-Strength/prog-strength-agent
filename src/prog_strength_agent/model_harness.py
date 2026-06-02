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
from datetime import datetime, timezone
from typing import Any

from anthropic import AsyncAnthropic
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from prog_strength_agent.intents import IntentRegistry
from prog_strength_agent.prompt import SYSTEM_PROMPT, compose_system_prompt
from prog_strength_agent.telemetry import (
    MessageRecord,
    ToolCallRecord,
    TurnInstrumentation,
    now_ms,
    truncate_result,
)

log = logging.getLogger(__name__)


async def build_intent_aware_prompt(
    *,
    base: str,
    intent: str,
    session: Any,
) -> str:
    """Compose the per-turn system prompt: base + intent rules + intent
    data. Pulled out of stream_chat so tests can pin the composition
    without mocking the entire Anthropic SDK + MCP session.
    """
    rules, data = await IntentRegistry.run(intent, session)
    return compose_system_prompt(base=base, rules=rules, data=data)


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
        telemetry: TurnInstrumentation | None = None,
        system_prompt: str | None = None,
        intent: str = "general",
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

        When a `telemetry` instrumentation is passed in, populates it
        as the turn unfolds (model, tokens, latency, tool calls, final
        assistant message). The server fires the telemetry POSTs after
        the generator exits so this method never blocks on them.
        """
        if telemetry is not None:
            telemetry.model = self.model
            # Capture the inbound user message right at the start — if
            # the harness crashes mid-turn we still have a row for what
            # the user said.
            _capture_user_message(messages, telemetry)

        first_token_seen = False
        turn_start_ms = now_ms()
        completion_reason = "error"
        completion_error: str | None = None

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

            prefetch_started = now_ms()
            try:
                composed_system_prompt = await build_intent_aware_prompt(
                    base=system_prompt or SYSTEM_PROMPT,
                    intent=intent,
                    session=session,
                )
                prefetch_failed = False
            except Exception:
                log.exception("intent prefetch composition failed")
                composed_system_prompt = system_prompt or SYSTEM_PROMPT
                prefetch_failed = True
            if telemetry is not None:
                telemetry.intent_prefetch_duration_ms = now_ms() - prefetch_started
                telemetry.intent_prefetch_failed = prefetch_failed

            # Accumulator for the assistant's user-facing text across
            # all iterations of the tool-use loop. Saved to telemetry
            # at the end as the assistant message.
            assistant_text_parts: list[str] = []

            for _iteration in range(MAX_TOOL_LOOP):
                assistant_blocks: list[dict[str, Any]] = []
                stop_reason: str | None = None

                async with self.client.messages.stream(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    # composed_system_prompt is assembled above from
                    # the base prompt + intent-specific rules and
                    # prefetched data. Falls back to base on prefetch
                    # failure so the turn always completes.
                    system=composed_system_prompt,
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
                                if telemetry is not None and not first_token_seen:
                                    telemetry.time_to_first_token_ms = (
                                        now_ms() - turn_start_ms
                                    )
                                    first_token_seen = True
                                assistant_text_parts.append(delta.text)
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
                    if telemetry is not None:
                        _accumulate_usage(telemetry, final)

                messages.append(
                    {"role": "assistant", "content": assistant_blocks}
                )

                if stop_reason != "tool_use":
                    yield _sse(
                        {"type": "done", "stop_reason": stop_reason or "unknown"}
                    )
                    completion_reason = stop_reason or "unknown"
                    return

                tool_results: list[dict[str, Any]] = []
                for block in final.content:
                    if block.type != "tool_use":
                        continue
                    tool_started = datetime.now(timezone.utc)
                    tool_start_ms = now_ms()
                    tool_error: str | None = None
                    try:
                        result = await session.call_tool(
                            block.name, dict(block.input or {})
                        )
                        text = "\n".join(
                            getattr(c, "text", str(c)) for c in result.content
                        )
                        is_error = bool(result.isError)
                        if is_error:
                            tool_error = text
                    except Exception as exc:
                        log.exception("mcp call_tool %s failed", block.name)
                        text = f"tool error: {exc}"
                        is_error = True
                        tool_error = str(exc)

                    if telemetry is not None:
                        telemetry.tool_calls.append(
                            ToolCallRecord(
                                tool_name=block.name,
                                arguments_json=_safe_json(block.input),
                                result_summary=truncate_result(text),
                                latency_ms=now_ms() - tool_start_ms,
                                error=tool_error,
                                started_at=tool_started,
                                ended_at=datetime.now(timezone.utc),
                            )
                        )

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
            completion_reason = "error"
            completion_error = f"exceeded tool-use loop limit ({MAX_TOOL_LOOP})"
        except Exception as exc:
            log.exception("chat stream failed (model=%s)", self.model)
            yield _sse({"type": "error", "message": f"agent error: {exc}"})
            completion_reason = "error"
            completion_error = str(exc)
        finally:
            await stack.aclose()
            if telemetry is not None:
                if assistant_text_parts:
                    telemetry.messages.append(
                        MessageRecord(
                            role="assistant",
                            content="".join(assistant_text_parts),
                        )
                    )
                telemetry.finalize(
                    completion_reason=completion_reason,
                    error=completion_error,
                )


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


def _safe_json(value: Any) -> str | None:
    """Best-effort JSON serialization of a tool's input args. Falls
    back to repr() when something isn't JSON-serializable so telemetry
    still records a row instead of swallowing the call.
    """
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return repr(value)


def _capture_user_message(
    messages: list[dict[str, Any]],
    telemetry: TurnInstrumentation,
) -> None:
    """Pull the most recent user-authored text out of the conversation
    and append it as a MessageRecord. The "user" role for tool_result
    injections is *also* "user" in Anthropic's schema; we filter those
    out by requiring string content (humans send strings; tool results
    are list-of-blocks).
    """
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content:
            telemetry.messages.append(MessageRecord(role="user", content=content))
            return


def _accumulate_usage(telemetry: TurnInstrumentation, final: Any) -> None:
    """Add this iteration's token usage to the running totals on the
    instrumentation. Multi-iteration tool-use loops sum across all the
    model calls within one user-facing turn.
    """
    usage = getattr(final, "usage", None)
    if usage is None:
        return
    telemetry.input_tokens += getattr(usage, "input_tokens", 0) or 0
    telemetry.output_tokens += getattr(usage, "output_tokens", 0) or 0
    telemetry.cache_creation_tokens += (
        getattr(usage, "cache_creation_input_tokens", 0) or 0
    )
    telemetry.cache_read_tokens += (
        getattr(usage, "cache_read_input_tokens", 0) or 0
    )


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
