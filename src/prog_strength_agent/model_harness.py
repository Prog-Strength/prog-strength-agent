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
from collections.abc import AsyncGenerator, Iterable, Iterator
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from anthropic import AsyncAnthropic
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from prog_strength_agent.intents import IntentRegistry
from prog_strength_agent.memory import format_memory_block
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
    client_timezone: str | None = None,
    memories: list[str] | None = None,
) -> tuple[str, bool]:
    """Compose the per-turn system prompt and return it alongside the
    prefetch-failed flag. Pulled out of stream_chat so tests can pin
    the composition without mocking the entire Anthropic SDK + MCP
    session.

    Retrieved `memories` (if any) are rendered as a trailing background
    block. Empty/None memories render to "" so the composed prompt is
    byte-for-byte identical to a turn with memory disabled.
    """
    rules, data, failed = await IntentRegistry.run(intent, session, client_timezone)
    background = format_memory_block(memories or [])
    return (
        compose_system_prompt(
            base=base, rules=rules, data=data, background=background
        ),
        failed,
    )


# Cap the tool-use loop to keep a runaway model from hammering MCP in
# an infinite cycle. Eight iterations is enough for any realistic
# workflow (list_exercises → create_workout is two) with room for
# follow-ups.
MAX_TOOL_LOOP = 8

# Tools that accept a `timezone` arg for day-boundary math (nutrition
# day rollups and the training snapshot's local week). The harness
# auto-injects the user's client_timezone into these calls so the model
# never has to thread it through itself.
_TZ_AWARE_TOOLS = {"list_nutrition_log", "get_daily_macros", "get_training_snapshot"}


def _batch_item_count(name: str, block_input: Any) -> int | None:
    """Item count for a log_consumption_batch tool call, for the web chip.
    None for every other tool, or when the input isn't available yet (the
    streamed tool_use input can be empty at content_block_start)."""
    if name != "log_consumption_batch":
        return None
    if not isinstance(block_input, dict):
        return None
    items = block_input.get("items")
    return len(items) if isinstance(items, list) else None


def _maybe_inject_timezone(
    name: str,
    tool_input: dict[str, Any],
    client_timezone: str | None,
) -> dict[str, Any]:
    """Return tool_input with the user's timezone injected when the call
    is a nutrition tool, the model didn't already supply `timezone`, and
    a client_timezone is known.

    Pure + side-effect-free on the input dict (returns a new dict) so the
    call site can record the FINAL args in telemetry and the behavior is
    unit-testable without driving the full Anthropic stream loop. When
    client_timezone is None we deliberately inject nothing — the
    downstream API fast-fails with a 400 rather than silently guessing.
    """
    result = dict(tool_input)
    if name in _TZ_AWARE_TOOLS and "timezone" not in result and client_timezone:
        result["timezone"] = client_timezone
    return result


def _tool_start_event(name: str, tool_input: Any) -> dict[str, Any]:
    """The `tool_use_start` SSE payload the web app reads to show which
    tool the agent is calling. Shared by the model loop and the prefetch
    path so the two can't drift on the wire shape (e.g. the batch chip)."""
    event: dict[str, Any] = {"type": "tool_use_start", "name": name}
    count = _batch_item_count(name, tool_input)
    if count is not None:
        event["item_count"] = count
    return event


def _tool_result_event(name: str, text: str | None, is_error: bool) -> dict[str, Any]:
    """The `tool_result` SSE payload, surfacing the backend correlation
    id when the result carries one. Shared by the model loop and the
    prefetch path (see _tool_start_event)."""
    event: dict[str, Any] = {
        "type": "tool_result",
        "name": name,
        "is_error": is_error,
    }
    if text is not None and (request_id := _request_id_from_result(text)):
        event["request_id"] = request_id
    return event


@dataclass
class _PrefetchCall:
    """One tool call captured during intent prefetch, carrying everything
    needed to replay it onto the telemetry + SSE paths."""

    record: "ToolCallRecord"
    arguments: Any
    text: str | None
    is_error: bool


class _PrefetchToolRecorder:
    """Wraps the MCP session handed to an intent's prefetch so every
    `call_tool` it makes is captured.

    Intent prefetch (e.g. analyze_training's get_training_snapshot) runs
    before the model loop. Those calls would otherwise bypass the only
    two paths that report tool usage to the web app — telemetry.tool_calls
    and the SSE stream — and so be invisible in the UI. The harness drains
    `.calls` after prefetch and replays them via `_prefetch_tool_events`.

    Everything other than `call_tool` is delegated to the real session, so
    a prefetch that uses other session methods still works.
    """

    def __init__(self, session: Any):
        self._session = session
        self.calls: list[_PrefetchCall] = []

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes not defined on the wrapper itself
        # (call_tool, _session, calls are) — delegate the rest.
        return getattr(self._session, name)

    async def call_tool(self, name: str, arguments: Any = None, *args: Any, **kwargs: Any) -> Any:
        started_at = datetime.now(UTC)
        start_ms = now_ms()
        try:
            result = await self._session.call_tool(name, arguments, *args, **kwargs)
        except Exception as exc:
            # Record the failed attempt (so it's visible) then let the
            # prefetch's own error handling see the exception unchanged.
            self.calls.append(
                _PrefetchCall(
                    record=ToolCallRecord(
                        tool_name=name,
                        arguments_json=_safe_json(arguments),
                        result_summary=None,
                        latency_ms=now_ms() - start_ms,
                        error=str(exc),
                        started_at=started_at,
                        ended_at=datetime.now(UTC),
                    ),
                    arguments=arguments,
                    text=None,
                    is_error=True,
                )
            )
            raise
        text = "\n".join(getattr(c, "text", str(c)) for c in result.content)
        is_error = bool(getattr(result, "isError", False))
        self.calls.append(
            _PrefetchCall(
                record=ToolCallRecord(
                    tool_name=name,
                    arguments_json=_safe_json(arguments),
                    result_summary=truncate_result(text),
                    latency_ms=now_ms() - start_ms,
                    error=text if is_error else None,
                    started_at=started_at,
                    ended_at=datetime.now(UTC),
                ),
                arguments=arguments,
                text=text,
                is_error=is_error,
            )
        )
        return result


def _prefetch_tool_events(
    calls: Iterable[_PrefetchCall],
    telemetry: "TurnInstrumentation | None",
) -> Iterator[bytes]:
    """Replay captured prefetch tool calls onto both reporting paths:
    append each to telemetry.tool_calls and yield the same
    tool_use_start / tool_result SSE pair the model loop emits."""
    for call in calls:
        if telemetry is not None:
            telemetry.tool_calls.append(call.record)
        yield _sse(_tool_start_event(call.record.tool_name, call.arguments))
        yield _sse(_tool_result_event(call.record.tool_name, call.text, call.is_error))


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
        client_timezone: str | None = None,
        memories: list[str] | None = None,
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
            # Wrap the session so any tool the prefetch calls is captured —
            # those calls happen before the model loop and would otherwise
            # never reach the web app's tool-usage log.
            prefetch_recorder = _PrefetchToolRecorder(session)
            composed_system_prompt, prefetch_failed = await build_intent_aware_prompt(
                base=system_prompt or SYSTEM_PROMPT,
                intent=intent,
                session=prefetch_recorder,
                client_timezone=client_timezone,
                memories=memories,
            )
            if telemetry is not None:
                telemetry.intent_prefetch_duration_ms = now_ms() - prefetch_started
                telemetry.intent_prefetch_failed = prefetch_failed

            # Surface the prefetch's tool calls (telemetry + SSE) before any
            # model text, so the UI shows them just like model-issued calls.
            for chunk in _prefetch_tool_events(prefetch_recorder.calls, telemetry):
                yield chunk

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
                    # failure so the turn always completes. Wrapped in
                    # a cache_control block — see _system_blocks.
                    system=_system_blocks(composed_system_prompt),
                    tools=tools_for_claude,
                    messages=messages,
                ) as stream:
                    async for event in stream:
                        if event.type == "content_block_start":
                            block = event.content_block
                            if block.type == "tool_use":
                                yield _sse(
                                    _tool_start_event(
                                        block.name, getattr(block, "input", None)
                                    )
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
                    tool_started = datetime.now(UTC)
                    tool_start_ms = now_ms()
                    tool_error: str | None = None
                    # Auto-inject the user's timezone into nutrition tool
                    # calls so the model never has to pass it. tool_input
                    # is the FINAL args (post-injection) — used both for
                    # the MCP call and the telemetry record below.
                    tool_input = _maybe_inject_timezone(
                        block.name, dict(block.input or {}), client_timezone
                    )
                    try:
                        result = await session.call_tool(block.name, tool_input)
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
                                arguments_json=_safe_json(tool_input),
                                result_summary=truncate_result(text),
                                latency_ms=now_ms() - tool_start_ms,
                                error=tool_error,
                                started_at=tool_started,
                                ended_at=datetime.now(UTC),
                            )
                        )

                    # Surface the backend's correlation id when the tool
                    # result carries one (lookup_food_nutrition does) —
                    # the frontend can read it off the SSE stream in
                    # devtools and pivot straight into CloudWatch with
                    # `filter request_id = "…"`.
                    yield _sse(_tool_result_event(block.name, text, is_error))
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


def _request_id_from_result(text: str) -> str | None:
    """Pluck a `request_id` out of a JSON-object tool result, if any.

    MCP tools that forward to the Prog Strength API may include the
    API's X-Request-ID in their result payload for end-to-end tracing.
    Anything that isn't a JSON object with a string request_id — plain
    text results, lists, error strings — quietly yields None; tracing
    is a bonus, never a failure mode.
    """
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(payload, dict):
        request_id = payload.get("request_id")
        if isinstance(request_id, str) and request_id:
            return request_id
    return None


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


def _system_blocks(system_prompt: str) -> list[dict[str, Any]]:
    """Wrap the system prompt in a content block carrying a prompt-cache
    breakpoint.

    This is the second of the harness's two cache breakpoints (the
    first sits on the tools array — see _build_tool_schemas). Anthropic
    caches the prefix up to each breakpoint, so iterations 2+ of the
    tool-use loop and follow-up turns within the cache TTL re-read
    tools + system at ~10% of the normal input price instead of
    re-paying full freight. The system prompt varies more than the
    tools (date prefix daily, intent data per turn), which is exactly
    why it gets its own breakpoint: a system miss still leaves the
    tools-array hit intact.
    """
    return [
        {
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def _build_tool_schemas(mcp_tools: list[Any]) -> list[dict[str, Any]]:
    """Convert MCP tool definitions to Anthropic tool schemas.

    Deep-copy the schema so we don't accidentally mutate the SDK's
    cached objects on subsequent calls.

    The LAST tool carries a prompt-cache breakpoint, which caches the
    entire tools array — the largest fully-stable block in every
    request (17 schemas, identical across all calls until the MCP
    server's tool set changes, which naturally busts the cache). The
    deliberately tiny router/title prompts are NOT cached: they sit
    below Anthropic's minimum cacheable prefix length, so marking them
    would only add cache-write premium for no reads.
    """
    tools: list[dict[str, Any]] = [
        {
            "name": t.name,
            "description": t.description or "",
            "input_schema": copy.deepcopy(t.inputSchema) if t.inputSchema else {},
        }
        for t in mcp_tools
    ]
    if tools:
        tools[-1]["cache_control"] = {"type": "ephemeral"}
    return tools
