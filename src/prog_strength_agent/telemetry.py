"""Agent runtime telemetry.

Each /chat request creates a `TurnInstrumentation` instance that gets
threaded through the router and the harness; they populate it as the
turn runs. When the SSE stream finishes (success or error), the
server fires three fire-and-forget HTTP POSTs to the API's
/internal/telemetry/* endpoints — one per data shape (turn, tool
calls, messages).

Failure semantics: telemetry writes must never affect the user's
chat. Every HTTP call is wrapped in a broad except that logs and
moves on. If the API is down or the schema is wrong, telemetry is
lost; the user's response is not.

See prog-strength-docs/sows/monitoring-and-observability.md.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx
from prometheus_client import Counter, Histogram

log = logging.getLogger(__name__)

# Cap how much of a tool's response we keep in `result_summary`.
# Useful for debugging "what did the model see?" without blowing up
# the database; the 90-day TTL nulls these anyway.
_RESULT_SUMMARY_MAX_CHARS = 2000

# Prometheus counters. Materialized once per turn from
# TurnInstrumentation by record_prometheus_metrics(). The same
# per-turn data also gets persisted to telemetry.db via the
# TelemetryClient — Prometheus gives us live dashboards over rates
# and totals; SQLite gives us the full structured history for
# ad-hoc queries.
#
# Cardinality: bounded by (model × tier × direction) and (model ×
# tier × completion_reason). With ~3 models, 2 tiers, 4 directions,
# 4 completion_reasons, max series count is in the dozens. Safe.
AGENT_TOKENS_TOTAL = Counter(
    "agent_tokens_total",
    "Tokens consumed by the agent across chat turns.",
    ["direction", "model", "tier"],
)
AGENT_ROUTING_DECISIONS_TOTAL = Counter(
    "agent_routing_decisions_total",
    "Router classification count by tier.",
    ["tier"],
)
AGENT_TURNS_TOTAL = Counter(
    "agent_turns_total",
    "Total chat turns processed by the agent.",
    ["model", "tier", "completion_reason"],
)
# MCP tool invocations. Cardinality bounded by the (small) catalog
# of MCP tools (~5 today); outcome is "ok" or "error", so worst case
# is ~10 series. The histogram for latency is by tool_name only so
# percentile queries (p95 by tool) work without further label
# combinations.
AGENT_TOOL_CALLS_TOTAL = Counter(
    "agent_tool_calls_total",
    "MCP tool invocations made by the agent during chat turns.",
    ["tool_name", "outcome"],
)
AGENT_TOOL_CALL_DURATION_SECONDS = Histogram(
    "agent_tool_call_duration_seconds",
    "Duration of a single MCP tool invocation.",
    ["tool_name"],
    # Buckets cover the practical range: a few ms for in-process
    # work up to a couple of seconds for an MCP -> API roundtrip
    # that touches SQLite. Anything above ~10s is "something is
    # wrong" rather than "slow", so we don't extend further.
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
)

# Voice-mode time-to-first-audio. Reported by the client via
# POST /telemetry/voice — measured end-to-end from "user pressed
# send" to "first audio_chunk's mp3 starts playing." This is the
# load-bearing UX metric for the streaming-tts SOW; a regression
# beyond ~2s makes voice mode feel pointless. The Grafana dashboard
# renders this histogram with a horizontal 2s threshold line.
#
# Buckets are dense around the 1-3s target range, sparse past 5s
# where any sample is already a usability failure. user_id is the
# only label — keeps cardinality at "number of users" rather than
# blowing up with per-session labels.
AGENT_VOICE_TIME_TO_FIRST_AUDIO_SECONDS = Histogram(
    "agent_voice_time_to_first_audio_seconds",
    "End-to-end time-to-first-audio in voice mode, reported by the client.",
    ["user_id"],
    buckets=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 8.0, 15.0),
)

# Intent classification and prefetch metrics. Cardinality bounded by
# the (small, enumerated) set of known intents (~5 today). The prefetch
# histogram shares the same intent label so latency can be broken down
# per-intent in Grafana.
AGENT_INTENT_CLASSIFICATIONS_TOTAL = Counter(
    "agent_intent_classifications_total",
    "Intent classifications produced by the model router.",
    ["intent"],
)
AGENT_INTENT_PREFETCH_DURATION_SECONDS = Histogram(
    "agent_intent_prefetch_duration_seconds",
    "Time spent running an intent's prefetch tool calls (parallel asyncio.gather).",
    ["intent"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
)


def record_prometheus_metrics(t: "TurnInstrumentation") -> None:
    """Materialize per-turn data into Prometheus counters. Called
    once per turn from server.py after the SSE stream completes.

    Synchronous and in-process — unlike the TelemetryClient there's
    no network failure mode here, so this runs unconditionally
    (the HTTP telemetry write is fire-and-forget; this counter
    bump is not).
    """
    # Defensive: if the harness exited before populating its fields
    # (e.g. router itself threw), the labels would be empty strings
    # and pollute the metric stream. Skip those.
    if not t.routed_tier or not t.model:
        return

    AGENT_TURNS_TOTAL.labels(
        model=t.model,
        tier=t.routed_tier,
        completion_reason=t.completion_reason or "unknown",
    ).inc()
    AGENT_ROUTING_DECISIONS_TOTAL.labels(tier=t.routed_tier).inc()

    # Token totals are summed across all iterations of the tool-use
    # loop in the harness; one inc per direction here.
    for direction, count in (
        ("input", t.input_tokens),
        ("output", t.output_tokens),
        ("cache_creation", t.cache_creation_tokens),
        ("cache_read", t.cache_read_tokens),
    ):
        if count > 0:
            AGENT_TOKENS_TOTAL.labels(
                direction=direction, model=t.model, tier=t.routed_tier
            ).inc(count)

    # Tool calls — one Prometheus event per MCP invocation recorded
    # during the turn. The harness already timed each call and stamped
    # its outcome onto the ToolCallRecord; we just materialize those
    # records into Counter/Histogram bumps here.
    for call in t.tool_calls:
        outcome = "error" if call.error else "ok"
        AGENT_TOOL_CALLS_TOTAL.labels(
            tool_name=call.tool_name, outcome=outcome
        ).inc()
        # latency_ms is integer milliseconds; the histogram is in
        # seconds to match Prometheus convention.
        AGENT_TOOL_CALL_DURATION_SECONDS.labels(
            tool_name=call.tool_name
        ).observe(call.latency_ms / 1000.0)

    if t.intent:
        AGENT_INTENT_CLASSIFICATIONS_TOTAL.labels(intent=t.intent).inc()
        if t.intent_prefetch_duration_ms > 0:
            AGENT_INTENT_PREFETCH_DURATION_SECONDS.labels(intent=t.intent).observe(
                t.intent_prefetch_duration_ms / 1000.0,
            )


@dataclass
class ToolCallRecord:
    """One MCP tool invocation made during a turn."""

    tool_name: str
    arguments_json: str | None
    result_summary: str | None
    latency_ms: int
    error: str | None
    started_at: datetime
    ended_at: datetime


@dataclass
class MessageRecord:
    """One user or assistant message worth persisting. The SOW limits
    this to the human-facing pair (last user prompt + final assistant
    response) — system prompts and tool-result injections do not get
    rows.
    """

    role: str  # "user" | "assistant"
    content: str
    token_count: int | None = None


@dataclass
class TurnInstrumentation:
    """Collects every per-turn datum that lands in telemetry.db.

    Mutated by the router (router_model, router_latency_ms, routed_tier)
    and by the harness (model, tokens, latency, completion_reason,
    tool_calls, messages). The server reads the final state and fires
    three POSTs to the API.
    """

    turn_id: str
    user_id: str
    session_id: str

    # Router decision — populated by ModelRouter.route().
    router_model: str = ""
    router_latency_ms: int = 0
    routed_tier: str = ""

    # Intent classification — populated by ModelRouter.route().
    # Empty string means the router never produced a value (cold
    # exception in the classifier call); "general" is a deliberate
    # output meaning "no specific intent recognized."
    intent: str = ""

    # Intent prefetch instrumentation — populated by ModelHarness.
    intent_prefetch_duration_ms: int = 0
    intent_prefetch_failed: bool = False

    # Main turn — populated by ModelHarness.stream_chat().
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    time_to_first_token_ms: int = 0
    completion_reason: str = ""
    error: str | None = None

    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    messages: list[MessageRecord] = field(default_factory=list)

    started_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    ended_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @classmethod
    def new(cls, user_id: str, session_id: str | None) -> "TurnInstrumentation":
        """Construct a fresh instrumentation with a generated turn ID
        and a started_at pinned to now. The session ID falls back to a
        fresh UUID if the client didn't supply one, so no turn lands
        in telemetry without a session.
        """
        return cls(
            turn_id=str(uuid.uuid4()),
            user_id=user_id,
            session_id=session_id or str(uuid.uuid4()),
        )

    def finalize(self, *, completion_reason: str, error: str | None = None) -> None:
        """Stamp the end of the turn. Called by the harness after the
        last SSE event, before the server fires the telemetry POSTs.
        """
        self.completion_reason = completion_reason
        self.error = error
        self.ended_at = datetime.now(timezone.utc)

    @property
    def total_latency_ms(self) -> int:
        """Wall-clock duration of the turn in milliseconds."""
        return int((self.ended_at - self.started_at).total_seconds() * 1000)


def _build_turn_payload(t: "TurnInstrumentation") -> dict:
    """Marshal a TurnInstrumentation into the JSON shape the API's
    POST /internal/telemetry/turns endpoint expects. Pulled out of
    the client so tests can assert on the wire format without
    poking through httpx mocks.
    """
    return {
        "id": t.turn_id,
        "user_id": t.user_id,
        "session_id": t.session_id,
        "model": t.model,
        "routed_tier": t.routed_tier,
        "router_model": t.router_model,
        "router_latency_ms": t.router_latency_ms,
        "input_tokens": t.input_tokens,
        "output_tokens": t.output_tokens,
        "cache_creation_tokens": t.cache_creation_tokens,
        "cache_read_tokens": t.cache_read_tokens,
        "total_latency_ms": t.total_latency_ms,
        "time_to_first_token_ms": t.time_to_first_token_ms,
        "completion_reason": t.completion_reason,
        "error": t.error,
        "started_at": _iso(t.started_at),
        "ended_at": _iso(t.ended_at),
        "intent": t.intent,
        "intent_prefetch_duration_ms": t.intent_prefetch_duration_ms,
        "intent_prefetch_failed": t.intent_prefetch_failed,
    }


class TelemetryClient:
    """Fire-and-forget client for the API's /internal/telemetry/*
    endpoints. Reachable only from inside the Docker network — Caddy
    refuses to proxy /internal/* to the public internet, so there's
    no auth header to set.

    A single instance is shared by the FastAPI app. The underlying
    httpx.AsyncClient pools connections, so the per-turn cost is just
    a few microseconds of dispatch plus the network hop.
    """

    def __init__(self, api_base_url: str, *, timeout_seconds: float = 5.0):
        self._client = httpx.AsyncClient(
            base_url=api_base_url,
            timeout=timeout_seconds,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def record_turn(self, t: TurnInstrumentation) -> None:
        """Schedule the three telemetry POSTs for this turn on the
        event loop and return immediately. Each task catches its own
        exceptions so a single failure doesn't sink the others.
        """
        asyncio.create_task(self._send_turn(t))
        if t.tool_calls:
            asyncio.create_task(self._send_tool_calls(t))
        if t.messages:
            asyncio.create_task(self._send_messages(t))

    async def _send_turn(self, t: TurnInstrumentation) -> None:
        body = _build_turn_payload(t)
        await self._post("/internal/telemetry/turns", body)

    async def _send_tool_calls(self, t: TurnInstrumentation) -> None:
        body = {
            "calls": [
                {
                    "turn_id": t.turn_id,
                    "tool_name": c.tool_name,
                    "arguments_json": c.arguments_json,
                    "result_summary": c.result_summary,
                    "latency_ms": c.latency_ms,
                    "error": c.error,
                    "started_at": _iso(c.started_at),
                    "ended_at": _iso(c.ended_at),
                }
                for c in t.tool_calls
            ],
        }
        await self._post("/internal/telemetry/tool-calls", body)

    async def _send_messages(self, t: TurnInstrumentation) -> None:
        body = {
            "messages": [
                {
                    "turn_id": t.turn_id,
                    "role": m.role,
                    "content": m.content,
                    "token_count": m.token_count,
                }
                for m in t.messages
            ],
        }
        await self._post("/internal/telemetry/messages", body)

    async def _post(self, path: str, body: dict) -> None:
        """One-shot POST. Broad except is intentional — telemetry must
        never raise into the calling /chat task.
        """
        try:
            resp = await self._client.post(path, json=body)
            if resp.status_code >= 400:
                log.warning(
                    "telemetry: %s returned %d %s",
                    path,
                    resp.status_code,
                    resp.text[:200],
                )
        except Exception:
            # No retries — fire-and-forget. The chat already returned.
            log.exception("telemetry: %s failed", path)


def _iso(dt: datetime) -> str:
    """RFC3339 timestamp the Go API parses with time.Parse(time.RFC3339, …)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def now_ms() -> int:
    """Monotonic millisecond timestamp for measuring elapsed durations.
    Use perf_counter for elapsed math; use datetime.now for wall-clock
    stamps that get persisted.
    """
    return int(time.perf_counter() * 1000)


def truncate_result(result: str | None) -> str | None:
    """Cap the tool result for telemetry. Subject to the 90-day TTL on
    the API side; keeping it short here also keeps the request body
    small."""
    if result is None:
        return None
    if len(result) <= _RESULT_SUMMARY_MAX_CHARS:
        return result
    return result[:_RESULT_SUMMARY_MAX_CHARS] + "…[truncated]"
