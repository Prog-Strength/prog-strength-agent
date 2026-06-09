"""Per-user daily usage gate.

Before every Claude or TTS call the agent pre-checks the user's daily
allowance via the API's `GET /me/usage` endpoint. Results are cached
in-process for a short TTL (30s by default) per user — the cap is
daily, so a few seconds of staleness is harmless, and the vast
majority of chat turns never touch the API for the check.

Failure posture is deliberately permissive: any API failure
(timeout, 5xx, network, malformed JSON) soft-allows. Telemetry
post-writes record actual spend every turn, so the next successful
check converges regardless of whether the gate enforced. A flaky API
must never block chat for a user who isn't actually overspending.

`USAGE_GATE_ENABLED=false` short-circuits `check_or_raise` to a no-op
so the first deploy can land the TTS telemetry writes without
changing user-facing behavior. The existing `TTSGenerator._Quota`
in-process char cap stays as a second belt.

See prog-strength-docs/sows/per-user-daily-usage-cap.md.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import httpx
from prometheus_client import Counter, Histogram

log = logging.getLogger(__name__)

# 429 body shown to the user when over the daily allowance. Shared
# with the web client's capped-state copy; the web computes the
# precise "resets in Xh Ym" countdown from `resets_at` itself.
CAP_EXCEEDED_MESSAGE = "You've used your daily AI allowance. New allowance available soon."

# Prometheus metrics. All cardinalities bounded — never label by
# user_id (unbounded). `surface` is the small enum {chat, speak};
# `outcome` is {hit, miss, allowed, blocked, api_error}.
AGENT_USAGE_GATE_BLOCKS_TOTAL = Counter(
    "agent_usage_gate_blocks_total",
    "Times the usage gate raised CapExceeded (user over daily allowance).",
    ["surface"],
)
AGENT_USAGE_GATE_CHECK_DURATION_SECONDS = Histogram(
    "agent_usage_gate_check_duration_seconds",
    "Wall-clock duration of a single UsageGate.check_or_raise call.",
    # Dense around cache hits (sub-ms) and the API roundtrip band
    # (50-500ms); anything past ~1s means the timeout misbehaved.
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
AGENT_USAGE_GATE_CHECK_OUTCOME_TOTAL = Counter(
    "agent_usage_gate_check_outcome_total",
    "UsageGate check outcomes — one bump per check_or_raise call.",
    ["outcome"],
)


class CapExceeded(Exception):
    """User has hit their daily allowance. Maps to HTTP 429 at the
    handler boundary.
    """


@dataclass
class UsageSnapshot:
    """One cached `GET /me/usage` result for a single user.

    `fetched_at_monotonic` is a `time.monotonic()` stamp used purely
    for TTL math — monotonic so it's immune to wall-clock jumps.
    """

    percent_used: int
    capped: bool
    fetched_at_monotonic: float


class UsageGate:
    """In-process pre-call gate over the API's `GET /me/usage`.

    One instance is shared by the FastAPI app. The underlying
    httpx.AsyncClient pools connections; the per-user cache + per-user
    lock keep concurrent turns from stampeding the API after the TTL
    expires.
    """

    def __init__(
        self,
        api_base_url: str,
        *,
        enabled: bool = True,
        cache_ttl_seconds: float = 30.0,
        timeout_seconds: float = 0.5,
    ):
        self._enabled = enabled
        self._ttl = cache_ttl_seconds
        # Disabled gate never opens a connection. We still hold a
        # client object only when enabled; an empty base_url with
        # enabled=True is the caller's responsibility (server.py
        # constructs disabled in that case).
        self._client = httpx.AsyncClient(base_url=api_base_url, timeout=timeout_seconds)
        self._cache: dict[str, UsageSnapshot] = {}
        # Per-user locks prevent a stampede: when several concurrent
        # calls find an expired entry, only the first fetches; the
        # rest wait on the lock and read the freshly stored snapshot.
        # The locks dict itself is guarded by _locks_parent so two
        # tasks racing to create the same user's lock don't make two.
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_parent = asyncio.Lock()

    async def aclose(self) -> None:
        await self._client.aclose()

    def invalidate(self, user_id: str) -> None:
        """Drop the cached snapshot for one user so the next call
        re-fetches. Called right after a CapExceeded raise so a stale
        entry can't false-allow the next call.
        """
        self._cache.pop(user_id, None)

    async def _user_lock(self, user_id: str) -> asyncio.Lock:
        async with self._locks_parent:
            lock = self._locks.get(user_id)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[user_id] = lock
            return lock

    async def check_or_raise(
        self,
        *,
        user_id: str,
        token: str,
        tz: str | None,
        surface: str = "chat",
    ) -> None:
        """Pre-call gate. Raises CapExceeded when the cached or freshly
        fetched `/me/usage` reports `capped=true`. Soft-allows (returns
        None) on any API failure. No-op when the gate is disabled.

        `token` is the caller's raw bearer JWT — `/me/usage` is JWT-gated
        and identifies the user solely from the bearer token, so it MUST
        be forwarded on the fetch or every call 401s (soft-allow) and the
        cap is never enforced. The cache key stays `user_id`: the token
        for a given user is equivalent across calls.
        """
        if not self._enabled:
            return None

        start = time.monotonic()
        try:
            return await self._check(user_id=user_id, token=token, tz=tz, surface=surface)
        finally:
            AGENT_USAGE_GATE_CHECK_DURATION_SECONDS.observe(time.monotonic() - start)

    async def _check(self, *, user_id: str, token: str, tz: str | None, surface: str) -> None:
        # Fast path: a fresh-enough cached snapshot needs no lock and
        # no HTTP. We re-validate freshness under the lock below for
        # the miss path so a stampede collapses to one fetch.
        cached = self._cache.get(user_id)
        if cached is not None and (time.monotonic() - cached.fetched_at_monotonic) < self._ttl:
            return self._decide(cached, user_id=user_id, surface=surface, outcome="hit")

        lock = await self._user_lock(user_id)
        async with lock:
            # Re-check under the lock: a sibling task may have just
            # populated the cache while we waited for the lock.
            cached = self._cache.get(user_id)
            if cached is not None and (time.monotonic() - cached.fetched_at_monotonic) < self._ttl:
                return self._decide(cached, user_id=user_id, surface=surface, outcome="hit")

            snapshot = await self._fetch(user_id=user_id, token=token, tz=tz)
            if snapshot is None:
                # Soft-allow on any API failure.
                AGENT_USAGE_GATE_CHECK_OUTCOME_TOTAL.labels(outcome="api_error").inc()
                return None
            self._cache[user_id] = snapshot
            return self._decide(snapshot, user_id=user_id, surface=surface, outcome="miss")

    def _decide(self, snapshot: UsageSnapshot, *, user_id: str, surface: str, outcome: str) -> None:
        """Record the cache outcome (hit/miss) then enforce the
        decision (allowed/blocked). Raises CapExceeded when capped.
        """
        AGENT_USAGE_GATE_CHECK_OUTCOME_TOTAL.labels(outcome=outcome).inc()
        if snapshot.capped:
            AGENT_USAGE_GATE_CHECK_OUTCOME_TOTAL.labels(outcome="blocked").inc()
            AGENT_USAGE_GATE_BLOCKS_TOTAL.labels(surface=surface).inc()
            # Drop the cached entry so the next call re-fetches rather
            # than false-allowing from a stale snapshot once usage resets.
            self.invalidate(user_id)
            raise CapExceeded(CAP_EXCEEDED_MESSAGE)
        AGENT_USAGE_GATE_CHECK_OUTCOME_TOTAL.labels(outcome="allowed").inc()
        return None

    async def _fetch(self, *, user_id: str, token: str, tz: str | None) -> UsageSnapshot | None:
        """GET /me/usage for one user. Returns a snapshot on success,
        or None on any failure (timeout, 5xx, network, bad JSON).
        Never raises.

        The endpoint is JWT-gated and identifies the user solely from
        the bearer token (same middleware as /me), so we forward the
        caller's raw JWT verbatim — the same pattern the harness uses to
        forward the user's token to MCP. The handler passes tz through
        from the originating request.
        """
        params: dict[str, str] = {}
        if tz:
            params["tz"] = tz
        headers = {"Authorization": f"Bearer {token}"}
        try:
            resp = await self._client.get("/me/usage", params=params, headers=headers)
            if resp.status_code >= 400:
                log.warning(
                    "usage_gate_api_error: /me/usage returned %d for user=%s",
                    resp.status_code,
                    user_id,
                )
                return None
            payload = resp.json()
        except Exception:
            log.warning("usage_gate_api_error: /me/usage failed for user=%s", user_id)
            return None

        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            log.warning("usage_gate_api_error: /me/usage missing data for user=%s", user_id)
            return None
        percent_used = data.get("percent_used")
        capped = data.get("capped")
        if not isinstance(percent_used, int) or not isinstance(capped, bool):
            log.warning("usage_gate_api_error: /me/usage bad shape for user=%s", user_id)
            return None
        return UsageSnapshot(
            percent_used=percent_used,
            capped=capped,
            fetched_at_monotonic=time.monotonic(),
        )
