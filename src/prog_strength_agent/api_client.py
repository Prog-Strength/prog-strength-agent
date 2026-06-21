"""Thin HTTP client for the Prog Strength API's internal endpoints.

Today's only consumer is the agent's pre-router step that reads the
session's last classified intent as a hint. Any failure (timeout,
5xx, network error) is swallowed and surfaced as `None` so a sluggish
API never adds user-visible latency on top of the classifier.

Network boundary: lives entirely under /internal/* on the API, which
Caddy refuses to proxy. No auth header to set.
"""

from __future__ import annotations

import logging

import httpx

log = logging.getLogger(__name__)


class APIClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 0.2,
        memory_timeout_seconds: float = 0.4,
    ):
        # 200ms timeout — fast enough that a single sluggish call
        # doesn't add user-visible latency on top of the ~500ms
        # classifier. Empty base_url disables the client; useful for
        # local dev when the API container isn't running.
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout_seconds)
        # Memory retrieval gets a slightly larger budget (400ms) than the
        # intent lookup because it covers an embedding round-trip, but it
        # stays under the router's ~500ms so it never extends the turn.
        self._memory_timeout = memory_timeout_seconds

    async def aclose(self) -> None:
        await self._client.aclose()

    async def get_session_intent(self, session_id: str) -> str | None:
        """Return the session's last classified intent, or None on
        missing-session / any failure. Never raises.
        """
        if not session_id:
            return None
        try:
            resp = await self._client.get(
                f"/internal/chat-sessions/{session_id}/intent",
            )
            if resp.status_code >= 400:
                return None
            payload = resp.json()
        except Exception:
            log.exception("api_client: get_session_intent failed")
            return None
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            return None
        intent = data.get("intent")
        if isinstance(intent, str) and intent:
            return intent
        return None

    async def retrieve_memories(self, user_id: str, query: str) -> list[str]:
        """Return durable memories about the user relevant to `query`, or
        an empty list on missing input / any failure. Never raises.

        Uses a larger per-request timeout than get_session_intent to cover
        the embedding round-trip, but still small enough that a sluggish
        memory service never adds user-visible latency on top of the
        router (the call runs concurrently with routing).
        """
        if not user_id or not query:
            return []
        try:
            resp = await self._client.post(
                "/internal/memory/retrieve",
                json={"user_id": user_id, "query": query},
                timeout=self._memory_timeout,
            )
            if resp.status_code >= 400:
                return []
            payload = resp.json()
        except Exception:
            log.exception("api_client: retrieve_memories failed")
            return []
        data = payload.get("data") if isinstance(payload, dict) else None
        memories = data.get("memories") if isinstance(data, dict) else None
        if not isinstance(memories, list):
            return []
        return [
            m["text"]
            for m in memories
            if isinstance(m, dict) and isinstance(m.get("text"), str)
        ]
