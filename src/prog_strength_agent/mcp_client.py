"""Persistent MCP ClientSession with reconnect-on-failure.

FastAPI's lifespan handler calls `connect()` at startup and `close()` at
shutdown. In between, `list_tools()` / `call_tool()` are safe to invoke
concurrently from many request handlers — an asyncio.Lock serializes
the (rare) reconnect path so two concurrent failures don't both tear
down and re-open the session.
"""

import asyncio
import logging
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

log = logging.getLogger(__name__)


class MCPClient:
    def __init__(self, url: str):
        self._url = url
        self._session: ClientSession | None = None
        self._stack: AsyncExitStack | None = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        """Open (or re-open) the underlying ClientSession.

        Caller is responsible for not racing this — `_call_with_retry`
        already holds the lock when it reconnects.
        """
        await self._close_unlocked()
        stack = AsyncExitStack()
        try:
            read, write, _ = await stack.enter_async_context(streamablehttp_client(self._url))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
        except Exception:
            await stack.aclose()
            raise
        self._stack = stack
        self._session = session
        log.info("mcp connected: %s", self._url)

    async def close(self) -> None:
        async with self._lock:
            await self._close_unlocked()

    async def _close_unlocked(self) -> None:
        if self._stack is not None:
            try:
                await self._stack.aclose()
            except Exception:
                log.warning("error closing mcp session", exc_info=True)
            self._stack = None
            self._session = None

    async def list_tools(self):
        return await self._call_with_retry("list_tools")

    async def call_tool(self, name: str, arguments: dict[str, Any]):
        return await self._call_with_retry("call_tool", name, arguments)

    async def _call_with_retry(self, method: str, *args: Any):
        """Invoke `method` on the session. On any failure, reconnect once
        and retry — covers the common case of MCP container restarts
        between agent requests.
        """
        for attempt in (1, 2):
            session = self._session
            if session is None:
                async with self._lock:
                    if self._session is None:
                        await self.connect()
                    session = self._session
            try:
                return await getattr(session, method)(*args)
            except Exception:
                log.warning("mcp %s failed (attempt %d)", method, attempt, exc_info=True)
                async with self._lock:
                    await self.connect()
                if attempt == 2:
                    raise
