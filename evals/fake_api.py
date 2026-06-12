"""Fake Prog Strength API for eval runs.

The MCP server is pointed here instead of the real Go API. Two jobs:

1. Serve an empty pantry/recipes so every dataset case takes the
   custom-meal path (the thing under evaluation).
2. Record every nutrition-log write, keyed by the Authorization header.
   The runner mints a unique bearer token per (case, trial), so trials
   can run concurrently against one shared recorder without
   interleaving — the token IS the correlation id.

Response shapes mirror the real API's `{service, message, data}`
envelope (see prog-strength-mcp's APIClient, which unwraps `data`).
"""

from __future__ import annotations

import asyncio
import socket
from collections import defaultdict
from typing import Any

import uvicorn
from fastapi import FastAPI, Request


class Recorder:
    def __init__(self) -> None:
        self._custom: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._consumption: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def record_custom(self, token: str, payload: dict[str, Any]) -> None:
        self._custom[token].append(payload)

    def record_consumption(self, token: str, payload: dict[str, Any]) -> None:
        self._consumption[token].append(payload)

    def custom_meals(self, token: str) -> list[dict[str, Any]]:
        return list(self._custom.get(token, []))

    def consumption_logs(self, token: str) -> list[dict[str, Any]]:
        return list(self._consumption.get(token, []))


def build_app(recorder: Recorder) -> FastAPI:
    app = FastAPI()

    def _token(request: Request) -> str:
        return request.headers.get("authorization", "")

    def _envelope(data: Any) -> dict[str, Any]:
        return {"service": "fake-api", "message": "ok", "data": data}

    @app.get("/pantry-items")
    async def pantry(_: Request) -> dict[str, Any]:
        return _envelope([])

    @app.get("/recipes")
    async def recipes(_: Request) -> dict[str, Any]:
        return _envelope([])

    @app.post("/nutrition-log/custom")
    async def custom(request: Request) -> dict[str, Any]:
        payload = await request.json()
        recorder.record_custom(_token(request), payload)
        return _envelope({"id": "eval-entry", **payload})

    @app.post("/nutrition-log")
    async def consumption(request: Request) -> dict[str, Any]:
        # With an empty pantry the agent shouldn't reach this; record it
        # anyway so a hallucinated-pantry-id regression shows up as
        # no_log + a visible consumption record, not silence.
        payload = await request.json()
        recorder.record_consumption(_token(request), payload)
        return _envelope({"id": "eval-entry", **payload})

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return _envelope({"healthy": True})

    # Anything else the agent's tools touch (daily macros, bodyweight,
    # workouts…) gets an empty list — enough for prefetch paths to
    # degrade gracefully without 404 noise in the transcripts.
    @app.get("/{rest:path}")
    async def catch_all(rest: str) -> dict[str, Any]:
        return _envelope([])

    return app


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class FakeAPIServer:
    """In-process uvicorn wrapper with explicit start/stop, so the
    runner can stand the API up before launching the MCP subprocess
    and tear it down after."""

    def __init__(self, recorder: Recorder, port: int):
        self.recorder = recorder
        self.port = port
        config = uvicorn.Config(
            build_app(recorder),
            host="127.0.0.1",
            port=port,
            log_level="warning",
        )
        self._server = uvicorn.Server(config)
        self._task: asyncio.Task[None] | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def start(self) -> None:
        self._task = asyncio.create_task(self._server.serve())
        while not self._server.started:
            if self._task.done():
                # Surface bind/startup errors instead of spinning.
                self._task.result()
                raise RuntimeError("fake API server exited before starting")
            await asyncio.sleep(0.05)

    async def stop(self) -> None:
        self._server.should_exit = True
        if self._task is not None:
            await self._task
