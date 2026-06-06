"""Shared pytest setup.

The `server` module instantiates `Config.from_env()` at import time, so
the required env vars must exist before any test imports it. We set
minimal placeholder credentials (never the real keys) and leave the
API/MCP URLs at their defaults; tests that exercise the routing path
monkeypatch `server.api_client`/`server.HARNESSES` rather than hitting
anything real. `PROG_STRENGTH_API_URL=""` keeps the telemetry + api
clients disabled (None) so no test accidentally opens a connection.
"""

import os

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test")
os.environ.setdefault("JWT_SIGNING_KEY", "test-signing-key")
os.environ.setdefault("PROG_STRENGTH_API_URL", "")
