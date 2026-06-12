"""Real Prog Strength API orchestration for eval runs.

The eval drives the actual Go API binary instead of a fake: lookup +
durable-cache code paths get exercised end-to-end, and an empty pantry
falls out naturally from an empty temp database.

Hermetic by construction (verified against internal/config/config.go):
the API needs only JWT_SIGNING_KEY and a DATABASE_URL; Google OAuth is
optional and unmounted when absent. Auth is stateless HS256 trusting
the `sub` claim with no users-table lookup, and nutrition rows carry
no FK to users — so the runner mints one JWT per (case, trial) and the
unique subject doubles as the correlation id. After a trial drains the
SSE stream, the macros the agent actually logged are read straight out
of the temp SQLite file.
"""

from __future__ import annotations

import asyncio
import os
import socket
import sqlite3
import subprocess
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import jwt

# Shared eval-only signing key. Never a real deployment value — the API
# process and the JWT mint below both receive it for the run's lifetime.
# ≥32 bytes to satisfy pyjwt's HS256 key-length recommendation.
SIGNING_KEY = "macro-eval-signing-key-not-a-real-secret"

MACRO_COLUMNS = ("calories", "protein_g", "fat_g", "carbs_g")

# Provider keys forwarded to the API process when present so the
# lookup endpoint runs live; absent keys mean the eval measures
# pure-estimation behavior, which is itself a valid baseline.
FORWARDED_ENV = (
    "FATSECRET_CLIENT_ID",
    "FATSECRET_CLIENT_SECRET",
    "USDA_FDC_API_KEY",
)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def mint_jwt(user_id: str, signing_key: str = SIGNING_KEY) -> str:
    """Mint an HS256 JWT the API accepts: sub + iat + exp, matching
    internal/auth/jwt.go's RegisteredClaims shape."""
    now = datetime.now(UTC)
    return jwt.encode(
        {"sub": user_id, "iat": now, "exp": now + timedelta(hours=2)},
        signing_key,
        algorithm="HS256",
    )


def read_logged_macros(db_path: Path | str, user_id: str) -> dict[str, float] | None:
    """Sum every nutrition-log row the trial's user produced. Agents may
    split one described meal into several log calls; the day's total is
    what the user experiences, so it's what we score. None = the agent
    never logged anything (the `no_log` failure mode)."""
    # The API process holds the write connection; a generous busy
    # timeout covers reads landing mid-transaction.
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10.0)
    try:
        rows = conn.execute(
            """
            SELECT calories, protein_g, fat_g, carbs_g
            FROM nutrition_log_entries
            WHERE user_id = ? AND deleted_at IS NULL
            """,
            (user_id,),
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        return None
    return {
        macro: sum(float(row[i]) for row in rows)
        for i, macro in enumerate(MACRO_COLUMNS)
    }


class GoAPIServer:
    """Builds and runs the real Go API against a temp SQLite file."""

    def __init__(self, api_path: str, workdir: Path, port: int):
        self.api_path = Path(api_path)
        self.workdir = workdir
        self.port = port
        self.db_path = workdir / "eval.db"
        self._binary = workdir / "prog-strength-api-eval"
        self._proc: subprocess.Popen[bytes] | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def build(self) -> None:
        """`go build` once per run — faster and quieter than `go run`,
        and a compile error surfaces before any LLM spend."""
        subprocess.run(
            ["go", "build", "-o", str(self._binary), "./cmd/api"],
            cwd=self.api_path,
            check=True,
        )

    async def start(self, timeout_s: float = 30.0) -> None:
        env = dict(os.environ)
        env.update(
            {
                "DATABASE_URL": str(self.db_path),
                "JWT_SIGNING_KEY": SIGNING_KEY,
                "SERVER_ADDR": f"127.0.0.1:{self.port}",
            }
        )
        for key in FORWARDED_ENV:
            if os.environ.get(key):
                env[key] = os.environ[key]
        self._proc = subprocess.Popen(
            [str(self._binary)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        deadline = time.monotonic() + timeout_s
        async with httpx.AsyncClient() as client:
            while time.monotonic() < deadline:
                if self._proc.poll() is not None:
                    stderr = (self._proc.stderr.read() if self._proc.stderr else b"").decode()
                    raise RuntimeError(f"API exited during startup:\n{stderr}")
                try:
                    resp = await client.get(f"{self.base_url}/health", timeout=2.0)
                    if resp.status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.2)
        raise RuntimeError(f"API not healthy after {timeout_s}s")

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            self._proc.wait(timeout=10)
