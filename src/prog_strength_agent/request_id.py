"""Per-request correlation id, mirroring the Go API's `internal/requestid`.

Every inbound HTTP request gets a `request_id`: a 32-char hex string
(`secrets.token_hex(16)` — the same shape as the API's `internal/id.New`,
which is `crypto/rand` 16 bytes hex-encoded, NOT a dashed uuid4). The id
is minted by `RequestIDMiddleware` unless the caller already sent an
`X-Request-ID` header, in which case that value is honored so a single
trace can span the frontend → agent → MCP → API hops.

The id is:
  - echoed back on the `X-Request-ID` response header of every request,
  - stored in a ContextVar so any code on the request's task can read it
    via `current_request_id()` (the /health body, the /chat SSE
    done/error events, the error envelopes) without threading it through
    every call signature,
  - stamped onto every `prog_strength_agent` log record by
    `RequestIDLogFilter` so a request can be reverse-searched in logs.

This is a pure-ASGI middleware (not Starlette's BaseHTTPMiddleware) on
purpose: BaseHTTPMiddleware runs the route handler in a separate task,
so a ContextVar set in its `dispatch` would not be visible downstream.
A pure-ASGI middleware sets the var in the same task the rest of the app
runs in, so the value propagates everywhere — including the
StreamingResponse generator that drives /chat.
"""

import logging
import secrets
from contextvars import ContextVar

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

# Canonical header name. Matches the API's requestid.HeaderName so the
# same id rides the same header across every service.
HEADER_NAME = "X-Request-ID"

# Default "" so code outside any request context (background tasks,
# tests that bypass the middleware) reads an empty id and treats it the
# same way the API's `omitempty` does — simply not surfaced.
_request_id: ContextVar[str] = ContextVar("request_id", default="")


def new_request_id() -> str:
    """Mint a 32-char hex id. Deliberately matches the API's
    `internal/id.New` (16 random bytes, hex-encoded) so ids look
    identical across services — not a dashed uuid4."""
    return secrets.token_hex(16)


def current_request_id() -> str:
    """The current request's id, or "" outside any request context."""
    return _request_id.get()


class RequestIDMiddleware:
    """Mint-or-accept a request id, expose it on the `X-Request-ID`
    response header, and seed the ContextVar for the request's lifetime.

    Add this LAST (outermost) so it wraps every other layer: the id is
    set before any handler runs and the response header is stamped on
    every reply, including CORS preflight and error responses.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        inbound = Headers(scope=scope).get(HEADER_NAME)
        request_id = inbound or new_request_id()
        token = _request_id.set(request_id)

        async def send_with_header(message: Message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)[HEADER_NAME] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_header)
        finally:
            _request_id.reset(token)


class RequestIDLogFilter(logging.Filter):
    """Stamp `request_id` onto every record so the configured formatter
    can render it. Records emitted outside a request get "-" so the
    format string never raises a KeyError."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = current_request_id() or "-"
        return True


_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s [request_id=%(request_id)s] %(message)s"
_logging_configured = False


def configure_logging(level: int = logging.INFO) -> None:
    """Attach a stdout handler that renders `request_id` to the root
    logger and raise the `prog_strength_agent` logger to `level`.

    Idempotent (safe across re-import and repeated calls). The handler
    lives on the root logger (not the package logger) so app records
    reach it via normal propagation — which means pytest's `caplog` and
    any other root handler still see them, with no double-emit. The
    RequestIDLogFilter sits on the handler, so every record it formats
    (including third-party libraries that propagate to root) gets a
    `request_id` attribute and the format string never KeyErrors.
    uvicorn's own loggers set propagate=False, so they are unaffected.
    """
    global _logging_configured
    if _logging_configured:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    handler.addFilter(RequestIDLogFilter())
    logging.getLogger().addHandler(handler)
    # The package logger's level gates whether INFO records are created
    # at all; root's level only gates records originating at root.
    logging.getLogger("prog_strength_agent").setLevel(level)
    _logging_configured = True
