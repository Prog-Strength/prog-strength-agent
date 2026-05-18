import os
from dataclasses import dataclass


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class Config:
    anthropic_api_key: str
    jwt_signing_key: str
    mcp_url: str
    # Base URL of the Go API, used by the telemetry client to POST
    # to /internal/telemetry/*. Defaults to the in-Docker hostname.
    # Empty disables telemetry entirely (useful for local dev where
    # the API isn't running).
    api_url: str
    host: str
    port: int
    # Three model slots — see model_router.py for routing logic.
    # simple_model: cheap/fast tier, serves CRUD-shaped requests.
    # complex_model: capable tier, serves reasoning/analysis requests.
    # router_model: classifier that picks between the two above.
    # All three are typically Claude family ids; nothing stops you
    # from pointing simple and router at the same model (cheaper).
    simple_model: str
    complex_model: str
    router_model: str
    max_tokens: int
    cors_allowed_origins: tuple[str, ...]

    @classmethod
    def from_env(cls) -> "Config":
        try:
            anthropic_api_key = os.environ["ANTHROPIC_API_KEY"]
        except KeyError as e:
            raise ConfigError(
                "ANTHROPIC_API_KEY is required — sk-ant-… from console.anthropic.com."
            ) from e
        try:
            jwt_signing_key = os.environ["JWT_SIGNING_KEY"]
        except KeyError as e:
            raise ConfigError(
                "JWT_SIGNING_KEY is required — must match the API's signing key so "
                "incoming user JWTs validate."
            ) from e

        # Comma-separated list of origins permitted by CORS. The frontend
        # is hosted on Vercel so /chat is always cross-origin; without
        # this the browser blocks the request before it ever leaves.
        cors_raw = os.environ.get("CORS_ALLOWED_ORIGINS", "")
        cors_allowed_origins = tuple(
            o.strip() for o in cors_raw.split(",") if o.strip()
        )

        return cls(
            anthropic_api_key=anthropic_api_key,
            jwt_signing_key=jwt_signing_key,
            # Default targets the shared docker network used in prod. Override
            # to http://localhost:8000/mcp (or similar) for local dev against
            # a uv-run MCP server.
            mcp_url=os.environ.get("PROG_STRENGTH_MCP_URL", "http://mcp:8000/mcp"),
            # Default targets the API container on the same Docker network.
            # Empty string disables telemetry — useful for local dev where
            # the agent runs against a uv-run MCP but no API.
            api_url=os.environ.get("PROG_STRENGTH_API_URL", "http://api:8080"),
            host=os.environ.get("AGENT_HOST", "0.0.0.0"),
            port=int(os.environ.get("AGENT_PORT", "8001")),
            # Tiered model defaults: Haiku for the common case (workout
            # logging, list operations, exercise lookup), Sonnet for the
            # uncommon analytical / planning case. Router uses Haiku
            # because the classification task itself is structured
            # (single-word output, well-bounded prompt).
            simple_model=os.environ.get(
                "CLAUDE_MODEL_SIMPLE", "claude-haiku-4-5-20251001"
            ),
            complex_model=os.environ.get(
                "CLAUDE_MODEL_COMPLEX", "claude-sonnet-4-6"
            ),
            router_model=os.environ.get(
                "CLAUDE_MODEL_ROUTER", "claude-haiku-4-5-20251001"
            ),
            max_tokens=int(os.environ.get("CLAUDE_MAX_TOKENS", "2048")),
            cors_allowed_origins=cors_allowed_origins,
        )
