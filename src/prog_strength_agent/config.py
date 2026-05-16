import os
from dataclasses import dataclass


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class Config:
    anthropic_api_key: str
    jwt_signing_key: str
    mcp_url: str
    host: str
    port: int
    claude_model: str
    max_tokens: int

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

        return cls(
            anthropic_api_key=anthropic_api_key,
            jwt_signing_key=jwt_signing_key,
            # Default targets the shared docker network used in prod. Override
            # to http://localhost:8000/mcp (or similar) for local dev against
            # a uv-run MCP server.
            mcp_url=os.environ.get("PROG_STRENGTH_MCP_URL", "http://mcp:8000/mcp"),
            host=os.environ.get("AGENT_HOST", "0.0.0.0"),
            port=int(os.environ.get("AGENT_PORT", "8001")),
            claude_model=os.environ.get("CLAUDE_MODEL", "claude-opus-4-7"),
            max_tokens=int(os.environ.get("CLAUDE_MAX_TOKENS", "2048")),
        )
