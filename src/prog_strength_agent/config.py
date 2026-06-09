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
    # OpenAI integration — used by /speak for text-to-speech. Empty
    # api_key disables the endpoint (returns 503 at request time
    # rather than failing startup; useful for local dev without an
    # OpenAI key on hand). See prog-strength-docs/sows/voice-chat.md.
    openai_api_key: str
    openai_tts_model: str
    tts_voice_default: str
    tts_daily_char_cap_per_user: int
    # Natural-language style instructions passed to gpt-4o-mini-tts.
    # The model interprets these as personality + pacing + tone cues,
    # so this is where we keep the Prog Strength coach's "vibe."
    # Ignored by older tts-1/tts-1-hd models that don't accept the
    # `instructions` param — those just fall back to default delivery.
    tts_instructions: str
    # When true, /chat and /speak pre-check GET /me/usage via the
    # UsageGate and return 429 when the user is over their daily
    # allowance. Defaults false so the first deploy lands the TTS
    # telemetry writes without changing user-facing behavior; flip to
    # true once telemetry is validated in prod. See
    # prog-strength-docs/sows/per-user-daily-usage-cap.md.
    usage_gate_enabled: bool

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
        cors_allowed_origins = tuple(o.strip() for o in cors_raw.split(",") if o.strip())

        # Enforcement flag for the usage gate. Truthy parse so any of
        # 1/true/yes/on (case-insensitive) enables it; everything else
        # (including the unset default "false") leaves it off.
        usage_gate_enabled = os.environ.get("USAGE_GATE_ENABLED", "false").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

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
            simple_model=os.environ.get("CLAUDE_MODEL_SIMPLE", "claude-haiku-4-5-20251001"),
            complex_model=os.environ.get("CLAUDE_MODEL_COMPLEX", "claude-sonnet-4-6"),
            router_model=os.environ.get("CLAUDE_MODEL_ROUTER", "claude-haiku-4-5-20251001"),
            max_tokens=int(os.environ.get("CLAUDE_MAX_TOKENS", "2048")),
            cors_allowed_origins=cors_allowed_origins,
            # OpenAI TTS for /speak. Key is optional at startup —
            # /speak returns 503 when unset so local dev without an
            # OpenAI key on hand still boots. Defaults pick the
            # newer gpt-4o-mini-tts model so we can use the `cedar`
            # voice (only available on this model, not on tts-1) and
            # pass an `instructions` string for personality + pacing.
            openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
            openai_tts_model=os.environ.get("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"),
            tts_voice_default=os.environ.get("TTS_VOICE_DEFAULT", "cedar"),
            tts_daily_char_cap_per_user=int(os.environ.get("TTS_DAILY_CHAR_CAP_PER_USER", "50000")),
            # Personality cue: knowledgeable strength coach with a
            # hyped gym-bro energy. One env var so we can tweak the
            # vibe without a code change. Empty string is fine — the
            # OpenAI client just drops the parameter and the model
            # uses its default neutral delivery.
            tts_instructions=os.environ.get(
                "TTS_INSTRUCTIONS",
                (
                    "Personality: a hyped-up strength coach who's "
                    "genuinely stoked to help the user crush their "
                    "workout. Friendly, encouraging, gym-bro energy "
                    "without overdoing it. "
                    "Pacing: FAST. Talk quickly and with momentum, "
                    "like an energized coach mid-set who's pumped "
                    "about what they're saying. Don't draw words "
                    "out; don't pause between sentences. Brisk "
                    "delivery throughout. "
                    "Tone: warm, supportive, real enthusiasm. Think "
                    "of a friend spotting you at the gym, hyping you "
                    "up on the last rep."
                ),
            ),
            usage_gate_enabled=usage_gate_enabled,
        )
