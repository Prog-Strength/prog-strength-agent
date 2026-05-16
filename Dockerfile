# Python 3.12 slim base + uv from the official upstream image. Mirrors
# the layout used by prog-strength-mcp so the two services build the
# same way (and the same caching tricks apply).

FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Run as a non-root user. The agent doesn't need root, and the attack
# surface of this container is "anything that can reach :8001" — drop
# privileges before installing or running anything.
RUN useradd --create-home --shell /bin/bash app
WORKDIR /app
RUN chown app:app /app
USER app

# Layer 1: dependencies only. `--no-install-project` keeps our own
# package out of this layer so it invalidates only when pyproject.toml
# or uv.lock change. README.md must be present because pyproject.toml
# declares `readme = "README.md"` and hatchling validates that field
# during the second sync below.
COPY --chown=app:app pyproject.toml uv.lock* README.md ./
RUN uv sync --frozen --no-dev --no-install-project || uv sync --no-dev --no-install-project

# Layer 2: project source.
COPY --chown=app:app src/ ./src/
RUN uv sync --frozen --no-dev || uv sync --no-dev

EXPOSE 8001

# `uv run` activates the project venv and execs the console script
# declared in pyproject.toml.
CMD ["uv", "run", "--no-dev", "prog-strength-agent"]
