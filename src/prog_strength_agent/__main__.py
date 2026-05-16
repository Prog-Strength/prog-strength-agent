"""Console entry point: `prog-strength-agent` (declared in pyproject.toml).

Starts the FastAPI app under uvicorn. `proxy_headers` + `forwarded_allow_ips`
tell uvicorn to trust Caddy's `X-Forwarded-Proto: https` / `X-Forwarded-Host`
headers so any URL-building in the app sees the public HTTPS context
rather than the plain-HTTP hop from Caddy on the docker network.
"""

import uvicorn

from prog_strength_agent.server import app, config


def main() -> None:
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    main()
