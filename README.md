# prog-strength-agent

User-facing agent service for the Prog Strength training tracker. Wraps Claude
with the [prog-strength-mcp](https://github.com/Prog-Strength/prog-strength-mcp)
tool layer so natural-language requests like "log Tuesday's push day" or
"how has my bench progressed?" map to real reads and writes against
[prog-strength-api](https://github.com/Prog-Strength/prog-strength-api).

## Architecture

```
user → frontend → POST /chat → AGENT ─→ Claude (tools=[…])
                                  │
                                  ├─→ MCP (list_workouts, list_exercises, create_workout)
                                  │       │
                                  │       └─→ API (Go) → SQLite
                                  │
                                  └─ injects authenticated user_id into every tool call
```

- **Persistent MCP session.** Opened at FastAPI startup via the `lifespan`
  handler, reused across all `/chat` requests. Reconnects on transport
  failure.
- **Stateless `/chat` endpoint.** The frontend sends the full message
  history each turn; the agent owns no per-conversation state.
- **JWT-validated requests.** The agent shares `JWT_SIGNING_KEY` with the
  API and trusts tokens minted by the API's OAuth callback. `sub` is the
  authoritative `user_id` — the agent strips `user_id` from the tool
  schemas it shows Claude and injects the authenticated value when
  forwarding to MCP, so Claude can never spoof a different user.

## Configuration

| Env var                     | Required | Default                  | Notes                                                                |
| --------------------------- | -------- | ------------------------ | -------------------------------------------------------------------- |
| `ANTHROPIC_API_KEY`         | yes      | —                        | Standard sk-ant-… key.                                               |
| `JWT_SIGNING_KEY`           | yes      | —                        | MUST match the API's signing key. Used to verify inbound user JWTs. |
| `PROG_STRENGTH_MCP_URL`     | no       | `http://mcp:8000/mcp`    | In prod this resolves over the shared docker network.                |
| `AGENT_HOST`                | no       | `0.0.0.0`                |                                                                      |
| `AGENT_PORT`                | no       | `8001`                   |                                                                      |
| `CLAUDE_MODEL`              | no       | `claude-sonnet-4-6`      | Any Anthropic model id. Default is Sonnet 4.6 — strong tool-use behavior at a fraction of Opus pricing. |
| `CLAUDE_MAX_TOKENS`         | no       | `2048`                   | Per-turn cap.                                                        |

## Local development

```sh
uv sync

# Run the MCP server first (separate shell) — see prog-strength-mcp.
# Then point the agent at it. Both must share the same JWT_SIGNING_KEY.

export ANTHROPIC_API_KEY=sk-ant-…
export JWT_SIGNING_KEY=<dev key>
export PROG_STRENGTH_MCP_URL=http://localhost:8000/mcp

uv run prog-strength-agent
```

The service is then reachable at `http://localhost:8001`. Try:

```sh
curl http://localhost:8001/health
```

## /chat

`POST /chat` accepts an Anthropic-format message list and returns the final
assistant turn after the tool-use loop terminates:

```sh
curl -X POST http://localhost:8001/chat \
  -H "Authorization: Bearer $JWT" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "what exercises do I have for chest?"}
    ]
  }'
```

The current implementation is **non-streaming** — the response arrives
after the full tool-use loop completes. SSE streaming is a planned
follow-up; it'll keep the same `/chat` shape but emit `text_delta` and
`tool_use` events as they happen.

## Deployment

Deployed on the same EC2 host as the API and MCP. The docker-compose
service joins the shared `prog-strength` external network so it can
resolve `mcp` and `api` by service name. Caddy reverse-proxies
`agent.progstrength.fitness` → `agent:8001`.

Deploys run through SSM Run Command (invoking
`prog-strength-infra/deploy/agent.sh`) rather than SSH, with the app's
runtime secrets read from AWS Secrets Manager via the host's instance
role.
