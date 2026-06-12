# Prog Strength Agent — Agent Contributor Guide

This file is for AI coding agents (and humans) contributing to
`prog-strength-agent`, the FastAPI service that wraps Claude with the
Prog Strength MCP tools. README.md has the architecture overview;
`prog-strength-docs/sows/` has the design history — read the relevant
SOW before changing a subsystem it covers.

## Releases: PR titles ARE the release trigger

This repo squash-merges. The PR title becomes the commit subject on
`main`, and semantic-release derives releases from those subjects:
`feat:` → minor, `fix:` → patch, anything else (`chore:`, `docs:`,
`ci:`, `refactor:`, `test:`) → **no release, no deploy**.

Consequences, learned the hard way (PR #8 merged with a
non-conventional title and its prompt changes silently didn't deploy):

- **Every PR title must be a conventional commit with a lowercase
  subject** — `feat(agent): lookup-first custom meal prompts`, not
  `Lookup-first custom meal prompts`. CI enforces this on every PR.
- **Pick the type for what should happen on merge.** Behavior changes
  the deployed agent must pick up (prompts, routing, harness, config)
  are `feat:`/`fix:`. CI/docs/eval-tooling-only changes are
  `chore:`/`docs:`/`ci:` and intentionally do not deploy.
- If a release was missed (merged with a wrong title), the remedy is an
  empty conventional commit on `main`:
  `git commit --allow-empty -m "feat(agent): release <what was missed>"`.

## Verify before authoring a PR

There are no per-clone git hooks in this repo; run the checks yourself:

```
uv run pytest
uv run ruff check src tests evals
```

## Eval cost policy — never add automatic LLM spend

The owner pays Anthropic per token on a small budget and has been
rate-limited. The macro-accuracy eval (`evals/`) is therefore
**strictly opt-in**: an `eval:run` PR label or a manual
`workflow_dispatch`, smoke subset (24 cases × 1 trial) by default.

- Do NOT add `push`/`pull_request` triggers that call the Anthropic
  API, raise the default trials/concurrency, or re-introduce automatic
  baseline re-runs.
- The everyday eval loop is local:
  `uv run python -m evals.run_eval --smoke --api-path ../prog-strength-api --mcp-path ../prog-strength-mcp`
- Full-dataset runs are reserved for deliberate baseline publishes
  (`workflow_dispatch` with `full` + `publish_baseline` from main).
- This applies to ALL new features: anything that calls an LLM defaults
  to the cheapest useful configuration and is opt-in, with the
  estimated token cost stated in the PR.

## Conventions worth knowing

- **Prompts live here, not in MCP.** The MCP server is a transparent
  forwarder; agent behavior (tone, tool-use rules, intent rules) is
  prompt code in `src/prog_strength_agent/prompt.py` + `intents.py`.
  Prompt content changes need matching assertions in
  `tests/test_prompt.py`.
- **Model routing**: `model_router.py` classifies each turn to a tier
  (simple=Haiku, complex=Sonnet) + intent in one Haiku call. Routing
  rule changes are cheap to write and expensive to validate — cite
  eval numbers (see `evals/`) when changing them.
- **External integrations** (anything that isn't an LLM-provider SDK)
  belong in `prog-strength-api`, reached through MCP tools — not here.
- Python 3.12+, `uv` for everything, ruff line length 100, pytest with
  `asyncio_mode = "auto"`.
