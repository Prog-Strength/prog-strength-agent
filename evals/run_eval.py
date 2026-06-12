"""Macro-accuracy eval runner.

Stands up the full production pipeline locally — the REAL Go API on a
temp SQLite file, the real MCP server (from --mcp-path) pointed at it,
and the real ModelRouter + ModelHarness — then drives every dataset
case through it, scoring the macros the agent actually logs (read
straight from the database) against published ground truth.

    uv run python -m evals.run_eval \
        --dataset evals/dataset/custom_meals.json \
        --api-path ../prog-strength-api \
        --mcp-path ../prog-strength-mcp \
        --trials 3 --out eval-results.json --summary-out eval-summary.md

Requires ANTHROPIC_API_KEY and a Go toolchain (the API is built from
--api-path). FATSECRET_CLIENT_ID/SECRET and USDA_FDC_API_KEY are
forwarded to the API process when set; without them the lookup
endpoint degrades and the eval measures pure-estimation behavior
(which is itself a valid baseline).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from anthropic import AsyncAnthropic

from evals import compare
from evals.eval_api import GoAPIServer, free_port, mint_jwt, read_logged_macros
from evals.scoring import (
    Case,
    CaseResult,
    load_dataset,
    results_to_json,
    score_trial,
)
from prog_strength_agent.model_harness import ModelHarness
from prog_strength_agent.model_router import ModelRouter

log = logging.getLogger("evals")

# Mirror the agent config's defaults (config.py) without importing
# Config.from_env, which requires JWT_SIGNING_KEY the eval doesn't have.
DEFAULT_SIMPLE = "claude-haiku-4-5-20251001"
DEFAULT_COMPLEX = "claude-sonnet-4-6"
DEFAULT_ROUTER = "claude-haiku-4-5-20251001"


async def run(args: argparse.Namespace) -> int:
    cases = load_dataset(args.dataset)
    if args.smoke:
        cases = [c for c in cases if c.smoke]
    if args.category:
        cases = [c for c in cases if c.category in args.category]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("no cases selected", file=sys.stderr)
        return 2
    log.info(
        "running %d cases × %d trial(s) = %d LLM trials%s",
        len(cases),
        args.trials,
        len(cases) * args.trials,
        " (smoke subset)" if args.smoke else "",
    )

    with tempfile.TemporaryDirectory(prefix="macro-eval-") as tmp:
        workdir = Path(tmp)
        api = GoAPIServer(args.api_path, workdir, free_port())
        log.info("building API from %s …", args.api_path)
        api.build()
        await api.start()
        mcp_port = free_port()
        mcp_proc = _spawn_mcp(args.mcp_path, api.base_url, mcp_port)
        try:
            await _wait_for_mcp(mcp_proc, mcp_port)
            results = await _run_cases(
                cases,
                db_path=api.db_path,
                mcp_url=f"http://127.0.0.1:{mcp_port}/mcp",
                trials=args.trials,
                concurrency=args.concurrency,
            )
        finally:
            mcp_proc.terminate()
            mcp_proc.wait(timeout=10)
            api.stop()

    doc = results_to_json(results, meta=_build_meta(args, len(cases)))
    Path(args.out).write_text(json.dumps(doc, indent=2) + "\n")
    summary = compare.render_markdown(current=doc, baseline=None)
    if args.summary_out:
        Path(args.summary_out).write_text(summary)
    print(summary)

    aggregates = doc["aggregates"]
    print(
        f"\ncomposite={aggregates['composite']} "
        f"({aggregates['total_cases']} cases × {args.trials} trials) "
        f"→ {args.out}",
        file=sys.stderr,
    )
    return 0


async def _run_cases(
    cases: list[Case],
    *,
    db_path: Path,
    mcp_url: str,
    trials: int,
    concurrency: int,
) -> list[CaseResult]:
    client = AsyncAnthropic()  # reads ANTHROPIC_API_KEY
    router = ModelRouter(client, os.environ.get("CLAUDE_MODEL_ROUTER", DEFAULT_ROUTER))
    harnesses = {
        "simple": ModelHarness(
            client, os.environ.get("CLAUDE_MODEL_SIMPLE", DEFAULT_SIMPLE), mcp_url
        ),
        "complex": ModelHarness(
            client, os.environ.get("CLAUDE_MODEL_COMPLEX", DEFAULT_COMPLEX), mcp_url
        ),
    }

    semaphore = asyncio.Semaphore(concurrency)
    results = [CaseResult(case=case) for case in cases]
    done = 0

    async def one_trial(case_result: CaseResult, trial_index: int) -> None:
        nonlocal done
        async with semaphore:
            trial = await _run_trial(
                case_result.case, trial_index, db_path, router, harnesses
            )
            case_result.trials.append(trial)
            done += 1
            log.info(
                "[%d/%d] %s t%d %s",
                done,
                len(cases) * trials,
                case_result.case.id,
                trial_index,
                "no_log" if trial.no_log else f"cal_ape={trial.ape['calories']:.0f}%",
            )

    await asyncio.gather(*(one_trial(r, t) for r in results for t in range(trials)))
    return results


async def _run_trial(
    case: Case,
    trial_index: int,
    db_path: Path,
    router: ModelRouter,
    harnesses: dict[str, ModelHarness],
) -> Any:
    # The minted JWT's subject is the correlation id: the API writes
    # log rows under it, so concurrent trials never interleave. The
    # API trusts `sub` statelessly — no user bootstrapping needed.
    user_id = f"eval-{case.id}-t{trial_index}"
    token = mint_jwt(user_id)
    messages: list[dict[str, Any]] = [{"role": "user", "content": case.message}]
    try:
        decision = await router.route(messages)
        harness = harnesses.get(decision.tier, harnesses["simple"])
        # Drain the SSE stream; the observable outcome we score is what
        # landed in the database, not the streamed text.
        async for _ in harness.stream_chat(
            list(messages),
            token,
            intent=decision.intent,
            client_timezone="UTC",
        ):
            pass
    except Exception as exc:
        log.warning("trial errored (%s t%d): %s", case.id, trial_index, exc)
        trial = score_trial(None, case.expect)
        trial.error = str(exc)
        return trial

    logged = await asyncio.to_thread(read_logged_macros, db_path, user_id)
    return score_trial(logged, case.expect)


def _spawn_mcp(mcp_path: str, api_base_url: str, port: int) -> subprocess.Popen[bytes]:
    env = dict(os.environ)
    env.update(
        {
            "PROG_STRENGTH_API_BASE_URL": api_base_url,
            "MCP_HOST": "127.0.0.1",
            "MCP_PORT": str(port),
        }
    )
    return subprocess.Popen(
        ["uv", "run", "--project", mcp_path, "python", "-m", "prog_strength_mcp"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


async def _wait_for_mcp(
    proc: subprocess.Popen[bytes], port: int, timeout_s: float = 60.0
) -> None:
    deadline = time.monotonic() + timeout_s
    url = f"http://127.0.0.1:{port}/health"
    async with httpx.AsyncClient() as client:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                stderr = (proc.stderr.read() if proc.stderr else b"").decode()
                raise RuntimeError(f"MCP server exited during startup:\n{stderr}")
            try:
                resp = await client.get(url, timeout=2.0)
                if resp.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.25)
    raise RuntimeError(f"MCP server not healthy after {timeout_s}s")


def _build_meta(args: argparse.Namespace, case_count: int) -> dict[str, Any]:
    return {
        "ran_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": os.environ.get("GITHUB_SHA") or _local_git_sha(),
        "dataset": str(args.dataset),
        "cases": case_count,
        "trials": args.trials,
        "models": {
            "router": os.environ.get("CLAUDE_MODEL_ROUTER", DEFAULT_ROUTER),
            "simple": os.environ.get("CLAUDE_MODEL_SIMPLE", DEFAULT_SIMPLE),
            "complex": os.environ.get("CLAUDE_MODEL_COMPLEX", DEFAULT_COMPLEX),
        },
        "lookup_configured": bool(
            os.environ.get("FATSECRET_CLIENT_ID") or os.environ.get("USDA_FDC_API_KEY")
        ),
    }


def _local_git_sha() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                check=True,
            )
            .stdout.decode()
            .strip()
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="evals/dataset/custom_meals.json")
    parser.add_argument(
        "--api-path",
        required=True,
        help="path to a prog-strength-api checkout (Go toolchain required)",
    )
    parser.add_argument(
        "--mcp-path",
        required=True,
        help="path to a prog-strength-mcp checkout (uv project)",
    )
    # Cost-conscious defaults: 1 trial and low concurrency. Every trial
    # spends real tokens; bump --trials for noise-damped baselines, not
    # for the everyday loop. Concurrency 2 stays under low-tier
    # Anthropic rate limits (the SDK retries 429s, but not hammering
    # the limit at all is faster and cheaper).
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run only the curated ~24-case smoke subset (the cheap default for CI/PRs)",
    )
    parser.add_argument("--out", default="eval-results.json")
    parser.add_argument("--summary-out", default="eval-summary.md")
    parser.add_argument(
        "--category",
        action="append",
        choices=["chain", "packaged", "generic"],
        help="restrict to one or more categories (repeatable)",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="cap case count (smoke runs)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is required", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
