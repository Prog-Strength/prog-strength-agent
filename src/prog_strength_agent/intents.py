"""Intent registry for the Prog Strength agent.

Each known intent declares a `prefetch` (async; runs MCP tool calls in
parallel against an already-open session), a `rules` string (system
prompt addendum the model sees verbatim), and a `format` function
(turns the prefetched dict into a data block for the prompt).

See prog-strength-docs/sows/intent-driven-context-enrichment.md.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable

log = logging.getLogger(__name__)

PrefetchFn = Callable[[Any], Awaitable[dict[str, Any]]]
FormatFn = Callable[[dict[str, Any]], str]

KNOWN_INTENTS: tuple[str, ...] = (
    "log_nutrition",
    "log_workout",
    "log_bodyweight",
    "analyze_progress",
    "general",
)


@dataclass(frozen=True)
class IntentSpec:
    intent: str
    rules: str
    prefetch: PrefetchFn
    format: FormatFn


_SPECS: dict[str, IntentSpec] = {}


def _register(spec: IntentSpec) -> None:
    _SPECS[spec.intent] = spec


async def _noop_prefetch(_session: Any) -> dict[str, Any]:
    return {}


def _noop_format(_data: dict[str, Any]) -> str:
    return ""


# general: no enrichment. Registered explicitly so look-ups succeed for
# the routine path; rules/prefetch/format are all no-ops.
_register(IntentSpec(
    intent="general",
    rules="",
    prefetch=_noop_prefetch,
    format=_noop_format,
))


class IntentRegistry:
    """Static facade — no instances. Each public method is a classmethod
    so the harness can call it as `IntentRegistry.run(...)`.
    """

    @classmethod
    def known(cls) -> Iterable[str]:
        return KNOWN_INTENTS

    @classmethod
    async def run(cls, intent: str, session: Any) -> tuple[str, str]:
        """Run the intent's prefetch and return (rules_block, data_block).

        Failure semantics: any exception inside prefetch is caught, the
        intent's rules are returned unchanged, and the data block is
        empty. The harness records `intent_prefetch_failed=True` based
        on the `failed` return tuple, but this method itself never
        raises.
        """
        spec = _SPECS.get(intent)
        if spec is None:
            return "", ""
        try:
            data = await spec.prefetch(session)
        except Exception:  # noqa: BLE001 — broad by design
            log.exception("intent prefetch failed: intent=%s", intent)
            return spec.rules, ""
        try:
            return spec.rules, spec.format(data)
        except Exception:
            log.exception("intent format failed: intent=%s", intent)
            return spec.rules, ""
