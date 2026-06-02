"""Intent registry for the Prog Strength agent.

Each known intent declares a `prefetch` (async; runs MCP tool calls in
parallel against an already-open session), a `rules` string (system
prompt addendum the model sees verbatim), and a `format` function
(turns the prefetched dict into a data block for the prompt).

See prog-strength-docs/sows/intent-driven-context-enrichment.md.
"""

from __future__ import annotations

import asyncio
import json
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


def _decode_tool_result(result: Any) -> Any:
    """MCP tool results come back as a `content` list of text blocks
    in JSON-stringified form. Stitch them together and json.loads the
    result. Returns [] on empty/bad payloads so callers don't have to
    branch.
    """
    parts = []
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    raw = "".join(parts).strip()
    if not raw:
        return []
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []


async def _log_nutrition_prefetch(session: Any) -> dict[str, Any]:
    pantry_task = session.call_tool("list_pantry_items", {})
    recipes_task = session.call_tool("list_recipes", {})
    pantry_res, recipes_res = await asyncio.gather(pantry_task, recipes_task)
    return {
        "pantry": _decode_tool_result(pantry_res),
        "recipes": _decode_tool_result(recipes_res),
    }


def _log_nutrition_format(data: dict[str, Any]) -> str:
    lines = []
    lines.append("USER'S CURRENT PANTRY (id · name · per-serving macros):")
    for item in data.get("pantry", []):
        macros = (
            f"{item.get('calories', 0)} kcal · "
            f"{item.get('protein_g', 0)}P / "
            f"{item.get('fat_g', 0)}F / "
            f"{item.get('carbs_g', 0)}C"
        )
        serving = f"{item.get('serving_size', 1)} {item.get('serving_unit', 'serving')}"
        lines.append(
            f"- {item.get('id', '?')} · {item.get('name', '?')} · "
            f"{macros} per {serving}"
        )
    if not data.get("pantry"):
        lines.append("- (empty — the user has not saved any pantry items yet)")
    lines.append("")
    lines.append("USER'S CURRENT RECIPES (id · name · component count):")
    for r in data.get("recipes", []):
        components = r.get("components") or []
        lines.append(
            f"- {r.get('id', '?')} · {r.get('name', '?')} · "
            f"{len(components)} component(s)"
        )
    if not data.get("recipes"):
        lines.append("- (empty — the user has not saved any recipes yet)")
    return "\n".join(lines)


_LOG_NUTRITION_RULES = """\
The user is logging a meal or snack. Assume one serving unless the \
user specifies otherwise. The user's saved pantry items and recipes \
are listed below — search them by name first before asking the user \
for macros or creating a new pantry item. Only ask follow-up \
questions about details you genuinely cannot infer.\
"""


_register(IntentSpec(
    intent="log_nutrition",
    rules=_LOG_NUTRITION_RULES,
    prefetch=_log_nutrition_prefetch,
    format=_log_nutrition_format,
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
