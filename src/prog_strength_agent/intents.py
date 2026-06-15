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
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

log = logging.getLogger(__name__)

PrefetchFn = Callable[[Any], Awaitable[dict[str, Any]]]
FormatFn = Callable[[dict[str, Any]], str]

KNOWN_INTENTS: tuple[str, ...] = (
    "log_nutrition",
    "log_workout",
    "log_bodyweight",
    "log_daily_steps",
    "analyze_progress",
    "plan_workout",
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
for macros or creating a new pantry item. For a one-off external meal \
that matches nothing in the pantry, call lookup_food_nutrition to get \
real macro data before estimating macros yourself. Only ask follow-up \
questions about details you genuinely cannot infer.\
"""


_register(IntentSpec(
    intent="log_nutrition",
    rules=_LOG_NUTRITION_RULES,
    prefetch=_log_nutrition_prefetch,
    format=_log_nutrition_format,
))


async def _log_workout_prefetch(session: Any) -> dict[str, Any]:
    catalog_task = session.call_tool("list_exercises", {})
    workouts_task = session.call_tool("list_workouts", {})
    catalog_res, workouts_res = await asyncio.gather(catalog_task, workouts_task)
    workouts = _decode_tool_result(workouts_res)
    # API returns ~50 most-recent (newest first, ORDER BY performed_at
    # DESC); take the first 5 for prompt size.
    return {
        "catalog": _decode_tool_result(catalog_res),
        "recent_workouts": workouts[:5],
    }


def _log_workout_format(data: dict[str, Any]) -> str:
    lines = []
    lines.append("EXERCISE CATALOG (slug · name · primary muscle groups):")
    for e in data.get("catalog", []):
        muscles = ", ".join(e.get("muscle_groups", []) or [])
        lines.append(f"- {e.get('id', '?')} · {e.get('name', '?')} · {muscles}")
    if not data.get("catalog"):
        lines.append("- (catalog unavailable)")
    lines.append("")
    lines.append("RECENT WORKOUTS (id · performed_at · exercise count):")
    for w in data.get("recent_workouts", []):
        lines.append(
            f"- {w.get('id', '?')} · "
            f"{w.get('performed_at', '?')} · "
            f"{len(w.get('exercises') or [])} exercise(s)"
        )
    if not data.get("recent_workouts"):
        lines.append("- (no recent workouts logged)")
    return "\n".join(lines)


_LOG_WORKOUT_RULES = """\
The user is logging a completed workout. The exercise catalog is \
below — match the user's wording to an exercise slug without asking \
unless genuinely ambiguous. The user's last few workouts are also \
below; if they say "same as last time" or use a shorthand, infer \
from those before asking. Look up exercise slugs from the catalog \
before calling create_workout.\
"""


_register(IntentSpec(
    intent="log_workout",
    rules=_LOG_WORKOUT_RULES,
    prefetch=_log_workout_prefetch,
    format=_log_workout_format,
))


async def _plan_workout_prefetch(session: Any) -> dict[str, Any]:
    catalog_task = session.call_tool("list_exercises", {})
    workouts_task = session.call_tool("list_workouts", {})
    catalog_res, workouts_res = await asyncio.gather(catalog_task, workouts_task)
    workouts = _decode_tool_result(workouts_res)
    # API returns ~50 most-recent (newest first, ORDER BY performed_at
    # DESC); take the first 5 for prompt size. Recent sessions inform a
    # sensible forward split.
    return {
        "catalog": _decode_tool_result(catalog_res),
        "recent_workouts": workouts[:5],
    }


def _plan_workout_format(data: dict[str, Any]) -> str:
    lines = []
    lines.append("EXERCISE CATALOG (slug · name · primary muscle groups):")
    for e in data.get("catalog", []):
        muscles = ", ".join(e.get("muscle_groups", []) or [])
        lines.append(f"- {e.get('id', '?')} · {e.get('name', '?')} · {muscles}")
    if not data.get("catalog"):
        lines.append("- (catalog unavailable)")
    lines.append("")
    lines.append("RECENT WORKOUTS (id · performed_at · exercise count):")
    for w in data.get("recent_workouts", []):
        lines.append(
            f"- {w.get('id', '?')} · "
            f"{w.get('performed_at', '?')} · "
            f"{len(w.get('exercises') or [])} exercise(s)"
        )
    if not data.get("recent_workouts"):
        lines.append("- (no recent workouts logged)")
    return "\n".join(lines)


_PLAN_WORKOUT_RULES = """\
The user wants to plan FUTURE training (not log a completed session). \
Use create_planned_workout once per training day, building the schedule \
in the user's timezone with RFC3339 windows; look up exercise slugs from \
the catalog below for any target agenda; space rest days sensibly. Only \
push to Google Calendar (schedule_workout_to_calendar) if the user \
explicitly asks.\
"""


_register(IntentSpec(
    intent="plan_workout",
    rules=_PLAN_WORKOUT_RULES,
    prefetch=_plan_workout_prefetch,
    format=_plan_workout_format,
))


async def _log_bodyweight_prefetch(session: Any) -> dict[str, Any]:
    since = (datetime.now(UTC) - timedelta(days=14)).isoformat().replace("+00:00", "Z")
    res = await session.call_tool("list_bodyweight", {"since": since})
    return {"entries": _decode_tool_result(res)}


def _log_bodyweight_format(data: dict[str, Any]) -> str:
    entries = data.get("entries") or []
    if not entries:
        return "RECENT BODYWEIGHT (last 14 days): (no entries yet)"
    lines = ["RECENT BODYWEIGHT (last 14 days, most recent first):"]
    for e in entries:
        lines.append(
            f"- {e.get('measured_at', '?')} · "
            f"{e.get('weight', '?')} {e.get('unit', '?')}"
        )
    return "\n".join(lines)


_LOG_BODYWEIGHT_RULES = """\
The user is logging a bodyweight reading. Default the unit to whatever \
they used most recently (visible in the entries below). If the new \
reading is meaningfully different from the recent trend, acknowledge \
it briefly; otherwise just confirm the log.\
"""


_register(IntentSpec(
    intent="log_bodyweight",
    rules=_LOG_BODYWEIGHT_RULES,
    prefetch=_log_bodyweight_prefetch,
    format=_log_bodyweight_format,
))


async def _log_daily_steps_prefetch(session: Any) -> dict[str, Any]:
    since = (datetime.now(UTC) - timedelta(days=14)).date().isoformat()
    res = await session.call_tool("get_steps", {"since": since})
    decoded = _decode_tool_result(res)
    # get_steps returns {steps, next_before}; pull the list out. Handle a
    # bare list defensively in case the tool shape changes.
    if isinstance(decoded, dict):
        entries = decoded.get("steps") or []
    elif isinstance(decoded, list):
        entries = decoded
    else:
        entries = []
    return {"entries": entries}


def _log_daily_steps_format(data: dict[str, Any]) -> str:
    entries = data.get("entries") or []
    if not entries:
        return "RECENT STEPS (last 14 days): (no entries yet)"
    lines = ["RECENT STEPS (last 14 days, most recent first):"]
    for e in entries:
        lines.append(
            f"- {e.get('date', '?')} · {e.get('steps', '?')} steps"
        )
    return "\n".join(lines)


_LOG_DAILY_STEPS_RULES = """\
The user is logging a daily step total. Resolve any relative date \
("today", "yesterday") to an explicit calendar day (YYYY-MM-DD) before \
calling log_steps, and CONFIRM the date you logged back to the user. \
Logging a day replaces that day's total. If a step goal is set, you may \
briefly note progress toward it.\
"""


_register(IntentSpec(
    intent="log_daily_steps",
    rules=_LOG_DAILY_STEPS_RULES,
    prefetch=_log_daily_steps_prefetch,
    format=_log_daily_steps_format,
))


async def _analyze_progress_prefetch(session: Any) -> dict[str, Any]:
    now = datetime.now(UTC)
    since = (now - timedelta(days=30)).isoformat().replace("+00:00", "Z")
    until = now.isoformat().replace("+00:00", "Z")
    workouts_task = session.call_tool("list_workouts", {})
    macros_task = session.call_tool("get_daily_macros", {"since": since, "until": until})
    workouts_res, macros_res = await asyncio.gather(workouts_task, macros_task)
    workouts = _decode_tool_result(workouts_res)
    # API returns ~50 most-recent newest-first; take the first 20 for prompt size.
    return {
        "workouts": workouts[:20],
        "daily_macros": _decode_tool_result(macros_res),
    }


def _analyze_progress_format(data: dict[str, Any]) -> str:
    lines = ["RECENT WORKOUTS (last 20, most recent first):"]
    for w in data.get("workouts", []):
        lines.append(
            f"- {w.get('id', '?')} · "
            f"{w.get('performed_at', '?')} · "
            f"{len(w.get('exercises') or [])} exercise(s)"
        )
    lines.append("")
    lines.append("DAILY MACROS (last 30 days; date · kcal · P/F/C):")
    for d in data.get("daily_macros", []):
        lines.append(
            f"- {d.get('date', '?')} · "
            f"{d.get('calories', 0)} kcal · "
            f"{d.get('protein_g', 0)}/{d.get('fat_g', 0)}/{d.get('carbs_g', 0)}"
        )
    return "\n".join(lines)


_ANALYZE_PROGRESS_RULES = """\
The user wants analysis or planning advice. You already have a recent \
window of workouts and macros below. Favor citing specifics from this \
data over calling more tools; only fetch more if the user asks about \
a time range outside the included window.\
"""


_register(IntentSpec(
    intent="analyze_progress",
    rules=_ANALYZE_PROGRESS_RULES,
    prefetch=_analyze_progress_prefetch,
    format=_analyze_progress_format,
))


class IntentRegistry:
    """Static facade — no instances. Each public method is a classmethod
    so the harness can call it as `IntentRegistry.run(...)`.
    """

    @classmethod
    def known(cls) -> Iterable[str]:
        return KNOWN_INTENTS

    @classmethod
    async def run(cls, intent: str, session: Any) -> tuple[str, str, bool]:
        """Run the intent's prefetch and return (rules_block, data_block, failed).

        `failed` is True when prefetch (or formatting) raised. The harness
        reads this to set `telemetry.intent_prefetch_failed` so the
        dashboard surfaces the failure rate. The data block is empty on
        failure; rules are preserved so the model still gets the
        behavioral nudge even when the data is missing.

        This method itself never raises — caller doesn't need a try/except.
        """
        spec = _SPECS.get(intent)
        if spec is None:
            return "", "", False
        try:
            data = await spec.prefetch(session)
        except Exception:  # noqa: BLE001 — broad by design
            log.exception("intent prefetch failed: intent=%s", intent)
            return spec.rules, "", True
        try:
            return spec.rules, spec.format(data), False
        except Exception:
            log.exception("intent format failed: intent=%s", intent)
            return spec.rules, "", True
