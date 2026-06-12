"""Unit tests for prompt.build_chat_system_prompt.

The helper composes a per-request system prompt by prepending a
"today is..." date line — the explicit grounding eliminates the
"agent hallucinates yesterday" failure mode. Tests cover the
timezone-handling branches: valid TZ, missing TZ (UTC fallback),
unrecognized TZ (UTC fallback + warning), and the day-boundary case
where local + UTC dates differ.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from prog_strength_agent.prompt import (
    SYSTEM_PROMPT,
    build_chat_system_prompt,
)


def test_includes_original_system_prompt():
    """The prefix is additive — the rest of SYSTEM_PROMPT must still be
    in the result so the model still gets the tools / conventions /
    tone sections it always has.
    """
    out = build_chat_system_prompt("UTC", now=datetime(2026, 5, 31, 12, 0, tzinfo=ZoneInfo("UTC")))
    assert SYSTEM_PROMPT in out
    # Prefix should sit at the very start, not somewhere in the middle.
    assert out.startswith("Today's date is")


def test_valid_timezone_uses_local_date():
    """A user in America/Denver at 11:00pm on May 31 (UTC = 05:00 on
    June 1) should see "Today is 2026-05-31" — local day, not UTC.
    Catches the original bug where /chat would think "yesterday" meant
    UTC-yesterday for users west of UTC.
    """
    # 2026-06-01 05:00 UTC = 2026-05-31 23:00 in America/Denver (MDT, UTC-6)
    moment = datetime(2026, 6, 1, 5, 0, tzinfo=ZoneInfo("UTC"))
    out = build_chat_system_prompt("America/Denver", now=moment)
    assert "Today's date is 2026-05-31" in out
    assert "(America/Denver)" in out


def test_missing_timezone_falls_back_to_utc():
    moment = datetime(2026, 5, 31, 12, 0, tzinfo=ZoneInfo("UTC"))
    out = build_chat_system_prompt(None, now=moment)
    assert "Today's date is 2026-05-31" in out
    assert "(UTC)" in out


def test_empty_string_timezone_falls_back_to_utc():
    moment = datetime(2026, 5, 31, 12, 0, tzinfo=ZoneInfo("UTC"))
    out = build_chat_system_prompt("", now=moment)
    assert "(UTC)" in out


def test_unknown_timezone_falls_back_to_utc_without_raising(caplog):
    """A garbage TZ name (e.g. a client passing the JS Date string by
    mistake) should not crash /chat — fall back to UTC and log a
    warning so we can spot a broken client without blocking the user.
    """
    moment = datetime(2026, 5, 31, 12, 0, tzinfo=ZoneInfo("UTC"))
    with caplog.at_level("WARNING"):
        out = build_chat_system_prompt("Not/A_Real_Zone", now=moment)
    assert "(UTC)" in out
    assert any(
        "unrecognized client_timezone" in rec.message for rec in caplog.records
    )


def test_weekday_in_prefix():
    """The weekday name is included so the model can answer "what day
    is it today?" without separately reasoning about dates.
    2026-05-31 is a Sunday.
    """
    moment = datetime(2026, 5, 31, 12, 0, tzinfo=ZoneInfo("UTC"))
    out = build_chat_system_prompt("UTC", now=moment)
    assert "Sunday" in out


def test_system_prompt_includes_single_serving_default():
    """Defense in depth: the assume-one-serving convention lands in
    the base prompt so the model has it even on a 'general' intent
    classification."""
    assert "Default to one serving" in SYSTEM_PROMPT
    assert "quantity=1" in SYSTEM_PROMPT


def test_compose_system_prompt_includes_all_sections_when_provided():
    from prog_strength_agent.prompt import compose_system_prompt
    out = compose_system_prompt(base="BASE", rules="RULES", data="DATA")
    assert "BASE" in out
    assert "RULES" in out
    assert "DATA" in out
    # Ordering matters: base first, then rules, then data.
    assert out.index("BASE") < out.index("RULES") < out.index("DATA")


def test_system_prompt_nudges_get_daily_macros_for_totals():
    """The daily-totals guidance must steer the model to get_daily_macros
    (computed server-side) and mention the date + timezone args it needs,
    rather than summing list_nutrition_log items itself.
    """
    out = build_chat_system_prompt("UTC", now=datetime(2026, 5, 31, 12, 0, tzinfo=ZoneInfo("UTC")))
    assert "get_daily_macros" in out
    assert "date" in out
    assert "timezone" in out
    # And it lives in the base prompt regardless of the date prefix.
    assert "get_daily_macros" in SYSTEM_PROMPT


def test_system_prompt_covers_custom_meal_logging():
    """The custom-meals guidance must steer the model through the
    pantry-first lookup, the log_custom_meal fallback, and the
    promote-to-pantry ask. Lives in the base prompt so the model has it
    regardless of intent classification."""
    assert "list_pantry_items" in SYSTEM_PROMPT
    assert "log_custom_meal" in SYSTEM_PROMPT
    assert "create_pantry_item" in SYSTEM_PROMPT
    # The exact save-to-pantry ask phrasing the agent should append.
    assert "save" in SYSTEM_PROMPT
    assert "to your pantry so I can find it next time" in SYSTEM_PROMPT
    # The promote-to-pantry call uses serving_unit "meal".
    assert "serving_unit" in SYSTEM_PROMPT
    assert '"meal"' in SYSTEM_PROMPT


def test_system_prompt_directs_lookup_before_estimating():
    """Custom-meal macros must come from lookup_food_nutrition when
    possible: lookup-first ordering, copy total_for_quantity verbatim,
    prefer warning-free candidates, cite the source in the reply, and
    only estimate (saying so) on a lookup miss. See
    prog-strength-docs/sows/custom-meal-macro-accuracy.md."""
    assert "lookup_food_nutrition" in SYSTEM_PROMPT
    assert "BEFORE estimating" in SYSTEM_PROMPT
    assert "total_for_quantity" in SYSTEM_PROMPT
    assert "never re-multiply" in SYSTEM_PROMPT
    assert "plausibility_warning" in SYSTEM_PROMPT
    # Cite-your-source instruction with the worked example.
    assert "cite the source" in SYSTEM_PROMPT
    # Estimation is the explicit fallback, flagged as such to the user.
    assert "your estimate" in SYSTEM_PROMPT


def test_custom_meal_logging_survives_date_prefix():
    """The custom-meals guidance is still present after the per-request
    date prefix is prepended (it lives in the base, not the prefix)."""
    out = build_chat_system_prompt(
        "UTC", now=datetime(2026, 5, 31, 12, 0, tzinfo=ZoneInfo("UTC"))
    )
    assert "list_pantry_items" in out
    assert "log_custom_meal" in out
    assert "lookup_food_nutrition" in out
    assert "create_pantry_item" in out
    assert "to your pantry so I can find it next time" in out
    assert "serving_unit" in out
    assert '"meal"' in out


def test_compose_system_prompt_omits_empty_sections():
    from prog_strength_agent.prompt import compose_system_prompt
    assert compose_system_prompt(base="BASE", rules="", data="") == "BASE"
    assert "RULES" not in compose_system_prompt(base="BASE", rules="", data="DATA")


_NOW = datetime(2026, 5, 31, 12, 0, tzinfo=ZoneInfo("UTC"))


def test_display_name_adds_identity_line():
    """A non-empty display_name produces an identity line so the agent
    calls the user by the name they picked."""
    out = build_chat_system_prompt("UTC", now=_NOW, display_name="Sam")
    assert "You are talking to Sam." in out
    # Identity sits after the date prefix but before the base prompt.
    assert out.index("Today's date is") < out.index("You are talking to Sam.")
    assert out.index("You are talking to Sam.") < out.index(SYSTEM_PROMPT)


def test_height_adds_cm_tall_clause():
    """height_cm present -> a 'cm tall' clause; 180.0 renders as '180'
    (no trailing .0) via the :g format."""
    out = build_chat_system_prompt("UTC", now=_NOW, display_name="Sam", height_cm=180.0)
    assert "They are 180 cm tall." in out
    assert "180.0" not in out
    assert "cm tall" in out


def test_non_integer_height_renders_decimals():
    """:g keeps meaningful fractional digits (e.g. 177.5)."""
    out = build_chat_system_prompt("UTC", now=_NOW, display_name="Sam", height_cm=177.5)
    assert "They are 177.5 cm tall." in out


def test_height_none_omits_height_clause():
    """No height -> identity line present but no 'cm tall' clause."""
    out = build_chat_system_prompt("UTC", now=_NOW, display_name="Sam")
    assert "You are talking to Sam." in out
    assert "cm tall" not in out


def test_no_display_name_omits_identity_line():
    """Name None/empty -> no identity line at all (and no height clause)."""
    out_none = build_chat_system_prompt("UTC", now=_NOW)
    out_empty = build_chat_system_prompt("UTC", now=_NOW, display_name="")
    for out in (out_none, out_empty):
        assert "You are talking to" not in out
        assert "cm tall" not in out
    # Height without a name is still suppressed (name always present in practice).
    out_height_only = build_chat_system_prompt("UTC", now=_NOW, height_cm=180.0)
    assert "You are talking to" not in out_height_only
    assert "cm tall" not in out_height_only


def test_height_guidance_warns_against_body_composition_inferences():
    """The prepended context tells the agent height is context-only and
    not to volunteer BMI / body-composition inferences."""
    out = build_chat_system_prompt("UTC", now=_NOW, display_name="Sam", height_cm=180.0)
    assert "conversational context only" in out
    assert "BMI" in out
    # Guidance lives in the prepended context, NOT the static base prompt.
    assert "conversational context only" not in SYSTEM_PROMPT


def test_identity_does_not_disturb_timezone_prefix():
    """The date prefix is unchanged when identity fields are supplied."""
    out = build_chat_system_prompt(
        "America/Denver",
        now=datetime(2026, 6, 1, 5, 0, tzinfo=ZoneInfo("UTC")),
        display_name="Sam",
        height_cm=180.0,
    )
    assert out.startswith("Today's date is 2026-05-31")
    assert "(America/Denver)" in out
