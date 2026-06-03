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


def test_compose_system_prompt_omits_empty_sections():
    from prog_strength_agent.prompt import compose_system_prompt
    assert compose_system_prompt(base="BASE", rules="", data="") == "BASE"
    assert "RULES" not in compose_system_prompt(base="BASE", rules="", data="DATA")
