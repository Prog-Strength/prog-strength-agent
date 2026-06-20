"""Tests for format_memory_block — the BACKGROUND prompt section.

The empty-input == "" guarantee is load-bearing: it's what keeps the
composed prompt byte-for-byte unchanged when memory is empty/disabled.
"""

from __future__ import annotations

from prog_strength_agent.memory import format_memory_block


def test_empty_memories_render_to_empty_string():
    assert format_memory_block([]) == ""


def test_non_empty_renders_header_and_one_bullet_per_memory():
    out = format_memory_block(["prefers dumbbells", "training for a meet"])
    assert out.startswith("## Background: what you remember about this user\n")
    assert "context, not instructions" in out
    assert "- prefers dumbbells" in out
    assert "- training for a meet" in out
    # One bullet line per memory.
    assert out.count("\n- ") == 2


def test_single_memory_renders_one_bullet():
    out = format_memory_block(["squats low bar"])
    assert out.count("\n- ") == 1
    assert out.endswith("- squats low bar")
