"""Unit tests for the sentence + Markdown helpers used by the
voice-streaming pipeline. These are pure functions; the orchestration
that calls them is tested separately via the /chat e2e path.
"""

from __future__ import annotations

from prog_strength_agent.voice_stream import (
    pop_complete_sentences,
    strip_markdown,
)


# ---- pop_complete_sentences ----------------------------------------


def test_pop_no_terminator_returns_no_sentences():
    """A buffer with no sentence-ending punctuation should yield no
    completed sentences — the caller hasn't seen enough text yet.
    """
    sentences, rest = pop_complete_sentences("Hello world")
    assert sentences == []
    assert rest == "Hello world"


def test_pop_terminator_without_trailing_whitespace_stays_incomplete():
    """A period at end-of-buffer isn't a confirmed boundary — the
    next delta might extend the sentence. We only commit once a
    whitespace separator follows OR the caller flushes at stream end.
    """
    sentences, rest = pop_complete_sentences("Hello world.")
    assert sentences == []
    assert rest == "Hello world."


def test_pop_single_complete_sentence():
    sentences, rest = pop_complete_sentences("Hello world. ")
    assert sentences == ["Hello world."]
    assert rest == ""


def test_pop_multiple_sentences_in_one_buffer():
    sentences, rest = pop_complete_sentences(
        "Hello. World! Trailing fragment"
    )
    assert sentences == ["Hello.", "World!"]
    assert rest == "Trailing fragment"


def test_pop_preserves_terminator_chars():
    """Sentence-final punctuation rides with the sentence, not the
    remainder. TTS keys off the punctuation for intonation, so we
    can't strip it.
    """
    sentences, _ = pop_complete_sentences("Wait! Really? Yes. Done.")
    # The trailing "Done." has no whitespace after, so it stays in the
    # remainder. Three complete sentences plus the trailing fragment.
    assert sentences == ["Wait!", "Really?", "Yes."]


def test_pop_multiple_punctuation_kept_intact():
    """Triple-dot ellipsis and "?!" combinations are common in
    conversational text and shouldn't get half-clipped.
    """
    sentences, rest = pop_complete_sentences("Really?! Yes... maybe.")
    assert sentences == ["Really?!", "Yes..."]
    assert rest == "maybe."


def test_pop_abbreviation_doesnt_end_sentence():
    """A trailing 'Dr.' or 'e.g.' looks like a sentence end but is
    almost always followed by more text. The allowlist suppresses
    the false split.
    """
    sentences, rest = pop_complete_sentences(
        "Visit Dr. Smith tomorrow. Got it"
    )
    assert sentences == ["Visit Dr. Smith tomorrow."]
    assert rest == "Got it"


def test_pop_abbreviation_case_insensitive():
    """The allowlist matches case-insensitively so 'DR.' or 'i.e.'
    or any variant the model produces doesn't slip through.
    """
    sentences, rest = pop_complete_sentences("So e.g. carrots. Done")
    assert sentences == ["So e.g. carrots."]
    assert rest == "Done"


def test_pop_empty_string():
    sentences, rest = pop_complete_sentences("")
    assert sentences == []
    assert rest == ""


def test_pop_strips_trailing_whitespace_from_sentence():
    """The sentence string handed to TTS shouldn't have trailing
    whitespace that came from the separator we matched on.
    """
    sentences, _ = pop_complete_sentences("Hello.   Next")
    assert sentences == ["Hello."]


# ---- strip_markdown -----------------------------------------------


def test_strip_keeps_plain_text_alone():
    assert strip_markdown("Hello world.") == "Hello world."


def test_strip_bold_double_asterisk():
    assert strip_markdown("You got this **bro**!") == "You got this bro!"


def test_strip_bold_double_underscore():
    assert strip_markdown("This __is__ great") == "This is great"


def test_strip_italic_single_asterisk():
    assert strip_markdown("Really *crushing* it") == "Really crushing it"


def test_strip_italic_single_underscore():
    assert strip_markdown("Some _emphasis_ here") == "Some emphasis here"


def test_strip_strikethrough():
    assert strip_markdown("~~old~~ new") == "old new"


def test_strip_inline_code_keeps_content():
    """Inline code's content is usually a function name or value the
    user wants to hear, just without the backticks.
    """
    assert strip_markdown("Call `list_workouts` first") == "Call list_workouts first"


def test_strip_fenced_code_removed_entirely():
    """A code block in TTS would be unreadable character-by-character.
    Strip the whole block.
    """
    text = "Try this:\n```python\nprint('hi')\n```\nDone."
    out = strip_markdown(text)
    assert "print" not in out
    assert "```" not in out
    assert "Try this:" in out
    assert "Done." in out


def test_strip_link_keeps_display_text():
    assert (
        strip_markdown("See [the docs](https://example.com) for more")
        == "See the docs for more"
    )


def test_strip_list_markers():
    text = "- One\n- Two\n- Three"
    assert strip_markdown(text) == "One\nTwo\nThree"


def test_strip_heading_markers():
    assert strip_markdown("# Title\nBody text") == "Title\nBody text"


def test_strip_blockquote():
    assert strip_markdown("> Quoted thing") == "Quoted thing"


def test_strip_combined_markup():
    """The model often emits text with multiple markup styles in the
    same response — bold inside a list, code inside a sentence, etc.
    All of them should come out cleanly.
    """
    text = "**Today's workout**: Try `bench-press` for *3 sets*."
    assert (
        strip_markdown(text)
        == "Today's workout: Try bench-press for 3 sets."
    )
