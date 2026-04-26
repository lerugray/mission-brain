"""Tests for source-type dispatch in ``mission_brain.ingest.prompt`` (rayb-014-c).

* ``source_type='plain_markdown'`` (or empty) preserves the pre-refactor
  prompt text verbatim — asserted via a byte-identical golden string.
* ``source_type='music_data'`` per-song emits the five required section
  headings and the ``time=HH:MM:SS.fff-HH:MM:SS.fff`` locator grammar.
* ``source_type='music_data'`` catalog aggregate references ``para=<key>``
  locators instead of ``time=``.
"""

from __future__ import annotations

from pathlib import Path

from mission_brain.ingest.prompt import (
    build_ingest_prompt,
    build_music_ingest_prompt,
    build_plain_markdown_ingest_prompt,
)
from mission_brain.loaders.base import SourceDocument


def _plain_doc() -> SourceDocument:
    return SourceDocument(
        title="Asymmetric Objectives",
        body="line one\nline two\nline three\n",
        frontmatter={},
        line_count=3,
        content_hash="deadbeef",
        source_path=Path("journal-2016-09.md"),
        source_type="plain_markdown",
        loader_version="1.0",
        source_id="journal-2016-09",
    )


def _music_per_song_doc() -> SourceDocument:
    return SourceDocument(
        title="Synthetic Fake Song",
        body=(
            "title: Synthetic Fake Song\n"
            "tempo: 120.0\n"
            "notes:\n"
            "  00:00:00.000-00:00:00.500 C4 vel=80\n"
            "  00:00:00.500-00:00:01.000 E4 vel=75\n"
        ),
        frontmatter={"tempo": 120.0, "num_instruments": 2},
        line_count=5,
        content_hash="cafe" * 16,
        source_path=Path("music/per-song/fake-song.json"),
        source_type="music_data",
        loader_version="1.0",
        source_id="music-per-song-fake-song",
    )


def _music_catalog_doc() -> SourceDocument:
    return SourceDocument(
        title="Music Catalog Aggregate",
        body=(
            "title: Music Catalog Aggregate\n"
            "scale_fingerprint:\n"
            "  C major: 2\n"
            "chord_vocabulary:\n"
            "  D power chord (5th): 10\n"
        ),
        frontmatter={
            "catalog_keys": [
                "catalog_stats",
                "chord_vocabulary",
                "scale_fingerprint",
            ],
        },
        line_count=5,
        content_hash="beef" * 16,
        source_path=Path("music/catalog/catalog_analysis.json"),
        source_type="music_data",
        loader_version="1.0",
        source_id="music-catalog-analysis",
    )


# --- golden-string regression (plain_markdown is byte-identical) ---

_SECTION = "=" * 60

EXPECTED_PLAIN_PROMPT = (
    f"{_SECTION}\nSCHEMA (CLAUDE.md)\n{_SECTION}\nPRETEND_SCHEMA\n"
    + "\n"
    + f"{_SECTION}\nSOURCE DOCUMENT (source_id=journal-2016-09)\n"
    f"{_SECTION}\ntitle: Asymmetric Objectives\n\n"
    "line one\nline two\nline three\n\n"
    + "\n"
    + f"{_SECTION}\nTASK\n{_SECTION}\n"
    "Emit a wiki page in markdown. Every non-empty paragraph must\n"
    "carry at least one [ref:journal-2016-09:<locator>] marker — use the\n"
    "literal source_id 'journal-2016-09' shown above. Locator grammar:\n"
    "lines=N-M | page=N | timestamp=HH:MM:SS-HH:MM:SS | para=<anchor>.\n"
    "Headings and code blocks are exempt. Output raw markdown only —\n"
    "no preamble, no commentary, and DO NOT wrap the whole response\n"
    "in a ```markdown ... ``` code fence.\n"
)


def test_plain_markdown_prompt_is_byte_identical_golden() -> None:
    """Plain-markdown branch must match the pre-refactor output exactly."""
    got = build_ingest_prompt(_plain_doc(), None, "PRETEND_SCHEMA")
    assert got == EXPECTED_PLAIN_PROMPT


def test_plain_markdown_helper_matches_dispatch() -> None:
    direct = build_plain_markdown_ingest_prompt(_plain_doc(), None, "PRETEND_SCHEMA")
    dispatch = build_ingest_prompt(_plain_doc(), None, "PRETEND_SCHEMA")
    assert direct == dispatch


def test_empty_source_type_falls_back_to_plain_markdown() -> None:
    """Legacy call sites with empty ``source_type`` keep the old prompt."""
    doc = _plain_doc()
    doc.source_type = ""
    got = build_ingest_prompt(doc, None, "PRETEND_SCHEMA")
    assert got == EXPECTED_PLAIN_PROMPT


# --- music_data dispatch ---


def test_music_per_song_prompt_contains_five_section_headings() -> None:
    got = build_ingest_prompt(_music_per_song_doc(), None, "SCHEMA")
    for heading in (
        "## Tempo and meter",
        "## Harmonic language",
        "## Rhythm and groove",
        "## Structural arc",
        "## Notable motifs",
    ):
        assert heading in got, f"missing music section heading: {heading}"


def test_music_per_song_prompt_names_time_locator_grammar() -> None:
    got = build_ingest_prompt(_music_per_song_doc(), None, "SCHEMA")
    assert "time=HH:MM:SS.fff-HH:MM:SS.fff" in got
    # The instruction must name the concrete source_id so the LLM
    # can't substitute a different one.
    assert "music-per-song-fake-song" in got


def test_music_prompt_warns_against_bar_locators() -> None:
    got = build_ingest_prompt(_music_per_song_doc(), None, "SCHEMA")
    assert "bar=" in got
    assert "Phase 3" in got  # the deferral note


def test_music_catalog_prompt_uses_para_locators() -> None:
    got = build_ingest_prompt(_music_catalog_doc(), None, "SCHEMA")
    assert "para=<key>" in got
    # The five catalog top-level keys are named so the LLM knows the
    # anchor vocabulary without having to guess.
    for key in (
        "scale_fingerprint",
        "chord_vocabulary",
        "top_motifs",
        "register_usage",
        "rhythm_distribution",
    ):
        assert key in got


def test_music_catalog_prompt_does_not_instruct_time_locators() -> None:
    """Catalog has no time axis; the LLM must be told to use para= only."""
    got = build_ingest_prompt(_music_catalog_doc(), None, "SCHEMA")
    # The catalog branch's locator instruction line must not instruct
    # the time= format (the deferral note still legally mentions bar=).
    assert "time=HH:MM:SS.fff-HH:MM:SS.fff" not in got


def test_music_helper_dispatch_matches() -> None:
    doc = _music_per_song_doc()
    direct = build_music_ingest_prompt(doc, None, "SCHEMA")
    dispatch = build_ingest_prompt(doc, None, "SCHEMA")
    assert direct == dispatch


def test_music_prompt_includes_source_body() -> None:
    doc = _music_per_song_doc()
    got = build_ingest_prompt(doc, None, "SCHEMA")
    assert "00:00:00.000-00:00:00.500 C4" in got
