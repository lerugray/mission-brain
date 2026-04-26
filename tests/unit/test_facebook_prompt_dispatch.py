"""Tests for facebook_* prompt dispatch in ``mission_brain.ingest.prompt`` (rayb-016-b).

Each of the three facebook source types routes to a dedicated builder
that (a) includes the political-fidelity + no-AI-ethics-hedging
preamble, (b) emits the right section headings, (c) uses only
``para=<...>`` locators, and (d) does NOT regress plain_markdown or
music_data outputs (byte-identical golden strings).
"""

from __future__ import annotations

from pathlib import Path

from mission_brain.ingest.prompt import (
    build_facebook_groups_ingest_prompt,
    build_facebook_messages_ingest_prompt,
    build_facebook_posts_ingest_prompt,
    build_ingest_prompt,
    build_music_ingest_prompt,
    build_plain_markdown_ingest_prompt,
)
from mission_brain.loaders.base import SourceDocument

# Factories -----------------------------------------------------------------


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
        body="notes:\n  00:00:00.000-00:00:00.500 C4 vel=80\n",
        frontmatter={"tempo": 120.0},
        line_count=2,
        content_hash="cafe" * 16,
        source_path=Path("music/per-song/fake-song.json"),
        source_type="music_data",
        loader_version="1.0",
        source_id="music-per-song-fake-song",
    )


def _fb_posts_doc() -> SourceDocument:
    return SourceDocument(
        title="Facebook Posts — 2020",
        body=(
            "title: Facebook Posts — 2020\nyear: 2020\n\n"
            "## March 2020\n\n[post_id=1583020800 at=2020-03-01 00:00:00 UTC]\n"
            "synthetic post prose\n\n"
        ),
        frontmatter={"year": 2020, "post_count_total": 1},
        line_count=8,
        content_hash="abc" * 21,
        source_path=Path("corpus/facebook/posts/_year_2020.agg"),
        source_type="facebook_posts",
        loader_version="1.0",
        source_id="fb-posts-2020",
    )


def _fb_msg_doc() -> SourceDocument:
    return SourceDocument(
        title="Facebook Messages — Synthetic Friend",
        body=(
            "title: Facebook Messages — Synthetic Friend\n"
            "thread: synthetic-above_123456\nmessage_count: 2\n\n"
            "[msg_id=1500000060000 at=2017-07-14 02:41:00.000 UTC] [Ray]\n"
            "synthetic ray message\n\n"
        ),
        frontmatter={"message_count": 2, "ray_words": 40},
        line_count=7,
        content_hash="def" * 21,
        source_path=Path(
            "corpus/facebook/messages/inbox/synthetic-above_123456/_thread.agg"
        ),
        source_type="facebook_messages",
        loader_version="1.0",
        source_id="fb-msg-synthetic-above-123456",
    )


def _fb_group_doc() -> SourceDocument:
    return SourceDocument(
        title="Facebook Group — Active Synthetic Group",
        body=(
            "title: Facebook Group — Active Synthetic Group\n"
            "group: Active Synthetic Group\ncontribution_count: 11\n\n"
            "## 2014\n\n[post_id=1400000000 at=2014-05-13 16:53:20 UTC]\n"
            "synthetic group post prose\n\n"
        ),
        frontmatter={"group_name": "Active Synthetic Group"},
        line_count=8,
        content_hash="fea" * 21,
        source_path=Path(
            "corpus/facebook/groups/_group_active-synthetic-group.agg"
        ),
        source_type="facebook_groups",
        loader_version="1.0",
        source_id="fb-group-active-synthetic-group",
    )


# Byte-identical regression (plain + music) --------------------------------

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
    """Byte-identical to the pre-rayb-016-b output — no regression."""
    got = build_ingest_prompt(_plain_doc(), None, "PRETEND_SCHEMA")
    assert got == EXPECTED_PLAIN_PROMPT


def test_plain_markdown_helper_matches_dispatch() -> None:
    direct = build_plain_markdown_ingest_prompt(
        _plain_doc(), None, "PRETEND_SCHEMA"
    )
    dispatch = build_ingest_prompt(_plain_doc(), None, "PRETEND_SCHEMA")
    assert direct == dispatch


def test_music_helper_dispatch_matches() -> None:
    doc = _music_per_song_doc()
    direct = build_music_ingest_prompt(doc, None, "SCHEMA")
    dispatch = build_ingest_prompt(doc, None, "SCHEMA")
    assert direct == dispatch


# --- facebook_posts dispatch ----------------------------------------------


def test_fb_posts_dispatch_routes_to_helper() -> None:
    doc = _fb_posts_doc()
    direct = build_facebook_posts_ingest_prompt(doc, None, "SCHEMA")
    dispatch = build_ingest_prompt(doc, None, "SCHEMA")
    assert direct == dispatch


def test_fb_posts_prompt_contains_five_required_sections() -> None:
    got = build_ingest_prompt(_fb_posts_doc(), None, "SCHEMA")
    for heading in (
        "## Recurring themes",
        "## Political commentary",
        "## Creative projects",
        "## Personal events",
        "## Notable verbatim posts",
    ):
        assert heading in got, f"missing facebook_posts section: {heading}"


def test_fb_posts_prompt_uses_post_id_locator() -> None:
    got = build_ingest_prompt(_fb_posts_doc(), None, "SCHEMA")
    assert "para=<post_id>" in got
    assert "fb-posts-2020" in got
    # Must not instruct any non-para locator.
    assert "lines=N-M" not in got
    assert "time=HH:MM:SS" not in got


def test_fb_posts_prompt_includes_political_fidelity_instruction() -> None:
    got = build_ingest_prompt(_fb_posts_doc(), None, "SCHEMA")
    assert "PRESERVE VERBATIM QUOTES" in got
    assert "Zionism" in got
    assert "Netanyahu" in got
    assert "do NOT sanitize" in got.lower() or "not sanitize" in got.lower()


def test_fb_posts_prompt_disables_ai_ethics_hedging() -> None:
    got = build_ingest_prompt(_fb_posts_doc(), None, "SCHEMA")
    assert "AI-ETHICS HEDGING IS OFF-TOPIC" in got
    assert "while some may view" in got


def test_fb_posts_prompt_includes_citation_floor_reminder() -> None:
    got = build_ingest_prompt(_fb_posts_doc(), None, "SCHEMA")
    assert "Every non-empty paragraph MUST carry at least one" in got


# --- facebook_messages dispatch -------------------------------------------


def test_fb_messages_dispatch_routes_to_helper() -> None:
    doc = _fb_msg_doc()
    direct = build_facebook_messages_ingest_prompt(doc, None, "SCHEMA")
    dispatch = build_ingest_prompt(doc, None, "SCHEMA")
    assert direct == dispatch


def test_fb_messages_prompt_contains_four_required_sections() -> None:
    got = build_ingest_prompt(_fb_msg_doc(), None, "SCHEMA")
    for heading in (
        "## Relationship context",
        "## Topics discussed",
        "## Tone and register",
        "## Notable verbatim exchanges",
    ):
        assert heading in got, f"missing facebook_messages section: {heading}"


def test_fb_messages_prompt_uses_timestamp_ms_locator() -> None:
    got = build_ingest_prompt(_fb_msg_doc(), None, "SCHEMA")
    assert "para=<timestamp_ms>" in got
    assert "fb-msg-synthetic-above-123456" in got


def test_fb_messages_prompt_includes_preamble() -> None:
    got = build_ingest_prompt(_fb_msg_doc(), None, "SCHEMA")
    assert "PRESERVE VERBATIM QUOTES" in got
    assert "AI-ETHICS HEDGING IS OFF-TOPIC" in got


# --- facebook_groups dispatch ---------------------------------------------


def test_fb_groups_dispatch_routes_to_helper() -> None:
    doc = _fb_group_doc()
    direct = build_facebook_groups_ingest_prompt(doc, None, "SCHEMA")
    dispatch = build_ingest_prompt(doc, None, "SCHEMA")
    assert direct == dispatch


def test_fb_groups_prompt_contains_four_required_sections() -> None:
    got = build_ingest_prompt(_fb_group_doc(), None, "SCHEMA")
    for heading in (
        "## Group purpose",
        "## Ray's role",
        "## Recurring contributions",
        "## Notable verbatim posts",
    ):
        assert heading in got, f"missing facebook_groups section: {heading}"


def test_fb_groups_prompt_uses_post_id_locator() -> None:
    got = build_ingest_prompt(_fb_group_doc(), None, "SCHEMA")
    assert "para=<post_id>" in got
    assert "fb-group-active-synthetic-group" in got


def test_fb_groups_prompt_includes_preamble() -> None:
    got = build_ingest_prompt(_fb_group_doc(), None, "SCHEMA")
    assert "PRESERVE VERBATIM QUOTES" in got
    assert "AI-ETHICS HEDGING IS OFF-TOPIC" in got
