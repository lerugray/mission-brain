"""Tests for :class:`MissionBulletLoader` (rb-036).

Covers the raw entry layer (``corpus/journal/entries/YYYY/MM/DD.md``).
The refined and chat variants live under separate suffixes
(``DD.refined.md``, ``DD.claude.md``) and are out of scope for rb-036
— they're picked up by rb-037 / rb-039 respectively.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mission_brain.loaders.journal import MissionBulletLoader


# ---------------------------------------------------------------------------
# Fixtures


@pytest.fixture
def corpus_with_journal(tmp_path: Path) -> Path:
    """Build a corpus_root with a sample journal entry tree.

    Layout (rb-036 + rb-037):
      <corpus>/journal/entries/2026/04/21.md           (raw — picked up)
      <corpus>/journal/entries/2026/04/21.refined.md   (refined — picked up)
      <corpus>/journal/entries/2026/04/22.md           (raw — picked up)
      <corpus>/journal/entries/2026/04/22.refined.md   (refined — picked up)
      <corpus>/journal/entries/2026/04/22.claude.md    (chat — skipped)
      <corpus>/journal/entries/2026/04/monthly.md      (monthly — picked up)
      <corpus>/journal/entries/2026/04/notes.md        (non-day stem — skipped)
      <corpus>/journal/entries/2026/05/01.md           (raw — picked up)
    """
    entries = tmp_path / "journal" / "entries"
    apr = entries / "2026" / "04"
    may = entries / "2026" / "05"
    apr.mkdir(parents=True)
    may.mkdir(parents=True)

    (apr / "21.md").write_text(
        "Today I tested the journal loader. Things work.\n", encoding="utf-8"
    )
    (apr / "21.refined.md").write_text(
        "Refined entry for 04-21 — voice-bearing edit-toward layer.\n",
        encoding="utf-8",
    )
    (apr / "22.md").write_text(
        "Continued testing. Refined the loader signature.\n",
        encoding="utf-8",
    )
    (apr / "22.refined.md").write_text(
        "Refined version of 04-22 entry — voice signal.\n",
        encoding="utf-8",
    )
    (apr / "22.claude.md").write_text(
        "Chat transcript — deferred per rb-039.\n", encoding="utf-8"
    )
    (apr / "monthly.md").write_text(
        "Monthly aggregate for April — rb-037.\n", encoding="utf-8"
    )
    (apr / "notes.md").write_text(
        "Random non-entry markdown — must skip.\n", encoding="utf-8"
    )
    (may / "01.md").write_text(
        "May entry. Different month.\n", encoding="utf-8"
    )
    return tmp_path


# ---------------------------------------------------------------------------
# discover() filters non-raw variants


def test_discover_yields_raw_and_monthly_only(corpus_with_journal: Path) -> None:
    """Refined entries are AI-polish, not the user's voice — explicitly excluded.

    Mission-bullet's DD.refined.md files are produced by an LLM
    polishing the raw entry; ingesting them pollutes voice retrieval
    with two-layers-of-LLM-removal content. Loader skips them.
    """
    loader = MissionBulletLoader()
    paths = list(loader.discover(corpus_with_journal))
    names = sorted(p.name for p in paths)
    assert names == sorted(
        ["01.md", "21.md", "22.md", "monthly.md"]
    )


def test_discover_returns_sorted(corpus_with_journal: Path) -> None:
    loader = MissionBulletLoader()
    paths = list(loader.discover(corpus_with_journal))
    assert paths == sorted(paths)


def test_discover_skips_refined_chat_and_excalidraw(corpus_with_journal: Path) -> None:
    loader = MissionBulletLoader()
    paths = list(loader.discover(corpus_with_journal))
    for p in paths:
        assert not p.name.endswith(".refined.md")
        assert not p.name.endswith(".claude.md")
        assert not p.name.endswith(".excalidraw")


def test_discover_skips_arbitrary_md_files(corpus_with_journal: Path) -> None:
    loader = MissionBulletLoader()
    paths = list(loader.discover(corpus_with_journal))
    names = [p.name for p in paths]
    assert "notes.md" not in names


def test_discover_returns_empty_when_no_journal_dir(tmp_path: Path) -> None:
    loader = MissionBulletLoader()
    assert list(loader.discover(tmp_path)) == []


# ---------------------------------------------------------------------------
# load() emits SourceDocument with journal-private provenance + date title


def test_load_default_provenance_is_journal_private(corpus_with_journal: Path) -> None:
    loader = MissionBulletLoader()
    paths = list(loader.discover(corpus_with_journal))
    doc = loader.load(paths[0])
    assert doc.provenance == "journal-private"


def test_load_provenance_override_respected(corpus_with_journal: Path) -> None:
    loader = MissionBulletLoader(provenance="primary")
    paths = list(loader.discover(corpus_with_journal))
    doc = loader.load(paths[0])
    assert doc.provenance == "primary"


def test_load_title_from_path_when_no_frontmatter(corpus_with_journal: Path) -> None:
    loader = MissionBulletLoader()
    paths = list(loader.discover(corpus_with_journal))
    titles = [loader.load(p).title for p in paths]
    # Raw layer
    assert "Journal 2026-04-21" in titles
    assert "Journal 2026-04-22" in titles
    assert "Journal 2026-05-01" in titles
    # Monthly layer (rb-037 partial)
    assert "Journal 2026-04 (monthly)" in titles
    # Refined layer is not loaded — confirm no refined-suffixed titles
    assert not any("(refined)" in t for t in titles)


def test_monthly_title_format(corpus_with_journal: Path) -> None:
    loader = MissionBulletLoader()
    monthly_path = (
        corpus_with_journal / "journal" / "entries" / "2026" / "04" / "monthly.md"
    )
    doc = loader.load(monthly_path)
    assert doc.title == "Journal 2026-04 (monthly)"


def test_load_title_from_frontmatter_takes_precedence(tmp_path: Path) -> None:
    apr = tmp_path / "journal" / "entries" / "2026" / "04"
    apr.mkdir(parents=True)
    (apr / "21.md").write_text(
        "---\ntitle: Custom Title\n---\nBody.\n", encoding="utf-8"
    )
    loader = MissionBulletLoader()
    paths = list(loader.discover(tmp_path))
    doc = loader.load(paths[0])
    assert doc.title == "Custom Title"


def test_load_body_strips_frontmatter(tmp_path: Path) -> None:
    apr = tmp_path / "journal" / "entries" / "2026" / "04"
    apr.mkdir(parents=True)
    (apr / "21.md").write_text(
        "---\ntitle: X\n---\n\nBody content.\n", encoding="utf-8"
    )
    loader = MissionBulletLoader()
    paths = list(loader.discover(tmp_path))
    doc = loader.load(paths[0])
    assert "title:" not in doc.body
    assert "Body content." in doc.body


def test_load_content_hash_stable(corpus_with_journal: Path) -> None:
    loader = MissionBulletLoader()
    paths = list(loader.discover(corpus_with_journal))
    doc1 = loader.load(paths[0])
    doc2 = loader.load(paths[0])
    assert doc1.content_hash == doc2.content_hash


def test_load_source_type_and_version(corpus_with_journal: Path) -> None:
    loader = MissionBulletLoader()
    paths = list(loader.discover(corpus_with_journal))
    doc = loader.load(paths[0])
    assert doc.source_type == "mission_bullet_journal"
    assert doc.loader_version == "1.0"


# ---------------------------------------------------------------------------
# locator_for() — line ranges (mirrors plain_markdown loader)


def test_locator_for_line_range() -> None:
    loader = MissionBulletLoader()
    assert loader.locator_for((1, 5)) == "lines=1-5"
    assert loader.locator_for((10, 10)) == "lines=10-10"


def test_locator_for_rejects_invalid_spans() -> None:
    loader = MissionBulletLoader()
    with pytest.raises(ValueError):
        loader.locator_for((0, 5))
    with pytest.raises(ValueError):
        loader.locator_for((5, 3))
