"""``lines=`` locator realignment and ingest wiring."""

from __future__ import annotations

import re
from pathlib import Path

from mission_brain.ingest.citation_alignment import realign_line_locators_for_document
from mission_brain.ingest.claude_cli_client import ClaudeCLIClient
from mission_brain.ingest.pipeline import ingest_source
from mission_brain.loaders.base import SourceDocument
from mission_brain.schema.citation import LineRangeLocator


def _doc() -> SourceDocument:
    return SourceDocument(
        title="Asymmetric Objectives",
        body=(
            "The author describes asymmetric objectives as a third way\n"
            "between victory conditions and pure scoring.\n"
            "The notes return to this in a later design sketch.\n"
        ),
        frontmatter={},
        line_count=3,
        content_hash="deadbeef",
        source_path=Path("design-notes.md"),
        source_type="plain_markdown",
        loader_version="1.0",
        source_id="design-notes",
    )


def test_realign_corrects_misanchored_line_ref() -> None:
    """Model cites line 1 for a paragraph that matches line 3."""
    bad = (
        "# Asymmetric Objectives\n\n"
        "The author frames asymmetric objectives as a third way between\n"
        "victory conditions and pure scoring "
        "[ref:design-notes:lines=1-2].\n\n"
        "The notes return to the framing in a later design sketch "
        "[ref:design-notes:lines=1-1].\n"
    )
    fixed = realign_line_locators_for_document(bad, _doc())
    assert "lines=1-1]" not in fixed
    assert "lines=3-3]" in fixed
    assert "lines=1-2]" in fixed


def test_realign_prefers_thesis_bullet_over_trailing_yaml_metadata() -> None:
    """Metadata that echoes topic words should not beat the supporting line."""
    body = (
        "- RARE: unique token ZX9 in thesis body foxes bridges mapmaking arc.\n"
        "\n"
        "---\n"
        "date: 2026-01-01\n"
        "status: open\n"
        'tags_discovered: ["generic", "other"]\n'
        "---\n"
    )
    doc = SourceDocument(
        title="metadata bleed",
        body=body,
        frontmatter={},
        line_count=body.count("\n") + 1,
        content_hash="cafebabe",
        source_path=Path("metadata-bleed.md"),
        source_type="plain_markdown",
        loader_version="1.0",
        source_id="metadata-bleed",
    )
    bad = (
        "# T\n\n"
        "Synthesis of the RARE arc, ZX9, and foxes bridges context "
        "[ref:metadata-bleed:lines=3-5].\n"
    )
    fixed = realign_line_locators_for_document(bad, doc)
    assert "lines=1-1]" in fixed, fixed
    assert "lines=3-5" not in fixed


def test_realign_extends_compound_todo_to_cover_distinctive_trailing_bullet() -> None:
    """A wide ref can match a list yet miss a distinctive later clause."""
    body = (
        "## Migrated\n"
        "\n"
        "- [ ] Water the plants. (migrated) <!-- x -->\n"
        "- [ ] Fold laundry. (migrated) <!-- x -->\n"
        "- [ ] Reply to project emails. (migrated) <!-- x -->\n"
        "\n"
        "- 19:29 log line filler.\n"
        "- [ ] Clean desk sink and UNIQUEBATH9. (migrated) <!-- x -->\n"
    )
    doc = SourceDocument(
        title="surface words",
        body=body,
        frontmatter={},
        line_count=body.count("\n") + 1,
        content_hash="cafebabe",
        source_path=Path("surface-words.md"),
        source_type="plain_markdown",
        loader_version="1.0",
        source_id="surface-words",
    )
    bad = (
        "# T\n\n"
        "Open items: plants, laundry, project emails, and clean the desk with "
        "UNIQUEBATH9. [ref:surface-words:lines=3-5]\n"
    )
    fixed = realign_line_locators_for_document(bad, doc)
    assert "lines=3-5" not in fixed
    m = re.search(r"lines=(\d+)-(\d+)", fixed)
    assert m is not None, fixed
    a, b = int(m.group(1)), int(m.group(2))
    assert a <= 3 and b >= 8


def test_realign_tightens_span_when_keepworthy_but_wide() -> None:
    """Prefer a tighter keep-worthy window when one scores better."""
    fixtures = Path(__file__).resolve().parents[1] / "fixtures" / "rb-052-minimal-span"
    body = (fixtures / "source.md").read_text(encoding="utf-8")
    wiki = (fixtures / "wiki.md").read_text(encoding="utf-8")
    doc = SourceDocument(
        title="minimal span",
        body=body,
        frontmatter={},
        line_count=body.count("\n") + 1,
        content_hash="rb052",
        source_path=Path("rb052-min-span.md"),
        source_type="plain_markdown",
        loader_version="1.0",
        source_id="rb052-min-span",
    )
    fixed = realign_line_locators_for_document(wiki, doc)
    assert "lines=1-3" not in fixed, fixed
    assert "lines=1-1" in fixed, fixed


def test_realign_is_noop_when_excerpt_matches_paragraph() -> None:
    good = (
        "# T\n\n"
        "The author frames asymmetric objectives as a third way between\n"
        "victory conditions and pure scoring "
        "[ref:design-notes:lines=1-2].\n\n"
        "The notes return to the framing in a later design sketch "
        "[ref:design-notes:lines=3-3].\n"
    )
    out = realign_line_locators_for_document(good, _doc())
    assert out == good


class _StaticClient(ClaudeCLIClient):
    def __init__(self, canned_markdown: str) -> None:
        self._canned = canned_markdown

    def generate(self, prompt: str) -> str:  # type: ignore[override]
        return self._canned


def test_ingest_source_applies_realign_before_parse() -> None:
    """Canned output with a wrong second ref is corrected, then parsed."""
    misaligned = (
        "# Asymmetric Objectives\n\n"
        "The author frames asymmetric objectives as a third way between\n"
        "victory conditions and pure scoring "
        "[ref:design-notes:lines=1-2].\n\n"
        "The notes return to the framing in a later design sketch "
        "[ref:design-notes:lines=1-1].\n"
    )
    page = ingest_source(_doc(), None, _StaticClient(misaligned))  # type: ignore[arg-type]
    assert len(page.paragraphs) == 2
    c1, c2 = page.paragraphs[0].citations[0], page.paragraphs[1].citations[0]
    assert (c1.locator.start, c1.locator.end) == (1, 2)
    assert (c2.locator.start, c2.locator.end) == (3, 3)
    assert isinstance(c2.locator, LineRangeLocator)
    assert "later" in c2.excerpt.lower() or "return" in c2.excerpt.lower()
