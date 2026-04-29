"""``lines=`` locator realignment and ingest wiring."""

from __future__ import annotations

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
