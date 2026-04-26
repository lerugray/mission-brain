"""Tests for the provenance field round-trip (rayb-012).

Verifies that a `SourceDocument` with `provenance="derived"` carries
its tag through the ingest pipeline and out to the wiki store's
frontmatter, so queries can distinguish primary material (Ray-authored
or mechanical extractions from Ray's own work) from derived material
(content synthesized by another tool before reaching mission-brain).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mission_brain.ingest.claude_cli_client import ClaudeCLIClient
from mission_brain.ingest.pipeline import ingest_source
from mission_brain.loaders.base import SourceDocument
from mission_brain.schema import (
    BarRangeLocator,
    Citation,
    LineRangeLocator,
    Paragraph,
    SourceId,
    TimeRangeLocator,
    WikiPage,
)
from mission_brain.wiki import read_page, write_page


class _StaticClient(ClaudeCLIClient):
    """Test double that returns a canned markdown response instead of
    invoking the `claude` CLI. The response is well-formed enough to
    pass the citation-floor check — every paragraph carries a ref.
    """

    def __init__(self, canned_markdown: str) -> None:
        self._canned = canned_markdown

    def generate(self, prompt: str) -> str:  # type: ignore[override]
        return self._canned


def _source_doc(provenance: str, source_id: str = "fake-source") -> SourceDocument:
    return SourceDocument(
        title="Fake Source",
        body="line 1\nline 2\nline 3\nline 4\nline 5",
        frontmatter={},
        line_count=5,
        content_hash="a" * 64,
        source_path=Path("/tmp/fake.md"),
        source_type="plain_markdown",
        loader_version="0.1.0",
        source_id=source_id,
        provenance=provenance,  # type: ignore[arg-type]
    )


def test_sourcedocument_defaults_provenance_primary() -> None:
    """Existing call sites that don't pass provenance still get 'primary'."""
    doc = SourceDocument(
        title="x",
        body="y [ref:fake:lines=1-1]",
        frontmatter={},
        line_count=1,
        content_hash="a" * 64,
        source_path=Path("/tmp/x.md"),
    )
    assert doc.provenance == "primary"


def test_wikipage_defaults_provenance_primary() -> None:
    """Existing WikiPage construction without provenance defaults to 'primary'."""
    page = WikiPage(
        title="x",
        source_ids=[SourceId("fake-source")],
        paragraphs=[
            Paragraph(
                text="claim",
                citations=[
                    Citation(
                        source_id=SourceId("fake-source"),
                        locator=LineRangeLocator(start=1, end=1),
                        excerpt="line 1",
                    )
                ],
            ),
        ],
    )
    assert page.provenance == "primary"


def test_wikipage_rejects_unknown_provenance() -> None:
    """The Literal type refuses anything except 'primary'/'derived'."""
    with pytest.raises(Exception):  # pydantic ValidationError
        WikiPage(
            title="x",
            source_ids=[SourceId("fake")],
            paragraphs=[],
            provenance="speculative",  # type: ignore[arg-type]
        )


def test_ingest_derived_passes_through_to_wikipage() -> None:
    """A derived-provenance SourceDocument produces a derived WikiPage."""
    doc = _source_doc(provenance="derived")
    markdown = (
        "# Fake Source\n\n"
        "Synthesized claim. [ref:fake-source:lines=1-2]\n"
    )
    client = _StaticClient(markdown)
    page = ingest_source(doc, prior_wiki=None, client=client)
    assert page.provenance == "derived"


def test_ingest_primary_passes_through_to_wikipage() -> None:
    doc = _source_doc(provenance="primary")
    markdown = (
        "# Fake Source\n\n"
        "Claim grounded in source. [ref:fake-source:lines=3-4]\n"
    )
    client = _StaticClient(markdown)
    page = ingest_source(doc, prior_wiki=None, client=client)
    assert page.provenance == "primary"


def test_write_and_read_preserves_derived_provenance(tmp_path: Path) -> None:
    """write_page → read_page round-trip retains the provenance tag."""
    page = WikiPage(
        title="Derived Page",
        source_ids=[SourceId("derived-source")],
        paragraphs=[
            Paragraph(
                text="Synthesized-of-synthesized claim.",
                citations=[
                    Citation(
                        source_id=SourceId("derived-source"),
                        locator=LineRangeLocator(start=1, end=2),
                        excerpt="derived excerpt",
                    )
                ],
            ),
        ],
        provenance="derived",
    )
    path = write_page(page, tmp_path)

    # The frontmatter YAML carries the tag.
    raw = path.read_text(encoding="utf-8")
    fm_end = raw.find("\n---\n", 4)
    fm = yaml.safe_load(raw[4:fm_end + 1])
    assert fm["provenance"] == "derived"

    # The round-trip read reconstructs the same provenance.
    restored = read_page("derived-page", tmp_path)
    assert restored is not None
    assert restored.provenance == "derived"


def test_read_legacy_page_without_provenance_defaults_to_primary(
    tmp_path: Path,
) -> None:
    """Pages written before rayb-012 lack `provenance:` in frontmatter.
    Reading such a page must default to 'primary' — no migration needed.
    """
    legacy_yaml = (
        "title: Legacy Page\n"
        "source_ids:\n"
        "- legacy-source\n"
        "paragraphs:\n"
        "- text: Old claim.\n"
        "  citations:\n"
        "  - source_id: legacy-source\n"
        "    locator: lines=1-1\n"
        "    excerpt: old excerpt\n"
    )
    body = (
        "# Legacy Page\n\n"
        "Old claim. [ref:legacy-source:lines=1-1]\n"
    )
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / "legacy-page.md").write_text(
        f"---\n{legacy_yaml}---\n\n{body}", encoding="utf-8", newline="\n"
    )

    restored = read_page("legacy-page", tmp_path)
    assert restored is not None
    assert restored.provenance == "primary"


# ---------------------------------------------------------------------------
# Music locators (co-shipped with provenance as part of rayb-012)


def test_time_range_locator_accepts_fractional_seconds() -> None:
    loc = TimeRangeLocator(start="00:00:01.000", end="00:00:02.500")
    assert loc.format() == "time=00:00:01.000-00:00:02.500"


def test_time_range_locator_rejects_whole_seconds() -> None:
    """`time=` requires the `.fff` suffix — whole-seconds go through
    the existing `timestamp=` locator instead.
    """
    with pytest.raises(Exception):
        TimeRangeLocator(start="00:00:01", end="00:00:02")


def test_bar_range_locator_round_trip() -> None:
    loc = BarRangeLocator(start=17, end=32)
    assert loc.format() == "bar=17-32"


def test_bar_range_locator_rejects_descending() -> None:
    with pytest.raises(Exception):
        BarRangeLocator(start=32, end=17)


def test_citation_with_time_locator_renders_correctly() -> None:
    cite = Citation(
        source_id=SourceId("song-001"),
        locator=TimeRangeLocator(start="00:01:30.000", end="00:02:00.000"),
        excerpt="chorus entry",
    )
    assert cite.render() == "[ref:song-001:time=00:01:30.000-00:02:00.000]"


def test_citation_with_bar_locator_renders_correctly() -> None:
    cite = Citation(
        source_id=SourceId("song-001"),
        locator=BarRangeLocator(start=9, end=16),
        excerpt="second verse",
    )
    assert cite.render() == "[ref:song-001:bar=9-16]"
