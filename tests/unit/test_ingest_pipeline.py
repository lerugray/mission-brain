"""Tests for `mission_brain.ingest.pipeline` using a canned synthesis client."""

from __future__ import annotations

from pathlib import Path

import pytest

from mission_brain.ingest.citation_floor import CitationFloorViolation
from mission_brain.ingest.pipeline import MAX_RETRIES, ingest_source
from mission_brain.loaders.base import SourceDocument
from mission_brain.schema.citation import LineRangeLocator


class _CannedClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    def generate(self, prompt: str, **_: object) -> str:
        self.calls.append(prompt)
        if not self._responses:
            raise RuntimeError("canned client exhausted")
        return self._responses.pop(0)


def _source_doc() -> SourceDocument:
    return SourceDocument(
        title="Asymmetric Objectives",
        body=(
            "Ray describes asymmetric objectives as a third way\n"
            "between victory conditions and pure scoring.\n"
            "He returns to this in later design notes.\n"
        ),
        frontmatter={},
        line_count=3,
        content_hash="deadbeef",
        source_path=Path("journal-2016-09.md"),
        source_type="plain_markdown",
        loader_version="1.0",
    )


VALID_MARKDOWN = (
    "# Asymmetric Objectives\n\n"
    "Ray frames asymmetric objectives as a third way between\n"
    "victory conditions and pure scoring "
    "[ref:journal-2016-09:lines=1-2].\n\n"
    "He returns to the framing in later design notes "
    "[ref:journal-2016-09:lines=3-3].\n"
)

INVALID_MARKDOWN = "# Asymmetric Objectives\n\nThis paragraph has no citation at all.\n"


def test_valid_output_parses_to_wiki_page() -> None:
    client = _CannedClient([VALID_MARKDOWN])
    page = ingest_source(_source_doc(), None, client)  # type: ignore[arg-type]

    assert page.title == "Asymmetric Objectives"
    assert len(page.paragraphs) == 2
    assert all(len(p.citations) == 1 for p in page.paragraphs)
    first = page.paragraphs[0].citations[0]
    assert first.source_id == "journal-2016-09"
    assert isinstance(first.locator, LineRangeLocator)
    assert (first.locator.start, first.locator.end) == (1, 2)
    assert "[ref:" not in page.paragraphs[0].text
    assert "third way" in first.excerpt
    assert page.source_ids == ["journal-2016-09"]


def test_retry_succeeds_after_one_violation() -> None:
    client = _CannedClient([INVALID_MARKDOWN, VALID_MARKDOWN])
    page = ingest_source(_source_doc(), None, client)  # type: ignore[arg-type]
    assert len(client.calls) == 2
    assert "CORRECTION" in client.calls[1]
    assert page.title == "Asymmetric Objectives"


def test_retry_exhaustion_raises() -> None:
    client = _CannedClient([INVALID_MARKDOWN] * (MAX_RETRIES + 1))
    with pytest.raises(CitationFloorViolation):
        ingest_source(_source_doc(), None, client)  # type: ignore[arg-type]
    assert len(client.calls) == MAX_RETRIES + 1


