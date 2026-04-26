"""Tests for :class:`mission_brain.loaders.plain_markdown.PlainMarkdownLoader`.

Covers Phase 1 (rayb-003) acceptance shape:
  * discovery count over the mini-corpus fixture
  * title inference from frontmatter / H1 / filename stem
  * content-hash stability across repeated loads
  * locator rendering to the ``lines=N-M`` grammar
  * date inference from filename ``YYYY-MM`` pattern
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from mission_brain.loaders import PlainMarkdownLoader, SourceDocument

FIXTURES = (
    Path(__file__).resolve().parent.parent
    / "integration"
    / "fixtures"
    / "mini-corpus"
    / "plain-md"
)


@pytest.fixture
def loader() -> PlainMarkdownLoader:
    return PlainMarkdownLoader()


def test_loader_defaults_to_primary_provenance(loader: PlainMarkdownLoader) -> None:
    doc = _load_by_stem(loader, "fixture-with-frontmatter-title-2019-07")
    assert doc.provenance == "primary"


def test_loader_accepts_derived_provenance() -> None:
    """Loaders instantiated with ``provenance="derived"`` tag every emitted
    SourceDocument so downstream wiki pages inherit the tag.
    """
    derived_loader = PlainMarkdownLoader(provenance="derived")
    doc = _load_by_stem(derived_loader, "fixture-with-frontmatter-title-2019-07")
    assert doc.provenance == "derived"


def _load_by_stem(loader: PlainMarkdownLoader, stem: str) -> SourceDocument:
    path = FIXTURES / f"{stem}.md"
    return loader.load(path)


def test_discover_finds_three_fixtures(loader: PlainMarkdownLoader) -> None:
    found = list(loader.discover(FIXTURES))
    assert len(found) == 3
    assert all(p.suffix == ".md" for p in found)


def test_title_from_frontmatter_wins(loader: PlainMarkdownLoader) -> None:
    doc = _load_by_stem(loader, "fixture-with-frontmatter-title-2019-07")
    assert doc.title == "Synthetic Fixture — Frontmatter Title Wins"


def test_title_from_first_h1(loader: PlainMarkdownLoader) -> None:
    doc = _load_by_stem(loader, "fixture-h1-title-2020-03")
    assert doc.title == "H1 Title Wins When No Frontmatter Title"


def test_title_falls_back_to_stem(loader: PlainMarkdownLoader) -> None:
    doc = _load_by_stem(loader, "fixture-stem-title")
    assert doc.title == "fixture-stem-title"


def test_content_hash_is_sha256_of_raw_bytes(loader: PlainMarkdownLoader) -> None:
    path = FIXTURES / "fixture-stem-title.md"
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    doc = loader.load(path)
    assert doc.content_hash == expected


def test_content_hash_stable_across_loads(loader: PlainMarkdownLoader) -> None:
    path = FIXTURES / "fixture-h1-title-2020-03.md"
    assert loader.load(path).content_hash == loader.load(path).content_hash


def test_date_inferred_from_filename_when_frontmatter_lacks_it(
    loader: PlainMarkdownLoader,
) -> None:
    doc = _load_by_stem(loader, "fixture-h1-title-2020-03")
    assert doc.frontmatter.get("date") == "2020-03"


def test_frontmatter_date_not_overwritten(loader: PlainMarkdownLoader) -> None:
    doc = _load_by_stem(loader, "fixture-with-frontmatter-title-2019-07")
    assert str(doc.frontmatter["date"]) == "2019-07-04"


def test_line_count_matches_body(loader: PlainMarkdownLoader) -> None:
    doc = _load_by_stem(loader, "fixture-stem-title")
    assert doc.line_count == len(doc.body.splitlines())


def test_locator_for_renders_line_range(loader: PlainMarkdownLoader) -> None:
    assert loader.locator_for((3, 12)) == "lines=3-12"


def test_locator_for_rejects_invalid_span(loader: PlainMarkdownLoader) -> None:
    with pytest.raises(ValueError):
        loader.locator_for((0, 5))
    with pytest.raises(ValueError):
        loader.locator_for((10, 5))


def test_loader_identity_fields(loader: PlainMarkdownLoader) -> None:
    assert loader.source_type == "plain_markdown"
    assert ".md" in loader.supported_extensions
    assert loader.version
