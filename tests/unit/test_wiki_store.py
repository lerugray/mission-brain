"""Tests for the wiki store + index + log (rayb-009)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from mission_brain.ingest.citation_floor import check_citation_floor
from mission_brain.schema import (
    Citation,
    LineRangeLocator,
    PageLocator,
    ParaAnchorLocator,
    Paragraph,
    SourceId,
    TimestampLocator,
    WikiPage,
)
from mission_brain.wiki import append_log, read_page, update_index, write_page
from mission_brain.wiki.store import slugify

# ---------------------------------------------------------------------------
# Fixtures


def _cite(
    source_id: str = "plain-md-journal-2016-09",
    start: int = 45,
    end: int = 52,
    excerpt: str = "a third way between victory conditions and scoring",
) -> Citation:
    return Citation(
        source_id=SourceId(source_id),
        locator=LineRangeLocator(start=start, end=end),
        excerpt=excerpt,
    )


def _page_one() -> WikiPage:
    return WikiPage(
        title="Asymmetric Objectives",
        source_ids=[SourceId("plain-md-journal-2016-09")],
        paragraphs=[
            Paragraph(
                text="Ray frames asymmetric objectives as a third way.",
                citations=[_cite()],
            ),
            Paragraph(
                text="Later notes position it against Eurogame scoring.",
                citations=[
                    _cite("plain-md-notes-2018", 22, 30, "position against Eurogame"),
                ],
            ),
        ],
    )


def _page_two() -> WikiPage:
    return WikiPage(
        title="Citation Grammar",
        source_ids=[SourceId("plain-md-schema-notes")],
        paragraphs=[
            Paragraph(
                text="Every claim carries an anchor.",
                citations=[
                    Citation(
                        source_id=SourceId("plain-md-schema-notes"),
                        locator=PageLocator(page=3),
                        excerpt="anchor rule",
                    ),
                ],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# slugify


@pytest.mark.parametrize(
    "title,expected",
    [
        ("Asymmetric Objectives", "asymmetric-objectives"),
        ("Citation Grammar", "citation-grammar"),
        ("  Trim   Me  ", "trim-me"),
        ("Hello, World!", "hello-world"),
        ("!!!", "page"),
    ],
)
def test_slugify(title: str, expected: str) -> None:
    assert slugify(title) == expected


# ---------------------------------------------------------------------------
# write_page / read_page roundtrip


def test_write_page_returns_expected_path(tmp_path: Path) -> None:
    page = _page_one()
    out = write_page(page, tmp_path)
    assert out == tmp_path / "wiki" / "asymmetric-objectives.md"
    assert out.exists()


def test_roundtrip_identity(tmp_path: Path) -> None:
    page = _page_one()
    write_page(page, tmp_path)
    reloaded = read_page("asymmetric-objectives", tmp_path)
    assert reloaded == page


def test_roundtrip_all_locator_variants(tmp_path: Path) -> None:
    page = WikiPage(
        title="Locator Coverage",
        source_ids=[SourceId("src-x")],
        paragraphs=[
            Paragraph(
                text="Line range.",
                citations=[
                    Citation(
                        source_id=SourceId("src-x"),
                        locator=LineRangeLocator(start=1, end=3),
                        excerpt="ex-lines",
                    )
                ],
            ),
            Paragraph(
                text="Page.",
                citations=[
                    Citation(
                        source_id=SourceId("src-x"),
                        locator=PageLocator(page=7),
                        excerpt="ex-page",
                    )
                ],
            ),
            Paragraph(
                text="Timestamp.",
                citations=[
                    Citation(
                        source_id=SourceId("src-x"),
                        locator=TimestampLocator(start="00:01:30", end="00:02:45"),
                        excerpt="ex-ts",
                    )
                ],
            ),
            Paragraph(
                text="Para anchor.",
                citations=[
                    Citation(
                        source_id=SourceId("src-x"),
                        locator=ParaAnchorLocator(anchor="post-987654321"),
                        excerpt="ex-para",
                    )
                ],
            ),
        ],
    )
    write_page(page, tmp_path)
    reloaded = read_page("locator-coverage", tmp_path)
    assert reloaded == page


def test_write_twice_byte_identical(tmp_path: Path) -> None:
    page = _page_one()
    path_a = write_page(page, tmp_path)
    first = path_a.read_bytes()
    path_b = write_page(page, tmp_path)
    second = path_b.read_bytes()
    assert first == second


def test_rendered_body_satisfies_citation_floor(tmp_path: Path) -> None:
    page = _page_one()
    out = write_page(page, tmp_path)
    raw = out.read_text(encoding="utf-8")
    # Strip the YAML frontmatter — citation_floor sweeps the body only.
    _, _, body = raw.partition("\n---\n")
    check_citation_floor(body)


def test_read_missing_page_returns_none(tmp_path: Path) -> None:
    assert read_page("does-not-exist", tmp_path) is None


# ---------------------------------------------------------------------------
# update_index


def test_update_index_single_page(tmp_path: Path) -> None:
    update_index(tmp_path, _page_one())
    raw = (tmp_path / "index.md").read_text(encoding="utf-8")
    assert "# mission-brain wiki index" in raw
    assert "[Asymmetric Objectives](wiki/asymmetric-objectives.md)" in raw
    assert "plain-md-journal-2016-09" in raw


def test_update_index_two_pages_alphabetical(tmp_path: Path) -> None:
    update_index(tmp_path, _page_one())
    update_index(tmp_path, _page_two())
    raw = (tmp_path / "index.md").read_text(encoding="utf-8")
    a_pos = raw.index("Asymmetric Objectives")
    c_pos = raw.index("Citation Grammar")
    assert a_pos < c_pos
    # Both entries present.
    assert raw.count("(wiki/") == 2


def test_update_index_upsert_same_page(tmp_path: Path) -> None:
    update_index(tmp_path, _page_one())
    update_index(tmp_path, _page_one())
    raw = (tmp_path / "index.md").read_text(encoding="utf-8")
    assert raw.count("[Asymmetric Objectives]") == 1


# ---------------------------------------------------------------------------
# append_log


def test_append_log_two_actions(tmp_path: Path) -> None:
    append_log(tmp_path, "plain-md-journal-2016-09", "created")
    time.sleep(0.001)  # ensure microsecond timestamps differ
    append_log(tmp_path, "plain-md-notes-2018", "updated")
    raw = (tmp_path / "log.md").read_text(encoding="utf-8")
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    assert len(lines) == 2
    ts_a, action_a, sid_a = lines[0].split(" ", 2)
    ts_b, action_b, sid_b = lines[1].split(" ", 2)
    assert action_a == "created"
    assert action_b == "updated"
    assert sid_a == "plain-md-journal-2016-09"
    assert sid_b == "plain-md-notes-2018"
    assert ts_a != ts_b
