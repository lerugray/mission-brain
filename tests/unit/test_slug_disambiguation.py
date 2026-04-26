"""Tests for slug-collision disambiguation in :func:`mission_brain.wiki.store.write_page`.

Background — the music corpus has 97 of 436 sources (~22%) whose
JSON ``title`` field collides with another song's title (mostly
``recovered -*.json`` files duplicating album-organized versions).
Without disambiguation, ingest order determines which song's
content survives at a given slug. The helper appends a 6-char
hash suffix derived from the first source_id when a different
logical page already lives at the title-derived slug.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from mission_brain.schema import (
    Citation,
    LineRangeLocator,
    Paragraph,
    SourceId,
    WikiPage,
)
from mission_brain.wiki import write_page
from mission_brain.wiki.store import _disambiguate_slug, read_page, slugify


def _make_page(title: str, source_id: str, body_text: str = "claim") -> WikiPage:
    return WikiPage(
        title=title,
        source_ids=[SourceId(source_id)],
        paragraphs=[
            Paragraph(
                text=body_text,
                citations=[
                    Citation(
                        source_id=SourceId(source_id),
                        locator=LineRangeLocator(start=1, end=2),
                        excerpt="x",
                    ),
                ],
            ),
        ],
    )


def test_no_collision_returns_base_slug(tmp_path: Path) -> None:
    page = _make_page("Some Song", "music-per-song-some-song")
    out = write_page(page, tmp_path)
    assert out.name == "some-song.md"


def test_same_source_rewrite_keeps_base_slug(tmp_path: Path) -> None:
    page_a = _make_page("X", "music-per-song-x", body_text="first")
    write_page(page_a, tmp_path)
    page_a2 = _make_page("X", "music-per-song-x", body_text="updated")
    out = write_page(page_a2, tmp_path)
    assert out.name == "x.md"
    on_disk = read_page("x", tmp_path)
    assert on_disk is not None
    assert on_disk.paragraphs[0].text == "updated"
    files = sorted(p.name for p in (tmp_path / "wiki").glob("*.md"))
    assert files == ["x.md"]


def test_different_source_collision_gets_suffix(tmp_path: Path) -> None:
    page_a = _make_page("Same Title", "music-per-song-album-a-track-01")
    page_b = _make_page("Same Title", "music-per-song-recovered-track-01")
    out_a = write_page(page_a, tmp_path)
    out_b = write_page(page_b, tmp_path)
    assert out_a.name == "same-title.md"
    expected_disc = hashlib.sha1(
        b"music-per-song-recovered-track-01"
    ).hexdigest()[:6]
    assert out_b.name == f"same-title-{expected_disc}.md"
    files = sorted(p.name for p in (tmp_path / "wiki").glob("*.md"))
    assert files == [f"same-title-{expected_disc}.md", "same-title.md"]


def test_three_way_collision_each_disambiguated(tmp_path: Path) -> None:
    page_a = _make_page("Same", "src-a")
    page_b = _make_page("Same", "src-b")
    page_c = _make_page("Same", "src-c")
    out_a = write_page(page_a, tmp_path)
    out_b = write_page(page_b, tmp_path)
    out_c = write_page(page_c, tmp_path)
    disc_b = hashlib.sha1(b"src-b").hexdigest()[:6]
    disc_c = hashlib.sha1(b"src-c").hexdigest()[:6]
    assert out_a.name == "same.md"
    assert out_b.name == f"same-{disc_b}.md"
    assert out_c.name == f"same-{disc_c}.md"
    assert disc_b != disc_c


def test_disambiguator_deterministic(tmp_path: Path) -> None:
    """Two disambiguation runs over the same source_id yield the same slug."""
    page_first = _make_page("Conflict", "src-first")
    page_second = _make_page("Conflict", "src-second")
    write_page(page_first, tmp_path)
    out_b1 = write_page(page_second, tmp_path)
    out_b2 = write_page(page_second, tmp_path)
    assert out_b1.name == out_b2.name


def test_helper_returns_base_when_no_existing(tmp_path: Path) -> None:
    page = _make_page("Fresh", "src-fresh")
    base = slugify(page.title)
    assert _disambiguate_slug(base, page, tmp_path) == base


def test_helper_returns_base_for_same_source_existing(tmp_path: Path) -> None:
    page = _make_page("Same Source", "src-same")
    write_page(page, tmp_path)
    base = slugify(page.title)
    assert _disambiguate_slug(base, page, tmp_path) == base
