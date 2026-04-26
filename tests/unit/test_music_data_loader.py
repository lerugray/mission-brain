"""Tests for :class:`mission_brain.loaders.music_data.MusicDataLoader`.

Covers rayb-013-c acceptance:

* discover / discover_catalog return expected files
* per-song load() yields a SourceDocument with ``source_type="music_data"``
  and ``provenance="primary"``, non-empty body, stable content hash
* catalog aggregate load() renders a distinct keyed summary body
* render-twice determinism (§2.2 foundation)
* content-hash stability across byte-identical reads
* locator_for raises NotImplementedError per task scope
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mission_brain.loaders import MusicDataLoader, SourceDocument

FIXTURES = (
    Path(__file__).resolve().parent.parent
    / "integration"
    / "fixtures"
    / "mini-corpus"
    / "music"
)


@pytest.fixture
def loader() -> MusicDataLoader:
    return MusicDataLoader()


def test_discover_returns_per_song_files_sorted(loader: MusicDataLoader) -> None:
    corpus_root = FIXTURES.parent
    found = list(loader.discover(corpus_root))
    assert len(found) == 1
    assert found[0].name == "fake-song.json"
    assert found == sorted(found)


def test_discover_returns_empty_for_missing_per_song_dir(
    loader: MusicDataLoader, tmp_path: Path
) -> None:
    assert list(loader.discover(tmp_path)) == []


def test_discover_catalog_returns_aggregate_when_present(
    loader: MusicDataLoader,
) -> None:
    corpus_root = FIXTURES.parent
    found = list(loader.discover_catalog(corpus_root))
    assert len(found) == 1
    assert found[0].name == "catalog_analysis.json"


def test_discover_catalog_empty_when_missing(
    loader: MusicDataLoader, tmp_path: Path
) -> None:
    assert list(loader.discover_catalog(tmp_path)) == []


def test_per_song_load_shape(loader: MusicDataLoader) -> None:
    path = FIXTURES / "per-song" / "fake-song.json"
    doc = loader.load(path)
    assert isinstance(doc, SourceDocument)
    assert doc.source_type == "music_data"
    assert doc.provenance == "primary"
    assert doc.title == "Synthetic Fake Song"
    assert doc.body.strip() != ""
    assert doc.content_hash
    assert doc.frontmatter["source_file"] == "Synthetic - Fake Song.mp3"
    assert doc.frontmatter["num_instruments"] == 2
    assert doc.frontmatter["duration_seconds"] == 4.5


def test_per_song_body_contains_time_labels(loader: MusicDataLoader) -> None:
    path = FIXTURES / "per-song" / "fake-song.json"
    doc = loader.load(path)
    # Any note should render as HH:MM:SS.fff-HH:MM:SS.fff in the body.
    assert "00:00:00.000-00:00:00.500" in doc.body  # C4 note
    assert "00:00:00.250-00:00:01.250" in doc.body  # E2 (bass)
    assert "C4" in doc.body
    assert "E2" in doc.body


def test_per_song_notes_sorted_deterministically(loader: MusicDataLoader) -> None:
    path = FIXTURES / "per-song" / "fake-song.json"
    doc = loader.load(path)
    note_lines = [ln for ln in doc.body.splitlines() if ln.startswith("  00:")]
    # First note starts at 0.0 (C4, pitch 60); second at 0.25 (E2, pitch 40);
    # third at 0.5 (E4, pitch 64). Sort key (start, pitch, inst_idx).
    assert note_lines[0].startswith("  00:00:00.000")
    assert "C4" in note_lines[0]
    assert note_lines[1].startswith("  00:00:00.250")
    assert "E2" in note_lines[1]
    assert note_lines[2].startswith("  00:00:00.500")
    assert "E4" in note_lines[2]


def test_per_song_time_index_populated(loader: MusicDataLoader) -> None:
    path = FIXTURES / "per-song" / "fake-song.json"
    doc = loader.load(path)
    time_index = doc.extras["time_index"]
    assert len(time_index) == 3
    # Each entry is (start_sec, end_sec, body_line_1based).
    for start, end, line in time_index:
        assert isinstance(start, float)
        assert isinstance(end, float)
        assert isinstance(line, int)
        assert end >= start
        assert line >= 1
    # Body line indices refer into the rendered body.
    body_lines = doc.body.splitlines()
    for _start, _end, line in time_index:
        assert line <= len(body_lines)
        assert body_lines[line - 1].lstrip().startswith("00:")


def test_catalog_load_shape(loader: MusicDataLoader) -> None:
    path = FIXTURES / "catalog" / "catalog_analysis.json"
    doc = loader.load(path)
    assert doc.source_type == "music_data"
    assert doc.provenance == "primary"
    assert doc.title == "Music Catalog Aggregate"
    # Catalog body cites by top-level JSON keys, not line ranges.
    assert "scale_fingerprint:" in doc.body
    assert "chord_vocabulary:" in doc.body
    assert "top_motifs:" in doc.body
    assert "register_usage:" in doc.body
    assert "rhythm_distribution:" in doc.body
    # Catalog keys surfaced in frontmatter so the ingest prompt can
    # namespace [ref:...:para=<key>] markers.
    assert "scale_fingerprint" in doc.frontmatter["catalog_keys"]


def test_catalog_top_chord_ordering(loader: MusicDataLoader) -> None:
    """Top chords/motifs render sorted by count desc — stable across runs."""
    path = FIXTURES / "catalog" / "catalog_analysis.json"
    doc = loader.load(path)
    lines = doc.body.splitlines()
    # Find chord_vocabulary block and check first entry is the most frequent.
    idx = lines.index("chord_vocabulary:")
    # First indented entry after the heading is the highest-count chord.
    assert "D power chord (5th)" in lines[idx + 1]


def test_content_hash_is_stable_across_reads(loader: MusicDataLoader) -> None:
    path = FIXTURES / "per-song" / "fake-song.json"
    assert loader.load(path).content_hash == loader.load(path).content_hash


def test_content_hash_changes_when_bytes_change(
    loader: MusicDataLoader, tmp_path: Path
) -> None:
    src = FIXTURES / "per-song" / "fake-song.json"
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_bytes(src.read_bytes())
    b.write_bytes(src.read_bytes() + b" ")
    assert loader.load(a).content_hash != loader.load(b).content_hash


def test_body_rendering_is_deterministic(loader: MusicDataLoader) -> None:
    """Calling load() twice on the same bytes yields byte-identical bodies."""
    path = FIXTURES / "per-song" / "fake-song.json"
    first = loader.load(path).body
    second = loader.load(path).body
    assert first == second

    catalog = FIXTURES / "catalog" / "catalog_analysis.json"
    first_c = loader.load(catalog).body
    second_c = loader.load(catalog).body
    assert first_c == second_c


def test_locator_for_raises_not_implemented(loader: MusicDataLoader) -> None:
    with pytest.raises(NotImplementedError):
        loader.locator_for((0, 5))


def test_loader_identity_fields(loader: MusicDataLoader) -> None:
    assert loader.source_type == "music_data"
    assert ".json" in loader.supported_extensions
    assert loader.version == "1.0"
